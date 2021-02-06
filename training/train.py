from functools import partial
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
import os.path as osp
from tqdm.notebook import tqdm
import datetime


def masked_mse(y_true, y_pred, valid_mask):
    valid_keypoints_mask = tf.reshape(valid_mask, (tf.shape(valid_mask)[0], 1, 1, tf.shape(valid_mask)[1]))
    heatmap_loss = tf.square(y_pred - y_true)
    # 100 is roughly the ratio between high value points in the heatmap and zero points
    heatmap_loss_weight = 1 + (tf.cast((y_true > 0.3), tf.float32) * 100)
    return tf.reduce_mean(heatmap_loss * heatmap_loss_weight * valid_keypoints_mask)


def train_step_python(x, heatmap_gt, valid_mask,
                      model, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        heatmaps_out = model(x, training=True)
        loss_value = loss_fn(heatmap_gt, heatmaps_out, valid_mask)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


def test_step_python(x, heatmap_gt, valid_mask,
                     model, loss_fn):
    heatmaps_out = model(x, training=False)
    loss_value = loss_fn(heatmap_gt, heatmaps_out, valid_mask)
    return loss_value


class Evaluator(object):
    def __init__(self, model, val_dataset):
        self.val_loss = metrics.Mean()
        self.test_step = tf.function(partial(test_step_python,
                                             model=model, loss_fn=masked_mse))
        self.val_dataset = iter(val_dataset.repeat())

    def eval(self, val_steps=1):
        if val_steps > 1:
            self.val_loss.reset_states()
        for _ in range(val_steps):
            x, y, mask = self.val_dataset.next()
            loss = self.test_step(x, y, mask)
            if val_steps > 1:
                self.val_loss.update_state(loss)
            else:
                return loss.numpy()
        return self.val_loss.result().numpy()


class EMAValue:
    # EMA(current) = current x multiplier + EMA(old) x (1-multiplier)
    def __init__(self, alpha=0.9):
        self.ema = None
        self.alpha = alpha

    def update(self, value):
        if self.ema is None:
            self.ema = value
        else:
            self.ema = value * (1 - self.alpha) + self.ema * self.alpha


class TensorboardWriter(object):
    def __init__(self, cfg):
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = osp.join(cfg['path'], 'logs', self.current_time + '_train')
        self.test_log_dir = osp.join(cfg['path'], 'logs', self.current_time + '_test')
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)

    def update_train_plot(self, step, training_loss):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', training_loss, step=step)

    def update_val_plot(self, step, val_loss):
        with self.test_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss, step=step)


def train(cfg,
          model,
          train_dataset, train_dataset_len,
          val_dataset,
          epochs,
          profiling=False,
          restore_only=False,
          full_model_training=False):
    # learning rate schedule and optimizer
    optimizer = optimizers.Adam(cfg['lr'])

    # model unfreezing handling
    best_train_loss = 1000
    best_train_step = 0

    # model saving after each epoch
    best_val_loss = 1000

    # training early stopping
    best_val_epoch = 0

    # checkpointing init
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model, iterator=train_dataset)
    manager = tf.train.CheckpointManager(ckpt, cfg['ckpt_dir'],
                                         max_to_keep=5,
                                         keep_checkpoint_every_n_hours=1)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    if restore_only:
        return

    # ########################################
    if full_model_training:
        model.trainable = True
        optimizer.lr.assign(cfg['lr_full_model'])
    #######################################
    # print(optimizer.learning_rate.boundaries)
    # training init
    training_ema = EMAValue(cfg['alpha_train_ema'])
    val_ema = EMAValue(cfg['alpha_val_ema'])
    train_step = tf.function(partial(train_step_python,
                                     model=model, optimizer=optimizer, loss_fn=masked_mse))
    tboard = TensorboardWriter(cfg)
    evaluator = Evaluator(model, val_dataset)
    val_loss = evaluator.eval()  # init val_loss
    # training
    for epoch in range(1, epochs + 1):
        # Iterate over the batches of the dataset.
        progress_bar = tqdm(enumerate(train_dataset), total=train_dataset_len)
        progress_bar.set_description(f'Epoch {epoch}')
        for step, (x, y, mask) in progress_bar:
            # profiling
            if profiling:
                if ckpt.step == 100:
                    print('Profiling model...')
                    tf.profiler.experimental.start('./logs/')
                if ckpt.step == 150:
                    print('Stopping profiling...')
                    tf.profiler.experimental.stop()

            ckpt.step.assign_add(1)
            training_loss = train_step(x, y, mask)
            train_loss = training_loss.numpy()
            training_ema.update(train_loss)

            # validation
            if cfg['val_interval'] is not None:
                if step % cfg['val_interval'] == 0:
                    val_loss = evaluator.eval()
                    val_ema.update(val_loss)

            # model unfreezing
            if val_ema.ema is not None and best_train_loss > val_ema.ema:
                if step > 10:
                    best_train_loss = val_ema.ema
                    best_train_step = ckpt.step.value()
                    assert not isinstance(best_train_step, tf.Variable)
            elif ckpt.step - best_train_step > cfg['unfreeze_delay']:
                model.trainable = True
                print('THE MODEL HAS BEEN UNFROZEN')
                # I have to recreate the tf.function since I need new variables (cannot create variables after first
                # call)
                optimizer.lr.assign(cfg['lr_full_model'])
                train_step = tf.function(partial(train_step_python,
                                                 model=model, optimizer=optimizer, loss_fn=masked_mse))
                evaluator = Evaluator(model, val_dataset)

            # checkpointing
            if step > 0 and int(ckpt.step) % cfg['ckpt_interval'] == 0:
                manager.save()

            # Log every k batches.
            if step % cfg['log_interval'] == 0:
                progress_bar.set_postfix(loss=training_ema.ema,
                                         val_loss=val_ema.ema,
                                         best_train_loss=best_train_loss,
                                         best_val_loss=best_val_loss)
                tboard.update_train_plot(int(ckpt.step.numpy()), training_loss)
                tboard.update_val_plot(int(ckpt.step.numpy()), val_loss)

        # val_loss = eval(model, val_dataset)
        # progress_bar.set_postfix(val_loss=val_loss)
        # tboard.update_val_plot(epoch*step, val_loss)
        progress_bar.close()

        # saving at the end of the epoch
        if best_val_loss > val_ema.ema:
            best_val_loss = val_ema.ema
            model.save(osp.join(cfg['model_savepath']))
            best_val_epoch = epoch
        elif epoch - best_val_epoch >= cfg['es_epoch_delay']:
            print('Training Complete')
            break
