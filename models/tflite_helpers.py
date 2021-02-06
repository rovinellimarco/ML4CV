import tensorflow as tf


def to_tflite(model, batched_input_shape, optimizations=[tf.lite.Optimize.DEFAULT]):
    model.input.set_shape(batched_input_shape)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_types = [tf.float32]  # GPU optimization: tf.float16
    # converter.experimental_new_converter = True
    # converter.experimental_new_quantizer = True
    converter.optimizations = optimizations
    return converter.convert()


def run_tflite_model(model_path, original_input_data):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    if type(original_input_data) is not list:
        input_data = [original_input_data]
    else:
        input_data = original_input_data

    output_data = []
    for batch in input_data:
        assert all(
            batch.shape == input_shape), f'Wrong input data shape: model asks for {input_shape}, received {batch.shape}'
        interpreter.set_tensor(input_details[0]['index'], batch)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data.append(interpreter.get_tensor(output_details[0]['index']))

    if type(original_input_data) is not list:
        output_data = output_data[0]
    return output_data
