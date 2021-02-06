import os


class CustomTarFileReader:
    def __init__(self, filenames):
        self.filenames = filenames
        self.current_file_idx = 0
        self.current_file = open(self.filenames[self.current_file_idx], mode='rb')
        self.current_pos = 0
        self.file_lengths = []
        for fn in self.filenames:
            self.file_lengths.append(os.stat(fn).st_size)

    def read(self, target_bytes_to_read=-1):
        if target_bytes_to_read == -1:
            print('Asked to read whole file')
            # return read_bytes
        read_bytes = b''
        read_bytes = self.current_file.read(target_bytes_to_read)
        # print('requested', target_bytes_to_read)
        while len(read_bytes) != target_bytes_to_read:
            if self.current_file_idx < len(self.filenames):
                print('EOF on', self.filenames[self.current_file_idx])
                self.current_file.close()
                os.remove(self.filenames[self.current_file_idx])
            else:
                print('All files have ended')
                break
            self.current_file_idx += 1
            if self.current_file_idx < len(self.filenames):
                # print('using', self.filenames[self.current_file_idx])
                self.current_file = open(self.filenames[self.current_file_idx], mode='rb')
                bytes_to_read = target_bytes_to_read - len(read_bytes)
                read_bytes += self.current_file.read(bytes_to_read)
                # if self.current_file_idx + 1 != len(self.filenames):
                #     assert len(read_bytes) == bytes_to_read, f'{bytes_to_read}, {bytes2read_left}, {len(read_bytes)}'
        self.current_pos += len(read_bytes)
        return read_bytes

    def seekable(self):
        print('Requested seekable')
        return False

    def seek(self, offset, whence=0):
        if whence != 0:
            print('Requested seek:', offset, whence)
        # current_file_pos = self.current_file.tell()
        seeked_pos = 0
        new_file_index = -1
        while seeked_pos <= offset and new_file_index < len(self.file_lengths):
            new_file_index += 1
            seeked_pos += self.file_lengths[new_file_index]
        seeked_pos -= self.file_lengths[new_file_index]
        if new_file_index < self.current_file_idx:
            raise ValueError('Impossible to read deleted files')
        if new_file_index != self.current_file_idx:
            self.current_file.close()
            print('using', self.filenames[new_file_index])
            self.current_file = open(self.filenames[new_file_index], mode='rb')
        self.current_file_idx = new_file_index
        file_pos = self.current_file.seek(offset - seeked_pos)
        self.current_pos = seeked_pos + file_pos
        return self.current_pos

    def tell(self):
        return self.current_pos