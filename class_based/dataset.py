import numpy as np

class dataset:

    def __init__(self, data_path):

        # data I/O
        # should be simple plain text file. The sample from "Hamlet - Shakespeares" is provided in data/
        data = open(data_path, 'r').read()
        chars = sorted(list(set(data)))  # added sorted so that the character list is deterministic
        print(chars)
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        self.char_to_ix = {ch: i for i, ch in enumerate(chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(chars)}

        # this will load the data into memory
        self.data_stream = np.asarray([self.char_to_ix[char] for char in data])
        print(self.data_stream.shape)
        data.close()


    def get_cut_stream(self,seq_length, batch_size):
        bound = (self.data_stream.shape[0] // (seq_length * batch_size)) * (seq_length * batch_size)
        cut_stream = self.data_stream[:bound]
        cut_stream = np.reshape(cut_stream, (batch_size, -1))

        return cut_stream

    def get_ix_to_char(self):
        return self.ix_to_char

    def get_char_to_ix(self):
        return self.char_to_ix