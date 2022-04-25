from dataset.Dataset import Dataset
import os

class Spk251_test(Dataset):

    def __init__(self, spk_ids, root, return_file_name=False, wav_length=None):
        normalize = True
        bits = 16
        super().__init__(spk_ids, root, 'Spk251_test',
                        normalize=normalize, bits=bits,
                        return_file_name=return_file_name, wav_length=wav_length)