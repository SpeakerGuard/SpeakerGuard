from dataset.Dataset import Dataset
import os

class Spk10_imposter(Dataset):

    def __init__(self, spk_ids, root, return_file_name=False, wav_length=None):

        normalize = False
        bits = 16
        super().__init__(spk_ids, root, 'Spk10_imposter',
                        normalize=normalize, bits=bits,
                        return_file_name=return_file_name, wav_length=wav_length)
