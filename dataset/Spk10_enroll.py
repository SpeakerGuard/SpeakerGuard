from dataset.Dataset import Dataset
import os

class Spk10_enroll(Dataset):

    def __init__(self, spk_ids, root, return_file_name=False, wav_length=None):
        normalize = False
        bits = 16
        super().__init__(spk_ids, root, 'Spk10_enroll',
                        normalize=normalize, bits=bits, 
                        return_file_name=return_file_name, wav_length=wav_length)

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    root = '/p300/xx/Defense/iv/data'
    spk_ids = os.listdir('/p300/xx/Defense/iv/data/enrollment-set-10/')
    dataset = Spk10_enroll(spk_ids, root, return_file_name=True, wav_length=80_000)
    data_loader = DataLoader(dataset, batch_size=2)

    for x, y, name in data_loader:
        print(x.shape, y, name)


