from dataset.Dataset import Dataset
import os

class Spk251_train(Dataset):

    def __init__(self, spk_ids, root, return_file_name=False, wav_length=None):
        normalize = True
        bits = 16
        super().__init__(spk_ids, root, 'Spk251_train',
                        normalize=normalize, bits=bits, 
                        return_file_name=return_file_name, wav_length=wav_length)

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    root = './dataset'
    spk_ids = os.listdir('./dataset/Spk251_train')
    dataset = Spk251_train(spk_ids, root, return_file_name=True, wav_length=80_000)
    data_loader = DataLoader(dataset, batch_size=128, num_workers=8)

    for x, y, name in data_loader:
        print(x.shape, y, name)