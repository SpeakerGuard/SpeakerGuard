
import os
import torch
from torch.utils.data import Dataset as torch_dataset
import torchaudio
import numpy as np

url_prefix = 'https://drive.google.com/uc?id='
url_suffix = '&export=download'
name2url = {

    'Spk10_enroll': '1BBAo64JOahk0F3yBAovnRLZ1NvjwBy7y',
    'Spk10_test': '1WctqJtP5Es74-U7y3cFXqfHi7JkDz6g5',
    'Spk10_imposter': '1f1GULs0aj_Xrw8JRxe6zzvTN3r2nnOf6',
    'Spk251_train': '1iGcMPiPMzcCLI7xKJLwH1L0Ff_95-tmB',
    'Spk251_test': '1rsXzuEyi5Zqd1XAsr1_Op7mC7hqY0tsp',

}

class Dataset(torch_dataset):

    def __init__(self, spk_ids, root, name, normalize=False, bits=16, return_file_name=False, wav_length=None):
        """[summary]

        Parameters
        ----------
        spk_ids : list or tuple. Elements should be str.
            The ids of the speakers to be recognized. Should consistent with model.spk_ids
        root : [type]
            [description]
        normalize : bool, optional
            [description], by default False
        bits : int, optional
            [description], by default 16
        return_file_name : bool, optional
            [description], by default False
        """
        self.spk_ids = spk_ids
        self.root = os.path.join(root, name)
        # not exist, download
        if not os.path.exists(self.root):
            if name not in name2url.keys():
                raise NotImplementedError('No download url for {}'.format(name))
            url = url_prefix + name2url[name] + url_suffix
            download_command = 'gdown {}'.format(url)
            os.system(download_command)
            tar_command = 'tar -xzf {}.tar.gz'.format(name)
            os.system(tar_command)
        spk_iter = os.listdir(self.root)
        self.audio_paths = []
        for spk_id in spk_iter:
            spk_dir = os.path.join(self.root, spk_id)
            audio_iter = os.listdir(spk_dir)
            for audio_name in audio_iter:
                self.audio_paths.append((spk_id, audio_name))

        self.normalize = normalize
        self.bits = bits
        self.return_file_name = return_file_name
        self.wav_length = wav_length
    
    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        spk_id, audio_name = self.audio_paths[idx]
        if spk_id in self.spk_ids:
            spk_label = self.spk_ids.index(spk_id)
        else:
            spk_label = -1 # # -1 means the spk is an imposter (not any enrolled speakers)
        spk_label = torch.tensor(spk_label, dtype=torch.long)
        audio_path = os.path.join(self.root, spk_id, audio_name)
        audio, _ = torchaudio.load(audio_path)
        if not self.normalize:
            audio.data *= (2 ** (self.bits - 1))
        n_channel, audio_len = audio.shape
        if self.wav_length:
            if self.wav_length < audio_len:
                start = np.random.choice(audio_len - self.wav_length + 1)
                audio = audio[..., start:start+self.wav_length]
            elif self.wav_length > audio_len:
                pad_zero = torch.zeros((n_channel, self.wav_length - audio_len), dtype=audio.dtype)
                audio = torch.cat((audio, pad_zero), 1)
        if not self.return_file_name:
            return audio, spk_label
        else:
            return audio, spk_label, os.path.splitext(audio_name)[0] # audio: (n_channel=1, len)
        