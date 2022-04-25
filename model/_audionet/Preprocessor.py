''' Code drawn from
https://github.com/usc-sail/gard-adversarial-speaker-id
Paper: 
Jati et al. Adversarial attack and defense strategies for deep speaker recognition systems
'''

import numpy as np
import librosa

import torch
from torch import nn

class Hparams():
    def __init__(self):
        self.sr = 16_000
        self.n_mels = 32
        self.n_fft = 1024
        self.hop_length = 160
        self.win_length = 800
        self.preemphasis = 0.97
        self.fmin = 0
        self.fmax = self.sr // 2
hp = Hparams()

EPSILON = 1e-16


def _square(x):
    """
    Torch represents Complex Number using 2 channels,
    so the square of a complex number has to be implemented.
    Inputs:
        x: [..., 2]
    Ouptuts:
        y: [...]
    """
    return x.pow(2).sum(-1)  # [b, F=1025, T]


def mel_to_linear_matrix(mel_basis, sr, n_fft, n_mels):
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels)
    m_t = np.transpose(mel_basis)
    p = np.matmul(mel_basis, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))


class Preprocessor(nn.Module):
    """ Follow the Tacotron repo but using `torch` ops instead of `librosa`
    NOTE: 
        - torchaudio's default `MelSpectrogram` has different setting than `librosa`
        - We wrap the matrices with `nn.Parameter` so that they `.to(device)` works on them. 
    """

    def __init__(self, augmentation=False):
        super().__init__()
        mel_basis = librosa.filters.mel(
            hp.sr, hp.n_fft, hp.n_mels,
            fmin=hp.fmin, fmax=hp.fmax)
        if hp.n_fft // 4 == hp.n_mels:
            inverse_mel = mel_to_linear_matrix(
                mel_basis, hp.sr, hp.n_fft, hp.n_mels)
        else:
            inverse_mel = np.linalg.pinv(mel_basis)

        self._inverse_mel = torch.nn.Parameter(
            torch.from_numpy(inverse_mel).float(),
            requires_grad=False
        )
        self.mel_basis = torch.nn.Parameter(
            torch.tensor(mel_basis.T).float(),
            requires_grad=False
        )
        self.augmentation = augmentation

    def _convert_to_mel_frequency(self, mag_spec):
        """
        Input: linear magnitude spectrogram. [b, F, T] (float32)
        """
        mag_spec = mag_spec.transpose(2, 1)
        mel = torch.matmul(mag_spec, self.mel_basis)
        mel = mel.transpose(2, 1).contiguous()
        return mel

    def preemphasize(self, wav, coeff):
        return wav[:, 1:] - coeff * wav[:, :-1]

    def forward(self, wav):
        """ 
        Input: [b, T] `int16`
        output: [b, F, t], values are in the range of [0, 1]
        """
        # if wav.dtype == torch.int16:
        #     wav = int16_to_float32(wav)
        # elif wav.dtype is not torch.float32:
        #     raise TypeError(f"Unsupported type: {wav.dtype}")

        wav = self.preemphasize(wav, hp.preemphasis)

        spec = torch.stft(
            wav,
            n_fft=hp.n_fft,
            hop_length=hp.hop_length,
            win_length=hp.win_length,
            window=torch.hann_window(hp.win_length).to(wav.device)) 

        # can't use abs because `torch` use 2 channels to represent complex number
        mag_spec = _square(spec)
        mel = self._convert_to_mel_frequency(mag_spec)

        mel = 10 * torch.clamp(mel, EPSILON).log10()
        return mel