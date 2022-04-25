
from adaptive_attack.BPDA import BPDA
import numpy as np
import torch
import math
import torch
import torch.nn.functional as F


def QT_Non_Diff(audio, param=128, bits=16, same_size=True):

    assert torch.is_tensor(audio) == True
    ori_shape = audio.shape
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0) # (T, ) --> (1, T)
    elif len(audio.shape) == 2: # (B, T)
        pass
    elif len(audio.shape) == 3:
        audio = audio.squeeze(1) # (B, 1, T) --> (B, T)
    else:
        raise NotImplementedError('Audio Shape Error')

    max = 2 ** (bits-1) - 1
    min = -1. * 2 ** (bits-1)
    abs_max = abs(min)
    scale = False
    lower = -1 
    upper = 1
    # print('QT-1:', audio.max(), audio.min())
    # if audio.min() >= 2 * lower and audio.max() <= 2 * upper: # 2*lower and 2*upper due to floating point issue, e.g., sometimes will have 1.0002
    if 0.9 * audio.max() <= upper and 0.9 * audio.min() >= lower:
        audio = audio * abs_max
        scale = True
    # print('QT-2:', audio.max(), audio.min())
    q = param
    audio_q = torch.round(audio / q) * q # round operation makes it non-differentiable
    # print('QT-3:', audio_q.max(), audio_q.min())

    if scale:
        audio_q.data /= abs_max
    
    return audio_q.view(ori_shape)

QT = BPDA(QT_Non_Diff, lambda *args: args[0]) # BPDA wrapper, make it differentiable

def BDR(audio, param=8, bits=16, same_size=True):
    q = 2 ** (bits - param)
    return QT(audio, param=q, bits=bits, same_size=same_size)

def AT(audio, param=25, same_size=True):

    assert torch.is_tensor(audio) == True
    ori_shape = audio.shape
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0) # (T, ) --> (1, T)
    elif len(audio.shape) == 2: # (B, T)
        pass
    elif len(audio.shape) == 3:
        audio = audio.squeeze(1) # (B, 1, T) --> (B, T)
    else:
        raise NotImplementedError('Audio Shape Error')

    snr = param
    snr = 10 ** (snr / 10)
    batch, N = audio.shape
    power_audio = torch.sum((audio / math.sqrt(N)) ** 2, dim=1, keepdims=True) # (batch, 1)
    power_noise = power_audio / snr # (batch, 1)
    noise = torch.randn((batch, N), device=audio.device) * torch.sqrt(power_noise) # (batch, N)
    noised_audio = audio + noise
    return noised_audio.view(ori_shape)

def AS(audio, param=3, same_size=True):

    assert torch.is_tensor(audio) == True
    ori_shape = audio.shape
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0) # (T, ) --> (1, T)
    elif len(audio.shape) == 2: # (B, T)
        pass
    elif len(audio.shape) == 3:
        audio = audio.squeeze(1) # (B, 1, T) --> (B, T)
    else:
        raise NotImplementedError('Audio Shape Error')

    batch, _ = audio.shape

    kernel_size = param
    assert kernel_size % 2 == 1
    audio = audio.view(batch, 1, -1) # (batch, in_channel:1, max_len)

    ################# Using torch.nn.functional ###################
    kernel_weights = np.ones(kernel_size) / kernel_size
    weight = torch.tensor(kernel_weights, dtype=torch.float, device=audio.device).view(1, 1, -1) # (out_channel:1, in_channel:1, kernel_size)
    output = F.conv1d(audio, weight, padding=(kernel_size-1)//2) # (batch, 1, max_len)
    ###############################################################

    return output.squeeze(1).view(ori_shape) # (batch, max_len)


def MS(audio, param=3, same_size=True):
    r"""
    Apply median smoothing to the 1D tensor over the given window.
    """

    assert torch.is_tensor(audio) == True
    ori_shape = audio.shape
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0) # (T, ) --> (1, T)
    elif len(audio.shape) == 2: # (B, T)
        pass
    elif len(audio.shape) == 3:
        audio = audio.squeeze(1) # (B, 1, T) --> (B, T)
    else:
        raise NotImplementedError('Audio Shape Error')

    win_length = param
    # Centered windowed
    pad_length = (win_length - 1) // 2

    # "replicate" padding in any dimension
    audio = F.pad(audio, (pad_length, pad_length), mode="constant", value=0.)

    # indices[..., :pad_length] = torch.cat(pad_length * [indices[..., pad_length].unsqueeze(-1)], dim=-1)
    roll = audio.unfold(-1, win_length, 1)

    values, _ = torch.median(roll, -1)
    return values.view(ori_shape)
