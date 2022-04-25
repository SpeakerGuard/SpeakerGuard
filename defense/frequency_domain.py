
import torch
import torch
import torchaudio
from scipy import signal
from torch_lfilter import lfilter

def DS(audio, param=0.5, fs=16000, same_size=True):
    
    assert torch.is_tensor(audio) == True
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
    
    down_ratio = param
    new_freq = int(fs * down_ratio)
    resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=new_freq, resampling_method='sinc_interpolation').to(audio.device)
    up_sampler = torchaudio.transforms.Resample(orig_freq=new_freq, new_freq=fs, resampling_method='sinc_interpolation').to(audio.device)
    down_audio = resampler(audio)
    new_audio = up_sampler(down_audio)
    if same_size: ## sometimes the returned audio may have longer size (usually 1 point)
        return new_audio[..., :audio.shape[1]].view(ori_shape)
    else:
        return new_audio.view(ori_shape[:-1] + new_audio.shape[-1:])

def LPF(new, fs=16000, wp=4000, param=8000, gpass=3, gstop=40, same_size=True, bits=16):

    assert torch.is_tensor(new) == True
    ori_shape = new.shape
    if len(new.shape) == 1:
        new = new.unsqueeze(0) # (T, ) --> (1, T)
    elif len(new.shape) == 2: # (B, T)
        pass
    elif len(new.shape) == 3:
        new = new.squeeze(1) # (B, 1, T) --> (B, T)
    else:
        raise NotImplementedError('Audio Shape Error')
    
    if 0.9 * new.max() <= 1 and 0.9 * new.min() >= -1:
        clip_max = 1
        clip_min = -1
    else:
        clip_max = 2 ** (bits - 1) - 1
        clip_min = -2 ** (bits - 1)

    ws = param
    wp = 2 * wp / fs
    ws = 2 * ws / fs
    N, Wn = signal.buttord(wp, ws, gpass, gstop, analog=False, fs=None)
    b, a = signal.butter(N, Wn, btype='low', analog=False, output='ba')
    
    audio = new.T.to("cpu") # torch_lfilter only supports CPU tensor speed up
    a = torch.tensor(a, device="cpu", dtype=torch.float) 
    b = torch.tensor(b, device="cpu", dtype=torch.float)
    new_audio = None
    for ppp in range(audio.shape[1]): # torch_lfilter will give weird results for batch samples when using cpu tensor speed up; so we use naive loop here
        new_audio_ = lfilter(b, a, audio[:, ppp:ppp+1]).T
        if new_audio is None:
            new_audio = new_audio_
        else:
            new_audio = torch.cat((new_audio, new_audio_), dim=0)
    new_audio = new_audio.clamp(clip_min, clip_max)
    return new_audio.to(new.device).view(ori_shape)

def BPF(new, fs=16000, wp=[300, 4000], param=[50, 5000], gpass=3, gstop=40, same_size=True, bits=16):

    assert torch.is_tensor(new) == True
    ori_shape = new.shape
    if len(new.shape) == 1:
        new = new.unsqueeze(0) # (T, ) --> (1, T)
    elif len(new.shape) == 2: # (B, T)
        pass
    elif len(new.shape) == 3:
        new = new.squeeze(1) # (B, 1, T) --> (B, T)
    else:
        raise NotImplementedError('Audio Shape Error')

    if 0.9 * new.max() <= 1 and 0.9 * new.min() >= -1:
        clip_max = 1
        clip_min = -1
        # print(clip_max, clip_min)
    else:
        clip_max = 2 ** (bits - 1) - 1
        clip_min = -2 ** (bits - 1)
    
    ws = param
    wp = [2 * wp_ / fs for wp_ in wp]
    ws = [2 * ws_ / fs for ws_ in ws]
    N, Wn = signal.buttord(wp, ws, gpass, gstop, analog=False, fs=None)
    b, a = signal.butter(N, Wn, btype="bandpass", analog=False, output='ba', fs=None)

    audio = new.T.to("cpu")
    a = torch.tensor(a, device="cpu", dtype=torch.float)
    b = torch.tensor(b, device="cpu", dtype=torch.float)
    
    new_audio = None
    for ppp in range(audio.shape[1]):
        new_audio_ = lfilter(b, a, audio[:, ppp:ppp+1]).T
        if new_audio is None:
            new_audio = new_audio_
        else:
            new_audio = torch.cat((new_audio, new_audio_), dim=0)
    new_audio = new_audio.clamp(clip_min, clip_max)
    
    return new_audio.to(new.device).view(ori_shape)
