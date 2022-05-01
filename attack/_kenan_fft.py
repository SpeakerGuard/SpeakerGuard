
'''
Kenan attack with FFT as attack method which supports batch attack
'''

# import speech_recognition as sr
from attack.ssa_core import ssa, inv_ssa
import numpy as np
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#List of comparison terms
df_attack = 'Attack'
df_attack_factor = 'Attack Factor'
df_og_label = 'OG Label'
df_raster_width = 'Raster Width'
df_succ = 'Succ'

# Attacks
floor_atk_name = 'floor'
dct_atk_name = 'dct'
fft_atk_name = 'fft'
dct_base_atk_name = 'dct_base'
svd_atk_name = 'svd'
ssa_atk_name = 'ssa'
pca_atk_name = 'pca'
sin_atk_name = 'sin'


def normalize(data):
    normalized = data.ravel()*1.0/np.amax(np.abs(data.ravel()))
    magnitude = np.abs(normalized)
    return magnitude

# MSE between audio samples
def diff_avg(audio1,audio2):
    # Normalize
    n_audio1 = normalize(audio1)
    n_audio2 = normalize(audio2)
    
    # Diff
    diff = n_audio1 - n_audio2
    abs_diff = np.abs(diff)
    overall_change = sum(abs_diff)
    average_change = overall_change/len(audio1)
    return average_change

# L2 difference between audio samples
def diff_l2(audio1,audio2):
    # Normalize
    n_audio1 = normalize(audio1)
    n_audio2 = normalize(audio2)
    l2 = np.linalg.norm(n_audio2-n_audio1,2)
    return l2


def fft_compression(audio_image,factor,fs):
    '''
    # DFT Attack
    # path: path to audio file
    # Audio_image: audio file as an np.array object
    # factor: the intensity below which you want to zero out
    # fs: sample rate
    '''
    # Take FFT
    # fft_image = sc.fftpack.fft(audio_image.ravel())
    # fft_image = torch.fft.fft(audio_image, dim=2) # (N, 1, len)
    fft_image = torch.fft.rfft(audio_image, dim=2) # (N, 1, len)
    
    # Zero out values below threshold
    # print('FFT Factor: ',factor)
    fft_image[fft_image.abs() < factor.unsqueeze(1).unsqueeze(2)] = 0
    
    # inverse fft
    # ifft_audio = sc.fftpack.ifft(fft_image).real
    # ifft_audio = torch.fft.ifft(fft_image, dim=2).real
    ifft_audio = torch.fft.irfft(fft_image, dim=2)
    
    # New file name
    # new_audio_path = path[0:-4]+'_'+str(fs)+'_FFT_'+str(factor)+'.wav'
    # return new_audio_path, np.asarray(ifft_audio,dtype=np.int16)
    return ifft_audio


def ssa_compression(audio_image,factor,fs,percent = True, pc=None,v=None):
    '''
    # SSA Attack
    # path: path to audio file
    # Audio_image: audio file as an np.array object
    # factor: the total percent of the lowest SSA componenets you want to discard
    # pc: first element that the ssa(data, window). Pass it to make execution fase
    # v: third element that the ssa(data, window). Pass it to make execution fase
    # fs: sample rate
    '''
    data = audio_image.ravel()
    window = int(len(data)*0.05)
    # print('Factor Initial: '+str(factor))
    if(window>3000):
        window = 3000
    if(percent):
        factor = int((window)*factor/100)
    factor = 1 if(factor == 0) else int(factor)
    # print('Factor Percent: '+str(factor))
    # if type(pc) is not np.ndarray:
    if type(pc) is not torch.Tensor:
        pc, s, v = ssa(data, window)
    # print('Factor for K: '+str(factor))
    reconstructed = inv_ssa(pc, v, np.arange(0,factor,1))
    # new_audio_path = path[0:-4]+'_SSA_'+str(factor)+'.wav'
    # return new_audio_path, np.asarray(reconstructed,dtype=np.int16).ravel(),pc,v
    return np.asarray(reconstructed,dtype=np.int16).ravel(), pc, v

def perturb(
            audio,
            atk_name,
            fs, 
            factor,
            raster_width,
            pc=None,
            v=None,
         ):
    frame = audio  
    if(atk_name == ssa_atk_name):
        return ssa_compression(frame,factor,fs,pc = pc,v =v)

    elif(atk_name == fft_atk_name):
        perturbed_frame= fft_compression(frame,factor,fs)
        return perturbed_frame

    # return perturbed_frame.ravel()

# def bst_atk_factor(min_atk,max_atk,val_atk,atk_name,og_label,atk_label):
def bst_atk_factor(min_atk,max_atk,val_atk,atk_name,og_label,atk_label, length=None, percent=True):
    '''
    # For searching the best attack factor using binary search
    # For DCT, decrease factor if evasion success, increase other wise
    # For SSA, SVD and PCA, increase factor if evasion success, decrease other wise
    '''
    if(atk_label == og_label):
        succ = False
    else:
        succ = True
    
    init_val_atk = val_atk
    if(atk_name == dct_atk_name or atk_name == fft_atk_name or atk_name == floor_atk_name):
        if(succ):
            max_atk = val_atk
            val_atk = np.abs(min_atk+max_atk)/2
            
        else:
            min_atk = val_atk
            val_atk = np.abs(min_atk+max_atk)/2
            
    elif(atk_name == pca_atk_name or atk_name == svd_atk_name or atk_name == ssa_atk_name):
        if(succ):
            min_atk = val_atk
            val_atk = np.abs(min_atk+max_atk)/2
            
        else:
            max_atk = val_atk
            val_atk = np.abs(min_atk+max_atk)/2
            
    if atk_name != ssa_atk_name:
        # return int(min_atk),int(max_atk),int(val_atk),(init_val_atk==val_atk) 
        return min_atk, max_atk, val_atk, (init_val_atk == val_atk) 
    else:
        window = int(length*0.05)
        # print('Factor Initial: '+str(factor))
        if(window>3000):
            window = 3000
        if(percent):
            init_factor = int((window)*init_val_atk/100)
            init_factor = 1 if(init_factor == 0) else int(init_factor)

            factor = int((window)*val_atk/100)
            factor = 1 if(factor == 0) else int(factor)
            # print(init_val_atk, val_atk, init_factor, factor, init_factor == factor)
            return min_atk, max_atk, val_atk, init_factor == factor

def atk_bst_fft(data, fs, og_label, targeted, raster_width, models, attack, max_allowed_iterations, verbose, early_stop):

    n_audios, n_channels, max_len = data.shape
    device = data.device
        
    _raster_width = raster_width[0]
    _model = models[0]
    _attack_name = attack[0]

    # perturbed_audio = data.clone()
    mistranscribed_audio = data.clone()

    # Need the min var for BST
    min_attack_factor = torch.tensor([0.] * n_audios).to(device)

    # Max factor for perturbation
    max_attack_factor = torch.tensor([_raster_width] * n_audios).to(device) if (_attack_name != dct_atk_name) else torch.tensor([8000] * n_audios).to(device) 
    # max_attack_factor = max(abs(fftpack.fft(data))) if (_attack_name == fft_atk_name) else max_attack_factor
    max_attack_factor = torch.fft.fft(data, dim=2).abs().max(dim=2)[0].view(-1) if (_attack_name == fft_atk_name) else max_attack_factor
    
    _attack_factor= max_attack_factor / 2

    # For ssa
    pc = v = None

    # Initialize iteration counter
    itr = 0
    succ = [False] * n_audios

    while(itr < max_allowed_iterations):

        atk_result = perturb(
            audio = data.clone(),
            atk_name = _attack_name,
            fs = fs, 
            factor = _attack_factor,
            raster_width = _raster_width,
                             pc = pc,
                             v = v,
         )
        # Recycling pc and v to reduce computation time
        if(_attack_name == ssa_atk_name):
            perturbed_audio, pc,v = atk_result
        else:
            perturbed_audio = atk_result

        transcribed_perturbation, scores_p = _model.make_decision(perturbed_audio)

        if verbose:
            print('Iter: {} ori: {} atk: {} f: {}'.format(itr+1, og_label, transcribed_perturbation, _attack_factor))

        for ppp in range(n_audios):
            if (og_label[ppp] != transcribed_perturbation[ppp] and not targeted) or (og_label[ppp] == transcribed_perturbation[ppp] and targeted):
                # mistranscribed_audio = perturbed_audio # error
                mistranscribed_audio[ppp, ...] = perturbed_audio[ppp, ...]

                max_attack_factor[ppp] = _attack_factor[ppp]
                _attack_factor[ppp] = ((min_attack_factor[ppp] + max_attack_factor[ppp]) / 2).abs()
                succ[ppp] = True
            else:
                min_attack_factor[ppp] = _attack_factor[ppp]
                _attack_factor[ppp] = ((min_attack_factor[ppp] + max_attack_factor[ppp]) / 2).abs()

        itr = itr + 1

    return mistranscribed_audio, succ