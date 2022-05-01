

'''
Part of the code is drawn from https://github.com/kwarren9413/kenansville_attack
I made some modifications to make it work for speake recognition and compatible with SpeakerGuard Library
'''

from attack.ssa_core import ssa, inv_ssa
import numpy as np

import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from scipy import fftpack

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
    fft_image = fftpack.fft(audio_image.ravel())
    
    # Zero out values below threshold
    # print('FFT Factor: ',factor)
    fft_image[abs(fft_image) < factor] = 0
    
    # inverse fft
    # ifft_audio = sc.fftpack.ifft(fft_image).real
    ifft_audio = fftpack.ifft(fft_image).real
    
    # New file name
    # new_audio_path = path[0:-4]+'_'+str(fs)+'_FFT_'+str(factor)+'.wav'
    # return new_audio_path, np.asarray(ifft_audio,dtype=np.int16)
    return np.asarray(ifft_audio,dtype=np.int16)


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

    return perturbed_frame.ravel()

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

# def atk_bst(data, fs, og_label, targeted, raster_width, models, attack, max_allowed_iterations, verbose):
def atk_bst(data, fs, og_label, targeted, raster_width, models, attack, max_allowed_iterations, verbose, early_stop):

        
    _raster_width = raster_width[0]
    _model = models[0]
    _attack_name = attack[0]

    # Some times audio samples have two tracks
    # just get one track
    data = data[:,0] if(len(data.shape) != 1) else data

    # Copy data to a new mutable variables
    perturbed_audio = np.copy(data)
    mistranscribed_audio = np.copy(data)
    # This is the frame we will be attacking
    frame_to_perturb = data

    # Need the min var for BST
    min_attack_factor = 0

    # Max factor for perturbation
    max_attack_factor = _raster_width if (_attack_name != dct_atk_name) else 8000
    # max_attack_factor = max(abs(sc.fftpack.fft(data))) if (_attack_name == fft_atk_name) else max_attack_factor
    max_attack_factor = max(abs(fftpack.fft(data))) if (_attack_name == fft_atk_name) else max_attack_factor
    _attack_factor= max_attack_factor/2

    # For ssa
    pc = v = None

    # Initialize iteration counter
    itr = 0

    # For each iteration
    # Generate attack sample using the _attack_factor
    # if attack works reduce max_attack_factor and min_attack_factor
    # if attack does not work increase max_attack_factor and min_attack_factor
    # Save only the attack file that works to the dataframe

    succ = False
    f_label_1 = None
    scores_1 = None

    while(itr < max_allowed_iterations):
        # This variable is written to the dataframe to show the last iteration
        bst = False
        _window_size = 100
        # Attack!!
        atk_result = perturb(
            audio = frame_to_perturb,
            atk_name = _attack_name,
            fs = fs, 
            factor = _attack_factor,
            raster_width = _raster_width,
                             pc = pc,
                             v = v,
         )
        # Recycling pc and v to reduce computation time
        if(_attack_name == ssa_atk_name):
            perturbed_audio_frame, pc,v = atk_result
        else:
            perturbed_audio_frame = atk_result


        # perturbed_audio_path = perturbed_audio_path[:-4]+'_BST.wav'
        perturbed_audio[0:len(perturbed_audio_frame)] = \
        perturbed_audio_frame
        # print(perturbed_audio_frame.shape, perturbed_audio_frame.dtype)

        transcribed_perturbation, scores_p = _model.make_decision(torch.from_numpy(perturbed_audio.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(1))
        transcribed_perturbation = transcribed_perturbation.item()
                         
        # # Delete newly created audio
        # del_tmp = 'rm '+ perturbed_audio_path
        # system(del_tmp)

        if(og_label != transcribed_perturbation and not targeted) or (og_label == transcribed_perturbation and targeted):
            # mistranscribed_audio = perturbed_audio # error in the original repo since without copy, perturbed_audio and mistranscribed_audio will share the same memory
            mistranscribed_audio = np.copy(perturbed_audio)
            succ = True
            f_label_1 = transcribed_perturbation
            scores_1 = scores_p
    
        # Adjust max and min factor varaibles
        # new_min_attack_factor,new_max_attack_factor,new_attack_factor,complete = \
        # bst_atk_factor(min_atk = min_attack_factor ,max_atk = \
        #                max_attack_factor ,val_atk = _attack_factor \
        #                ,atk_name = _attack_name ,og_label = og_label ,atk_label = transcribed_perturbation)
        new_min_attack_factor,new_max_attack_factor,new_attack_factor,complete = \
        bst_atk_factor(min_atk = min_attack_factor ,max_atk = \
                       max_attack_factor ,val_atk = _attack_factor \
                       ,atk_name = _attack_name ,og_label = og_label ,atk_label = transcribed_perturbation, length=len(data), percent=True)


        min_attack_factor,max_attack_factor,_attack_factor = \
            new_min_attack_factor,new_max_attack_factor,new_attack_factor 

        # Distances between original and perturbe audio file
        l2 = diff_l2(data,perturbed_audio)
        avg = diff_avg(data,perturbed_audio)
        if transcribed_perturbation is None:
            transcribed_perturbation = 'None'
        if verbose:
            print('Iter: {} ori: {} atk: {} l2:{} avg: {}'.format(itr+1, og_label, transcribed_perturbation, round(l2, 3), round(avg, 3)))

        itr = itr + 1
        if(complete): break

    f_label, scores_f = _model.make_decision(torch.from_numpy(mistranscribed_audio.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(1))
    succ = ((f_label != og_label and not targeted) or (f_label == og_label and targeted))
    # print(f_label_1, f_label, scores_1, scores_f)
    return mistranscribed_audio.ravel(), succ
