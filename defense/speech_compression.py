
from adaptive_attack.BPDA import BPDA
import numpy as np
import torch
import os
from scipy.io.wavfile import read, write
import torch
import shlex
import subprocess
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def Speech_Compression_Non_Diff(new, lengths, bits_per_sample, 
                                name, param, fs, same_size, 
                                parallel, n_jobs, start_2, debug):
    
    def _worker(start_, end):
        st = time.time()
        for i in range(start_, end):
            origin_audio_path = tmp_dir + "/" + str(i) + ".wav"
            audio = np.clip(new[i][:lengths[i]], min, max).astype(np.int16)
            write(origin_audio_path, fs, audio)
            opus_audio_path = "{}/{}.{}".format(tmp_dir, i, name)
            command = "ffmpeg -i {} -ac 1 -ar {} {} {} -c:a {} {}".format(origin_audio_path, fs, 
                            param[0], param[1], param[2], opus_audio_path)
            args = shlex.split(command)
            if debug:
                p = subprocess.Popen(args)
            else:
                p = subprocess.Popen(args, stderr=subprocess.DEVNULL, 
                                    stdout=subprocess.DEVNULL)
            p.wait()

            pcm_type = "pcm_s16le" if bits_per_sample == 16 else "pcm_s8"
            target_audio_path = tmp_dir + "/" + str(i) + "-target.wav"
            command = "ffmpeg -i {} -ac 1 -ar {} -c:a {} {}".format(opus_audio_path, fs, pcm_type, target_audio_path)
            args = shlex.split(command)
            if debug:
                p = subprocess.Popen(args)
            else:
                p = subprocess.Popen(args, stderr=subprocess.DEVNULL, 
                                    stdout=subprocess.DEVNULL)
            p.wait()

            _, coding_audio = read(target_audio_path)
            if coding_audio.size <= lengths[i] or (coding_audio.size > lengths[i] and not same_size):
                opuseds[i] = list(coding_audio)
            else:
                start = start_2
                if start is None:
                    min_dist = np.infty
                    start = 0
                    for start_candidate in range(0, coding_audio.size - audio.size + 1, 1):
                        dist = np.sum(np.abs(audio / abs_max - coding_audio[start_candidate:start_candidate+audio.size] / abs_max))
                        if dist < min_dist:
                            start = start_candidate
                            min_dist = dist
                opuseds[i] = list(coding_audio[start:start+lengths[i]])
        et = time.time()
    
    if not bits_per_sample in [16, 8]:
            raise NotImplementedError("Currently We Only Support 16 Bit and 8 Bit Quantized Audio, \
                You Need to Modify 'pcm_type' for Other Bit Type")
        
    out_tensor = False
    device = None
    if torch.is_tensor(new):
        device = str(new.device)
        out_tensor = True
        new = new.clone().detach().cpu().numpy()
    
    ori_shape = new.shape
    if len(new.shape) == 1:
        new = new.reshape((1, new.shape[0])) # (T, ) --> (1, T)
    elif len(new.shape) == 2: # (B, T)
        pass
    elif len(new.shape) == 3:
        new = new.reshape((new.shape[0], new.shape[2])) # (B, 1, T) --> (B, T)
    else:
        raise NotImplementedError('Audio Shape Error')
    
    bit_rate = param
    n_audios, max_len = new.shape
    ### indicating the real length of each audio in new
    ### this parameter is only valid in speech coding method since other methods not use loop
    lengths = lengths if lengths else n_audios * [max_len] 
    max = 2 ** (bits_per_sample-1) - 1
    min = -1. * 2 ** (bits_per_sample-1)
    abs_max = abs(min)
    scale = False
    lower = -1
    upper = 1
    # if -1 <= new.max() <= 1:
    if new.min() >= 2 * lower and new.max() <= 2 * upper: # 2*lower and 2*upper due to floating point issue, e.g., sometimes will have 1.0002
        new = new * abs_max
        scale = True

    high = 100000
    while True:
        random_number = np.random.randint(0, high=high + 1) 
        tmp_dir = "{}-Coding-".format(name) + str(random_number)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            break
    opuseds = [0] * n_audios
    
    if not parallel or (parallel and n_jobs == 1) or n_audios == 1:
        _worker(0, n_audios)

    else:
        n_jobs = n_jobs if n_jobs <= n_audios else n_audios
        n_audios_per_job = n_audios // n_jobs
        process_index = []
        for ii in range(n_jobs):
            process_index.append([ii*n_audios_per_job, (ii+1)*n_audios_per_job])
        if n_jobs * n_audios_per_job != n_audios:
            process_index[-1][-1] = n_audios
        futures = set()
        with ThreadPoolExecutor() as executor:
            for job_id in range(n_jobs):
                future = executor.submit(_worker, process_index[job_id][0], process_index[job_id][1])
                futures.add(future)
            for future in as_completed(futures):
                pass

    shutil.rmtree(tmp_dir)
    opuseds = np.array([(x+[0]*(max_len-len(x)))[:max_len] for x in opuseds])
    opuseds = opuseds.reshape(ori_shape)
    if out_tensor:
        opuseds = torch.tensor(opuseds, dtype=torch.float, device=device)
    if scale:
        opuseds.data /= abs_max
    return opuseds

speech_compression = BPDA(Speech_Compression_Non_Diff, lambda *args: args[0])

def OPUS(new, lengths=None, bits_per_sample=16, param=16000, fs=16000, same_size=True, parallel=True, n_jobs=10, debug=False):

    return speech_compression(new, lengths, bits_per_sample, 
            'opus', ['-b:a', param, 'libopus'], 
            fs, same_size, 
            parallel, n_jobs, 69, debug) 


def SPEEX(new, lengths=None, bits_per_sample=16, param=43200, fs=16000, same_size=True, parallel=True, n_jobs=10, debug=False):

    return speech_compression(new, lengths, bits_per_sample, 
            'spx', ['-b:a', param, 'libspeex'], 
            fs, same_size, 
            parallel, n_jobs, None, debug)


def AMR(new, lengths=None, bits_per_sample=16, param=6600, fs=16000, same_size=True, parallel=True, n_jobs=10, debug=False):
    
    if fs == 16000:
        legal_bit_rate = [6600, 8850, 12650, 14250, 15850, 18250, 19850, 23050, 23850]
    elif fs == 8000:
        legal_bit_rate = [4750, 5150, 5900, 6700, 7400, 7950, 10200, 12200]
    else:
        raise NotImplementedError("AMR Compression only support sampling rate 16000 and 8000")
    if not int(param) in legal_bit_rate:
        raise NotImplementedError("%f Not Allowed When fs=%d" % (param, fs))

    return speech_compression(new, lengths, bits_per_sample, 
            'amr', ['-b:a', param, "libvo_amrwbenc" if fs == 16000 else "libopencore_amrnb"], 
            fs, same_size, 
            parallel, n_jobs, None, debug)


def AAC_V(new, lengths=None, bits_per_sample=16, param=5, fs=16000, same_size=True, parallel=True, n_jobs=10, debug=False):

    return speech_compression(new, lengths, bits_per_sample, 
            'aac', ['-vbr', param, 'libfdk_aac'], 
            fs, same_size, 
            parallel, n_jobs, 2048, debug)


def AAC_C(new, lengths=None, bits_per_sample=16, param=20000, fs=16000, same_size=True, parallel=True, n_jobs=10, debug=False):

    return speech_compression(new, lengths, bits_per_sample, 
            'aac', ['-b:a', param, 'libfdk_aac'], 
            fs, same_size, 
            parallel, n_jobs, 2048, debug)


def MP3_V(new, lengths=None, param=9, fs=16000, bits_per_sample=16, same_size=True, parallel=True, n_jobs=10, debug=False):

    return speech_compression(new, lengths, bits_per_sample, 
            'mp3', ['-q:a', param, 'mp3'], 
            fs, same_size, 
            parallel, n_jobs, 0, debug)


def MP3_C(new, lengths=None, param=16000, fs=16000, bits_per_sample=16, same_size=True, parallel=True, n_jobs=10, debug=False):

    return speech_compression(new, lengths, bits_per_sample, 
            'mp3', ['-b:a', param, 'mp3'], 
            fs, same_size,
            parallel, n_jobs, 0, debug)