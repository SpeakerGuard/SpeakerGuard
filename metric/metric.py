
import numpy as np
from numpy import linalg as LA
from pesq import pesq
from pystoi import stoi

LOWER = -1
UPPER = 1

def preprocess(benign_xx, bits=16):
    benign_xx = benign_xx.detach().cpu().numpy()
    if not LOWER <= benign_xx.max() <= UPPER:
        benign_xx = benign_xx / (2 ** (bits-1))
    return benign_xx.flatten() # one channel

def Lp(benign_xx, adver_xx, p, bits=16):
    benign_xx = preprocess(benign_xx, bits=bits)
    adver_xx = preprocess(adver_xx, bits=bits)
    return LA.norm(adver_xx-benign_xx, p)

def L2(benign_xx, adver_xx, bits=16):
    return Lp(benign_xx, adver_xx, 2, bits=bits)

def L0(benign_xx, adver_xx, bits=16):
    return Lp(benign_xx, adver_xx, 0, bits=bits)

def L1(benign_xx, adver_xx, bits=16):
    return Lp(benign_xx, adver_xx, 1, bits=bits)

def Linf(benign_xx, adver_xx, bits=16):
    return Lp(benign_xx, adver_xx, np.infty, bits=bits)

def SNR(benign_xx, adver_xx, bits=16):
    benign_xx = preprocess(benign_xx, bits=bits)
    adver_xx = preprocess(adver_xx, bits=bits)
    noise = adver_xx - benign_xx
    power_noise = np.sum(noise ** 2)
    if power_noise <= 0.:
        return np.infty
    power_benign = np.sum(benign_xx ** 2)
    snr = 10 * np.log10(power_benign / power_noise)
    return snr 

def PESQ(benign_xx, adver_xx, bits=16):
    benign_xx = preprocess(benign_xx, bits=bits)
    adver_xx = preprocess(adver_xx, bits=bits)
    pesq_value = pesq(16_000, benign_xx, adver_xx, 'wb' if bits == 16 else 'nb')
    return pesq_value

def STOI(benign_xx, adver_xx, fs=16_000, bits=16):
    benign_xx = preprocess(benign_xx, bits=bits)
    adver_xx = preprocess(adver_xx, bits=bits)
    d = stoi(benign_xx, adver_xx, fs, extended=False)
    return d

def get_all_metric(benign_xx, adver_xx, fs=16_000, bits=16):
    return [L2(benign_xx, adver_xx, bits),
            L0(benign_xx, adver_xx, bits),
            L1(benign_xx, adver_xx, bits),
            Linf(benign_xx, adver_xx, bits),
            SNR(benign_xx, adver_xx, bits),
            PESQ(benign_xx, adver_xx, bits),
            STOI(benign_xx, adver_xx, fs, bits)]
