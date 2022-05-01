

import numpy as np
import torch
from attack._kenan import atk_bst
from attack._kenan_fft import atk_bst_fft

class Kenan(object):

    def __init__(self, model, atk_name='fft', max_iter=15, raster_width=100, early_stop=False, targeted=False, verbose=1, BITS=16, batch_size=1):

        self.model = model # remember to call model.eval()
        self.atk_name = atk_name
        self.max_iter = max_iter
        self.raster_width = raster_width #  for ssa
        self.targeted = targeted
        self.verbose = verbose
        self.BITS = BITS
        self.early_stop = early_stop

        if atk_name == 'ssa': # ssa attack does not support batch attack
            self.batch_size =1
        else:
            self.batch_size = batch_size

    def attack_batch(self, x_batch, y_batch, batch_id, fs=16_000):

        if self.atk_name == 'ssa':
            device = x_batch.device
            shape = x_batch.shape
            x_batch = x_batch.cpu().numpy()
            if 0.9 * x_batch.max() <= 1 and 0.9 * x_batch.min() >= -1:
                x_batch = x_batch * (2 ** (self.BITS-1))
            x_batch = x_batch.astype(np.int16).flatten()

            x_adv, success = atk_bst(x_batch, fs, y_batch.item(), self.targeted, 
                                [self.raster_width], [self.model], [self.atk_name], self.max_iter, self.verbose, self.early_stop)

            # return x_batch, success
            # return torch.from_numpy(x_adv/2**(BITS-1)).to(device).view(shape), [success]
            return x_adv.reshape(1, 1, -1), [success]
        elif self.atk_name == 'fft':
            device = x_batch.device
            shape = x_batch.shape
            # if 0.9 * x_batch.max() <= 1 and 0.9 * x_batch.min() >= -1:
            #     x_batch = x_batch * (2 ** (self.BITS-1))
            #     scale = True
            # else:
            #     scale = False
            x_adv, success = atk_bst_fft(x_batch, fs, y_batch, self.targeted, 
                                [self.raster_width], [self.model], [self.atk_name], self.max_iter, self.verbose, self.early_stop)
            # if scale:
            #     x_adv = x_adv / (2 ** (self.BITS-1))
            # # print(x_batch.shape, x_adv.shape)
            return x_adv, success
            

    def attack(self, x, y, fs=16_000):

        n_audios, n_channels, _ = x.size()
        assert n_channels == 1, 'Only Support Mono Audio'
        assert y.shape[0] == n_audios, 'The number of x and y should be equal' 

        # self.batch_size = 1 # Kenan not supports batch attack
        batch_size = min(self.batch_size, n_audios)
        n_batches = int(np.ceil(n_audios / float(batch_size)))
        for batch_id in range(n_batches):
            x_batch = x[batch_id*batch_size:(batch_id+1)*batch_size] # (batch_size, 1, max_len)
            y_batch = y[batch_id*batch_size:(batch_id+1)*batch_size]
            adver_x_batch, success_batch = self.attack_batch(x_batch, y_batch, batch_id, fs=fs)
            # print(adver_x_batch.shape)
            if batch_id == 0:
                adver_x = adver_x_batch
                success = success_batch
            else:
                if type(adver_x) == torch.Tensor:
                    adver_x = torch.cat((adver_x, adver_x_batch), 0)
                elif type(adver_x) == np.ndarray:
                    adver_x = np.concatenate((adver_x, adver_x_batch), 0)

                success += success_batch
        # print(adver_x.shape)
        return adver_x, success
