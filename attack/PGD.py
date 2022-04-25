

from adaptive_attack.EOT import EOT
from attack.FGSM import FGSM
from attack.utils import resolve_loss
import numpy as np
import torch

class PGD(FGSM):
    
    def __init__(self, model, task='CSI', epsilon=0.002, step_size=0.0004, max_iter=10, num_random_init=0, 
                loss='Entropy', targeted=False,
                batch_size=1, EOT_size=1, EOT_batch_size=1, 
                verbose=1):

        self.model = model # remember to call model.eval()
        self.task = task
        self.epsilon = epsilon
        self.step_size = step_size
        self.max_iter = max_iter
        self.num_random_init = num_random_init
        self.loss_name = loss
        self.targeted = targeted
        self.batch_size = batch_size
        EOT_size = max(1, EOT_size)
        EOT_batch_size = max(1, EOT_batch_size)
        assert EOT_size % EOT_batch_size == 0, 'EOT size should be divisible by EOT batch size'
        self.EOT_size = EOT_size
        self.EOT_batch_size = EOT_batch_size
        self.verbose = verbose

        self.threshold = None
        if self.task in ['SV', 'OSI']:
            self.threshold = self.model.threshold
            print('Running white box attack for {} task, directly using the true threshold {}'.format(self.task, self.threshold))
        self.loss, self.grad_sign = resolve_loss(loss_name=self.loss_name, targeted=self.targeted,
                                    task=self.task, threshold=self.threshold, clip_max=False)
        self.EOT_wrapper = EOT(self.model, self.loss, self.EOT_size, self.EOT_batch_size, True)

    def attack(self, x, y):

        lower = -1
        upper = 1
        assert lower <= x.max() < upper, 'generating adversarial examples should be done in [-1, 1) float domain' 
        n_audios, n_channels, max_len = x.size()
        assert n_channels == 1, 'Only Support Mono Audio'
        assert y.shape[0] == n_audios, 'The number of x and y should be equal' 
        upper = torch.clamp(x+self.epsilon, max=upper)
        lower = torch.clamp(x-self.epsilon, min=lower)

        batch_size = min(self.batch_size, n_audios)
        n_batches = int(np.ceil(n_audios / float(batch_size)))

        x_ori = x.clone()
        best_success_rate = -1
        best_success = None
        best_adver_x = None
        for init in range(max(1, self.num_random_init)):
            if self.num_random_init > 0:
                x = x_ori + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, \
                                (n_audios, n_channels, max_len)), device=x.device, dtype=x.dtype) 
            for batch_id in range(n_batches):
                x_batch = x[batch_id*batch_size:(batch_id+1)*batch_size] # (batch_size, 1, max_len)
                y_batch = y[batch_id*batch_size:(batch_id+1)*batch_size]
                lower_batch = lower[batch_id*batch_size:(batch_id+1)*batch_size]
                upper_batch = upper[batch_id*batch_size:(batch_id+1)*batch_size]
                adver_x_batch, success_batch = self.attack_batch(x_batch, y_batch, lower_batch, upper_batch, '{}-{}'.format(init, batch_id))
                if batch_id == 0:
                    adver_x = adver_x_batch
                    success = success_batch
                else:
                    adver_x = torch.cat((adver_x, adver_x_batch), 0)
                    success += success_batch
            if sum(success) / len(success) > best_success_rate:
                best_success_rate = sum(success) / len(success)
                best_success = success
                best_adver_x = adver_x

        return best_adver_x, best_success