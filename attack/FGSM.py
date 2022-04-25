
from attack.Attack import Attack
from attack.utils import resolve_loss, resolve_prediction
import numpy as np
import torch
from adaptive_attack.EOT import EOT

class FGSM(Attack):
    
    def __init__(self, model, task='CSI', epsilon=0.002, loss='Entropy', targeted=False, 
                batch_size=1, EOT_size=1, EOT_batch_size=1, 
                verbose=1):

        self.model = model # remember to call model.eval()
        self.task = task
        self.epsilon = epsilon
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

        self.max_iter = 1 # FGSM is single step attack, keep in consistency with PGD
        self.step_size = epsilon # keep in consistency with PGD
    
    def attack_batch(self, x_batch, y_batch, lower, upper, batch_id):
        
        x_batch = x_batch.clone() # avoid influcing
        # x_batch.retain_grad()
        x_batch.requires_grad = True
        success = None
        for iter in range(self.max_iter + 1):
            EOT_num_batches = int(self.EOT_size // self.EOT_batch_size) if iter < self.max_iter else 1
            real_EOT_batch_size = self.EOT_batch_size if iter < self.max_iter else 1
            use_grad = True if iter < self.max_iter else False
            # scores, loss, grad = EOT_wrapper(x_batch, y_batch, EOT_num_batches, real_EOT_batch_size, use_grad)
            scores, loss, grad, decisions = self.EOT_wrapper(x_batch, y_batch, EOT_num_batches, real_EOT_batch_size, use_grad)
            scores.data = scores / EOT_num_batches
            loss.data = loss / EOT_num_batches
            if iter < self.max_iter:
                grad.data = grad / EOT_num_batches
            # predict = torch.argmax(scores.data, dim=1).detach().cpu().numpy()
            predict = resolve_prediction(decisions)
            target = y_batch.detach().cpu().numpy()
            success = self.compare(target, predict, self.targeted)
            if self.verbose:
                print("batch:{} iter:{} loss: {} predict: {}, target: {}".format(batch_id, iter, loss.detach().cpu().numpy().tolist(), predict, target))
            
            if iter < self.max_iter:
                x_batch.grad = grad
                # x_batch.data += self.epsilon * torch.sign(x_batch.grad) * self.grad_sign
                ## x_batch.data += self.epsilon * torch.sign(grad) * self.grad_sign
                x_batch.data += self.step_size * torch.sign(x_batch.grad) * self.grad_sign
                x_batch.grad.zero_()
                # x_batch.data = torch.clamp(x_batch.data, min=lower, max=upper)
                x_batch.data = torch.min(torch.max(x_batch.data, lower), upper)
            
        return x_batch, success

    def attack(self, x, y):

        lower = -1
        upper = 1 
        assert lower <= x.max() < upper, 'generating adversarial examples should be done in [-1, 1) float domain' 
        n_audios, n_channels, _ = x.size()
        assert n_channels == 1, 'Only Support Mono Audio'
        assert y.shape[0] == n_audios, 'The number of x and y should be equal' 
        lower = torch.tensor(lower, device=x.device, dtype=x.dtype).expand_as(x)
        upper = torch.tensor(upper, device=x.device, dtype=x.dtype).expand_as(x)

        batch_size = min(self.batch_size, n_audios)
        n_batches = int(np.ceil(n_audios / float(batch_size)))
        for batch_id in range(n_batches):
            x_batch = x[batch_id*batch_size:(batch_id+1)*batch_size] # (batch_size, 1, max_len)
            y_batch = y[batch_id*batch_size:(batch_id+1)*batch_size]
            lower_batch = lower[batch_id*batch_size:(batch_id+1)*batch_size]
            upper_batch = upper[batch_id*batch_size:(batch_id+1)*batch_size]
            adver_x_batch, success_batch = self.attack_batch(x_batch, y_batch, lower_batch, upper_batch, batch_id)
            if batch_id == 0:
                adver_x = adver_x_batch
                success = success_batch
            else:
                adver_x = torch.cat((adver_x, adver_x_batch), 0)
                success += success_batch

        return adver_x, success
