
from attack.FGSM import FGSM
from attack.utils import SEC4SR_MarginLoss
import torch
import numpy as np

class CW2(FGSM):

    def __init__(self, model, task='CSI',
                targeted=False,
                confidence=0.,
                initial_const=1e-3, 
                binary_search_steps=9,
                max_iter=10000,
                stop_early=True,
                stop_early_iter=1000,
                lr=1e-2,
                batch_size=1,
                verbose=1):

        self.model = model
        self.task = task
        self.targeted = targeted
        self.confidence = confidence
        self.initial_const = initial_const
        self.binary_search_steps = binary_search_steps
        self.max_iter = max_iter
        self.stop_early = stop_early
        self.stop_early_iter = stop_early_iter
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose

        self.threshold = None
        if self.task in ['SV', 'OSI']:
            self.threshold = self.model.threshold
            print('Running white box attack for {} task, directly using the true threshold {}'.format(self.task, self.threshold))

        self.loss = SEC4SR_MarginLoss(targeted=self.targeted, confidence=self.confidence, task=self.task, threshold=self.threshold, clip_max=True)
    
    def attack_batch(self, x_batch, y_batch, lower, upper, batch_id):

        n_audios, _, _ = x_batch.shape

        const = torch.tensor([self.initial_const] * n_audios, dtype=torch.float, device=x_batch.device)
        lower_bound = torch.tensor([0] * n_audios, dtype=torch.float, device=x_batch.device)
        upper_bound = torch.tensor([1e10] * n_audios, dtype=torch.float, device=x_batch.device)

        global_best_l2 = [np.infty] * n_audios
        global_best_adver_x = x_batch.clone()
        # global_best_score = [-1] * n_audios
        global_best_score = [-2] * n_audios # do not use [-1] * n_audios since -1 is within the decision space of SV and OSI tasks 

        for _ in range(self.binary_search_steps):

            self.modifier = torch.zeros_like(x_batch, dtype=torch.float, requires_grad=True, device=x_batch.device)
            self.optimizer = torch.optim.Adam([self.modifier], lr=self.lr)

            best_l2 = [np.infty] * n_audios
            # best_score = [-1] * n_audios
            best_score = [-2] * n_audios

            continue_flag = True
            prev_loss = np.infty
            # we need to perform the gradient descent max_iter times; 
            # the additional one iteration is used to to evaluate the final updated examples
            for n_iter in range(self.max_iter+1): 
            # for n_iter in range(self.max_iter):
                if not continue_flag:
                    break
                # deal with box constraint, [-1, 1], different from image
                input_x = torch.tanh(self.modifier + torch.atanh(x_batch * 0.999999))
                # scores = self.model(input_x) # (n_audios, n_spks)
                decisions, scores = self.model.make_decision(input_x) # (n_audios, n_spks)
                loss1 = self.loss(scores, y_batch)
                loss2 = torch.sum(torch.square(input_x - x_batch), dim=(1,2))
                loss = const * loss1 + loss2

                if n_iter < self.max_iter: # we only perform gradient descent max_iter times
                    loss.backward(torch.ones_like(loss))
                    # update modifier
                    self.optimizer.step()
                    self.modifier.grad.zero_()

                # predict = torch.argmax(scores.data, dim=1).detach().cpu().numpy() # not suitable for SV and OSI tasks which will reject
                predict = decisions.detach().cpu().numpy()
                scores = scores.detach().cpu().numpy()
                loss = loss.detach().cpu().numpy().tolist()
                loss1 = loss1.detach().cpu().numpy().tolist()
                loss2 = loss2.detach().cpu().numpy().tolist()
                if self.verbose:
                    print("batch: {}, c: {}, iter: {}, loss: {}, loss1: {}, loss2: {}, y_pred: {}, y: {}".format(
                        batch_id, const.detach().cpu().numpy(), n_iter, 
                        loss, loss1, loss2, predict, y_batch.detach().cpu().numpy()))
                
                if self.stop_early and n_iter % self.stop_early_iter == 0:
                    if np.mean(loss) > 0.9999 * prev_loss:
                        print("Early Stop ! ")
                        continue_flag = False
                    prev_loss = np.mean(loss)

                for ii, (l2, y_pred, adver_x, l1) in enumerate(zip(loss2, predict, input_x, loss1)):
                    # IF-BRANCH-1
                    if l1 <= 0 and l2 < best_l2[ii]: # l1 <= 0 indicates the attack succeed with at least kappa confidence
                        best_l2[ii] = l2
                        best_score[ii] = y_pred
                    # IF-BRANCH-2
                    if l1 <= 0 and l2 < global_best_l2[ii]: # l1 <= 0 indicates the attack succeed with at least kappa confidence
                        global_best_l2[ii] = l2
                        global_best_score[ii] = y_pred
                        global_best_adver_x[ii] = adver_x

            for jj, y_pred in enumerate(best_score):
                if y_pred != -2: # y_pred != -2 infers that IF-BRANCH-1 is entered at least one time, thus the attack succeeds
                    upper_bound[jj] = min(upper_bound[jj], const[jj])
                    if upper_bound[jj] < 1e9:
                        const[jj] = (lower_bound[jj] + upper_bound[jj]) / 2
                else:
                    lower_bound[jj] = max(lower_bound[jj], const[jj])
                    if upper_bound[jj] < 1e9:
                        const[jj] = (lower_bound[jj] + upper_bound[jj]) / 2
                    else:
                        const[jj] *= 10
            
            print(const.detach().cpu().numpy(), best_l2, global_best_l2)
        
        success = [False] * n_audios
        for kk, y_pred in enumerate(global_best_score):
            if y_pred != -2: # y_pred != -2 infers that IF-BRANCH-2 is entered at least one time, thus the attack succeeds
                success[kk] = True 

        return global_best_adver_x, success

    def attack(self, x, y):
        return super().attack(x, y)


