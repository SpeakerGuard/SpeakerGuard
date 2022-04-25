import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import warnings

class SEC4SR_CrossEntropy(nn.CrossEntropyLoss): # deal with something special on top of CrossEntropyLoss

    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', task='CSI'):
        super().__init__(weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce, reduction=reduction)

        assert task == 'CSI' # CrossEntropy only supports CSI task

    def forward(self, scores, label):

        _, num_class = scores.shape
        device = scores.device
        label = label.to(device)
        loss = torch.zeros(label.shape[0], dtype=torch.float, device=scores.device)

        consider_index = torch.nonzero(label != -1, as_tuple=True)[0].detach().cpu().numpy().tolist()
        if len(consider_index) > 0:
            loss[consider_index] = super().forward(scores[consider_index], label[consider_index])
        
        imposter_index = torch.nonzero(label == -1, as_tuple=True)[0].detach().cpu().numpy().tolist()
        if len(imposter_index):
            loss[imposter_index] = 0. * torch.sum(scores[imposter_index]) # make backward
        
        return loss

class SEC4SR_MarginLoss(nn.Module): # deal with something special on top of MarginLoss
    
    def __init__(self, targeted=False, confidence=0., task='CSI', threshold=None, clip_max=True) -> None:
        super().__init__()
        self.targeted = targeted
        self.confidence = confidence
        self.task = task
        self.threshold = threshold
        self.clip_max = clip_max

    def forward(self, scores, label):
        _, num_class = scores.shape
        device = scores.device
        label = label.to(device)
        loss = torch.zeros(label.shape[0], dtype=torch.float, device=scores.device)
        confidence = torch.tensor(self.confidence, dtype=torch.float, device=device)

        if self.task == 'SV':
            enroll_index = torch.nonzero(label == 0, as_tuple=True)[0].detach().cpu().numpy().tolist() 
            imposter_index = torch.nonzero(label == -1, as_tuple=True)[0].detach().cpu().numpy().tolist() 
            assert len(enroll_index) + len(imposter_index) == label.shape[0], 'SV task should not have labels out of 0 and -1'
            if len(enroll_index) > 0:
                if self.targeted: 
                    loss[enroll_index] = self.threshold + confidence - scores[enroll_index].squeeze(1) # imposter --> enroll, authentication bypass
                else:
                    loss[enroll_index] = scores[enroll_index].squeeze(1) + confidence - self.threshold # enroll --> imposter, Denial of Service
            if len(imposter_index) > 0:
                if self.targeted:
                    loss[imposter_index] = scores[imposter_index].squeeze(1) + confidence - self.threshold # enroll --> imposter, Denial of Service
                else:
                    loss[imposter_index] = self.threshold + confidence - scores[imposter_index].squeeze(1) # imposter --> enroll, authentication bypass
        
        elif self.task == 'CSI' or self.task == 'OSI':
            # remove imposter index which is unmeaningful for CSI task
            consider_index = torch.nonzero(label != -1, as_tuple=True)[0].detach().cpu().numpy().tolist()
            if len(consider_index) > 0:
                label_one_hot = torch.zeros((len(consider_index), num_class), dtype=torch.float, device=device)
                for i, ii in enumerate(consider_index):
                    index = int(label[ii])
                    label_one_hot[i][index] = 1
                score_real = torch.sum(label_one_hot * scores[consider_index], dim=1)
                score_other = torch.max((1-label_one_hot) * scores[consider_index] - label_one_hot * 10000, dim=1)[0]
                if self.targeted:
                    loss[consider_index] = score_other + confidence - score_real if self.task == 'CSI' \
                        else torch.clamp(score_other, min=self.threshold) + confidence - score_real
                else:
                    if self.task == 'CSI':
                        loss[consider_index] = score_real + confidence - score_other
                    else:
                        f_reject = torch.max(scores[consider_index], 1)[0] + confidence - self.threshold # spk m --> reject
                        f_mis = torch.clamp(score_real, min=self.threshold) + confidence - score_other # spk_m --> spk_n
                        loss[consider_index] = torch.minimum(f_reject, f_mis)
            
            imposter_index = torch.nonzero(label == -1, as_tuple=True)[0].detach().cpu().numpy().tolist()
            if self.task == 'OSI':
                # imposter_index = torch.nonzero(label == -1, as_tuple=True)[0].detach().cpu().numpy().tolist()
                if len(imposter_index) > 0:
                    if self.targeted:
                        loss[imposter_index] = torch.max(scores[imposter_index], 1)[0] + confidence - self.threshold
                    else:
                        loss[imposter_index] = self.threshold + confidence - torch.max(scores[imposter_index], 1)[0]
            else: # CSI
                if len(imposter_index):
                    loss[imposter_index] = 0. * torch.sum(scores[imposter_index]) # make backward

            # else:
            #     loss[imposter_index] = torch.zeros(len(imposter_index))

        if self.clip_max:
            loss = torch.max(torch.tensor(0, dtype=torch.float, device=device), loss)
        
        return loss 

def resolve_loss(loss_name='Entropy', targeted=False, confidence=0., task='CSI', threshold=None, clip_max=True):
    assert loss_name in ['Entropy', 'Margin']
    assert task in ['CSI', 'SV', 'OSI']
    if task == 'SV' or task == 'OSI' or loss_name == 'Margin': # SV/OSI: ignore loss name, force using Margin Loss
        loss = SEC4SR_MarginLoss(targeted=targeted, confidence=confidence, task=task, threshold=threshold, clip_max=clip_max)
        if (task == 'SV' or task == 'OSI') and loss_name == 'Entropy':
            warnings.warn('You are targeting {} task. Force using Margin Loss.')
    else:
        # loss = nn.CrossEntropyLoss(reduction='none') # ONLY FOR CSI TASK
        loss = SEC4SR_CrossEntropy(reduction='none', task='CSI') # ONLY FOR CSI TASK
    grad_sign = (1 - 2 * int(targeted)) if loss_name == 'Entropy' else -1
    
    return loss, grad_sign

def resolve_prediction(decisions):
    # print(decisions)
    predict = []
    for d in decisions:
        counts = Counter(d)
        predict.append(counts.most_common(1)[0][0])
    predict = np.array(predict)
    return predict
