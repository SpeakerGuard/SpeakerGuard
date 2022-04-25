
'''
Part of the code is drawn from 
https://github.com/usc-sail/gard-adversarial-speaker-id
Paper: 
Jati et al. Adversarial attack and defense strategies for deep speaker recognition systems
'''

import numpy as np

import torch
import torch.nn as nn
# import torch.nn.functional as F

from model._audionet.Preprocessor import Preprocessor
from model.utils import check_input_range

BITS = 16

class audionet_csine(nn.Module):
    """Adaption of AudioNet (arXiv:1807.03418)."""
    def __init__(self, extractor_file=None, num_class=None, label_encoder=None, device='cpu'):
        '''
        if extractor_file is not None, we want to inference by providing pre-trained model ckpt
        if extractor_file is None, we want to train the model. In this case, num_class must not be None
        '''
        super().__init__()

        self.device = device

        num_class_1 = None
        if extractor_file is not None:
            model_state_dict = torch.load(extractor_file, map_location=self.device)
            num_class_1 = model_state_dict['fc.bias'].shape[0]
        
        num_class_2 = None
        if label_encoder is not None:
            # parser label info
            id_label = np.loadtxt(label_encoder, dtype=str, converters={0: lambda s: s[1:-1]})
            id2label = {}
            label2id = {}
            for row in id_label:
                id2label[row[0]] = int(row[1])
                label2id[int(row[1])] = row[0]
            self.spk_ids = [label2id[i] for i in range(len(list(label2id.keys())))]
            self.id2label = id2label
            self.label2id = label2id
            num_class_2 = len(self.spk_ids) # label encoder provides spk_ids info
        
        if len([kk for kk in [num_class_1, num_class_2] if kk is not None]) == 2:
            assert num_class_1 == num_class_2
            num_class = num_class_1
        elif len([kk for kk in [num_class_1, num_class_2] if kk is not None]) == 1:
            num_class = [kk for kk in [num_class_1, num_class_2] if kk is not None][0]
        else:
            assert num_class is not None
        
        self.num_spks = num_class
        if not hasattr(self, 'spk_ids'):
            self.spk_ids = [str(i) for i in range(self.num_spks)]

        self.prep = Preprocessor()

        # =========== EXPERIMENTAL pre-filtering ======
        # 32 x 100
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=[5, 5], stride=1, padding=[2, 2]),
            nn.BatchNorm2d(1),
        )
        # =========== ============= ======

        # 32 x 100
        self.conv2 = nn.Sequential( 
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        # 64 x 100
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # 128 x 100
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # 128 x 50
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        # 128 x 50
        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # 128 x 25
        self.conv7 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # 32 x 30
        self.fc = nn.Linear(32, num_class)

        if extractor_file is not None:
            self.load_state_dict(model_state_dict)
            self.eval().to(self.device)
        else:
            self.train().to(self.device)

        self.threshold = -np.infty # this model targeted CSI-NE task, so threshold is -infty
        self.allowed_flags = sorted([
            0, 1
        ])# 0: wav; 1: raw feat
        self.range_type = 'scale'

    
    def compute_feat(self, x, flag=1):
        """
        x: wav with shape [B, 1, T]
        flag: the flag indicating to compute what type of features (1: raw feat)
        return: feats with shape [B, T, F] (T: #Frames, F: #feature_dim)
        """
        assert flag in [f for f in self.allowed_flags if f != 0]
        x = check_input_range(x, range_type=self.range_type)

        feats = self.raw(x) # (B, T, F)
        if flag == 1: # calulate ori feat
            return feats
        else: # will not go to this branch
            pass

    
    def embedding(self, x, flag=0):
        """
        x: wav or acoustic features (raw/delta/cmvn)
        flag: indicating the type of x (0: wav; 1: raw feat)
        """
        assert flag in self.allowed_flags
        if flag == 0:
            # x = check_input_range(x, range_type=self.range_type) # no need, since compute_feat will check
            feats = self.compute_feat(x, flag=self.allowed_flags[-1])
        elif flag == 1:
            feats = x
        else: # will not go to this branch
            pass
        emb = self.extract_emb(feats)
        # return emb - self.emb_mean # [B, D]
        return emb # already subtract emb mean in self.extract_emb(feats)
    

    def raw(self, x):
        """
        x: (B, 1, T)
        """
        x = x.squeeze(1)
        feat = self.prep(x) # (B, F, T)
        return feat.transpose(1, 2) # (B, T, F)
    

    def extract_emb(self, x):
        '''
        x: (B, T, F)
        '''
        x = x.transpose(1, 2)
        # ===== pre-filtering ========
        # [B, F, T]
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze(1)
        # ===== pre-filtering ========

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        target_len = 3
        real_len = x.shape[2]
        if real_len < target_len:
            n = target_len // real_len
            if target_len % real_len == 0:
                n = n
            else:
                n = n + 1
            x = x.repeat(1, 1, n)

        x = self.conv8(x)
        x, _ = x.max(2)
        return x


    def predict_from_embeddings(self, x):
        return self.fc(x)

    
    def forward(self, x, flag=0, return_emb=False, enroll_embs=None):
        """
        x: wav or acoustic features (raw/delta/cmvn)
        flag: indicating the type of x (0: wav; 1: raw feat)
        """
        """
        enroll_embs is useless since this model targets CSI-NE task
        just to keep consistency with other models
        """
        embedding = self.embedding(x, flag=flag)
        logits = self.predict_from_embeddings(embedding)
        if not return_emb:
            return logits
        else:
            return logits, embedding
    

    def score(self, x, flag=0, enroll_embs=None):
        """
        x: wav or acoustic features (raw/delta/cmvn)
        flag: indicating the type of x (0: wav; 1: raw feat)
        """
        """
        enroll_embs is useless since this model targets CSI-NE task
        just to keep consistency with other models
        """
        logits = self.forward(x, flag=flag)
        # scores = F.softmax(logits, dim=1) # do not use softmax! Since CrossEntropy Loss in Pytorch already does this for us. Also to keep consistent with other models.
        scores = logits
        return scores
    

    def make_decision(self, x, flag=0, enroll_embs=None):
        """
        x: wav or acoustic features (raw/delta/cmvn)
        flag: indicating the type of x (0: wav; 1: raw feat)
        """
        """
        enroll_embs is useless since this model targets CSI-NE task
        just to keep consistency with other models
        """
        scores = self.score(x, flag=flag)
        decisions = torch.argmax(scores, dim=1)
        return decisions, scores
