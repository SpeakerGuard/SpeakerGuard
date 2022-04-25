
'''
The code is drawn from 
https://github.com/manojpamk/pytorch_xvectors
'''

import torch
import torch.nn as nn
from torch.nn import functional as F


class xvecTDNN(nn.Module): ## x-vector network for feature input

    def __init__(self, numSpkrs, p_dropout=0.):
        super(xvecTDNN, self).__init__()
        self.tdnn1 = nn.Conv1d(in_channels=30, out_channels=512, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn_tdnn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn_tdnn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(3000,512)
        self.bn_fc1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        self.fc2 = nn.Linear(512,512)
        self.bn_fc2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc2 = nn.Dropout(p=p_dropout)

        self.fc3 = nn.Linear(512,numSpkrs)
    
    def embedding(self, x, eps=1e-5):
        # Note: x must be (batch_size, feat_dim, chunk_len)

        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
        x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))

        if self.training:
        # if True:
            shape = x.size()
            noise = torch.cuda.FloatTensor(shape)
            torch.randn(shape, out=noise)
            x += noise*eps
        
        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        x = self.fc1(stats)
        return x
    
    def forward(self, x, eps=1e-5):
        # Note: x must be (batch_size, feat_dim, chunk_len)

        x = self.embedding(x, eps=eps)
        x = self.dropout_fc1(self.bn_fc1(F.relu(x)))
        x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x

    def score(self, x, softmax=True): 
        x = self.forward(x)
        if softmax:
            x = F.softmax(x, dim=1)
        return x
    
    def make_decision(self, x):
        score = self.score(x, softmax=True)
        decision = torch.argmax(score, dim=1)
        return decision, score