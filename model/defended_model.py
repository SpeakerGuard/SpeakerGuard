
import torch
import torch.nn as nn

import warnings

# ensemble manner of multiple defenses
sequential = 'sequential' # model(d_n(...d_2(d_(x))))
average = 'average' # average(model(d_(x)), ..., model(d_n(x)))

class defended_model(nn.Module):

    def __init__(self, base_model, defense=None, order=sequential):

        super().__init__()

        self.base_model = base_model
        self.threshold = base_model.threshold
        
        ## check the defense
        if defense is not None:
            flag2defense = dict()
            for flag in self.base_model.allowed_flags:
                flag2defense[flag] = []
            assert isinstance(defense, (list, tuple))
            assert order in [sequential, average]
            prev_flag = -1
            for flag_method in defense:
                assert isinstance(flag_method, (list, tuple))
                assert len(flag_method) == 2
                flag = flag_method[0]
                if flag not in self.base_model.allowed_flags:
                    warnings.warn('Unsupported Input Level Flag. Ignore the Defense!')
                    continue
                method = flag_method[1]
                flag2defense[flag].append(method)
                if order == sequential:
                    if flag < prev_flag:
                        warnings.warn('You want to combine multiple defenses in sequential order, but it seems that the order of your defense is wrong. I have reranged for you!')
                    prev_flag = flag
            self.order = order
            self.flag2defense = flag2defense
        self.defense = defense

    
    def process_sequential(self, x):
        '''
        x: wav with shape [B, 1, T]
        return: the final type of input to the base model, e.g., cmvn_feat for iv_plda/xv_plda and raw_feat for audionet_csine
        '''
        if self.defense is not None:
            for flag in sorted(list(self.flag2defense.keys())): # sorted is important here
                if flag == 0:
                    xx = x.clone()
                elif flag == 1:
                    xx = self.base_model.compute_feat(xx, flag=1)
                else:
                    xx = self.base_model.comput_feat_from_feat(xx, ori_flag=flag-1, des_flag=flag)
                for defense in self.flag2defense[flag]:
                    # print(xx)
                    xx = defense(xx)
                    # print(xx)
            return xx
        else:
            return x


    def embedding(self, x):
        '''
        x: wav with shape [B, 1, T]
        return the same thing as the base model
        '''
        if self.defense is not None:
            if self.order == sequential:
                xx = self.process_sequential(x)
                return self.base_model.embedding(xx, flag=sorted(list(self.flag2defense.keys()))[-1])
            else:
                avg_emb = None
                for flag in sorted(list(self.flag2defense.keys())): # sorted is important here
                    if flag == 0:
                        xx = x.clone()
                    else:
                        xx = self.base_model.compute_feat(x, flag=flag)
                    for defense in self.flag2defense[flag]:
                        xxx = defense(xx)
                        emb = self.base_model.embedding(xxx, flag=flag)
                        if avg_emb is None:
                            avg_emb = emb
                        else:
                            avg_emb.data += emb
                avg_emb.data /= len(self.defense)
                return avg_emb
        else:
            return self.base_model.embedding(x, flag=0)

            

    def forward(self, x, return_emb=False, enroll_embs=None):
        '''
        x: wav with shape [B, 1, T]
        return the same thing as the base model
        '''
        if self.defense is not None:
            if self.order == sequential:
                xx = self.process_sequential(x)
                return self.base_model(xx, flag=sorted(list(self.flag2defense.keys()))[-1], return_emb=return_emb, enroll_embs=enroll_embs)
            else:
                avg_logits = None
                avg_emb = None
                for flag in sorted(list(self.flag2defense.keys())): # sorted is important here
                    if flag == 0:
                        xx = x.clone()
                    else:
                        xx = self.base_model.compute_feat(x, flag=flag)
                    for defense in self.flag2defense[flag]:
                        xxx = defense(xx)
                        logits, emb = self.base_model(xxx, flag=flag, return_emb=True, enroll_embs=enroll_embs)
                        if avg_emb is None:
                            avg_emb = emb
                            avg_logits = logits
                        else:
                            avg_emb.data += emb
                            avg_logits.data += logits
                avg_logits.data /= len(self.defense)
                avg_emb.data /= len(self.defense)
                return avg_logits if not return_emb else (avg_logits, avg_emb)
        else:
            return self.base_model(x, flag=0, return_emb=return_emb, enroll_embs=enroll_embs)

    
    def score(self, x, enroll_embs=None):
        '''
        x: wav with shape [B, 1, T]
        return the same thing as the base model
        '''
        if self.defense is not None:
            if self.order == sequential:
                xx = self.process_sequential(x)
                return self.base_model.score(xx, flag=sorted(list(self.flag2defense.keys()))[-1], enroll_embs=enroll_embs)
            else:
                avg_scores = None
                for flag in sorted(list(self.flag2defense.keys())): # sorted is important here
                    if flag == 0:
                        xx = x.clone()
                    else:
                        xx = self.base_model.compute_feat(x, flag=flag)
                    for defense in self.flag2defense[flag]:
                        xxx = defense(xx)
                        scores = self.base_model.score(xxx, flag=flag, enroll_embs=enroll_embs)
                        if avg_scores is None:
                            avg_scores = scores
                        else:
                            avg_scores.data += scores
                avg_scores.data /= len(self.defense)
                return avg_scores
        else:
            return self.base_model.score(x, flag=0, enroll_embs=enroll_embs)
    

    def make_decision(self, x, enroll_embs=None):
        '''
        x: wav with shape [B, 1, T]
        return the same thing as the base model
        '''
        scores = self.score(x, enroll_embs=enroll_embs)

        decisions = torch.argmax(scores, dim=1)
        max_scores = torch.max(scores, dim=1)[0]
        decisions = torch.where(max_scores > self.base_model.threshold, decisions,
                        torch.tensor([-1] * decisions.shape[0], dtype=torch.int64, device=decisions.device))

        return decisions, scores