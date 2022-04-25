
import pickle
import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np

from model._iv_plda.gmm import FullGMM
from model._iv_plda.ivector_extract import ivectorExtractor
from model._iv_plda.plda import PLDA

from model.utils import check_input_range, parse_enroll_model_file, parse_mean_file, parse_transform_mat_file

BITS = 16

class iv_plda(nn.Module):

    def __init__(self, fgmm_file, extractor_file, plda_file, mean_file, transform_mat_file, 
                        model_file=None, threshold=None, device="cpu", gmm_frame_bs=200):

        super().__init__()

        self.device = device

        self.fgmm_file = fgmm_file
        self.extractor_file = extractor_file
        self.plda_file = plda_file
        
        ## using cache to save time since the from-scratch loading of the model is time consuming
        fgmm_cache_path = "{}/fgmm.pickle".format(os.path.dirname(fgmm_file))
        if not os.path.exists(fgmm_cache_path):
            self.fgmm = FullGMM(self.fgmm_file)
            with open(fgmm_cache_path, "wb") as writer:
                pickle.dump(self.fgmm, writer, -1)
        else:
            with open(fgmm_cache_path, "rb") as reader:
                self.fgmm = pickle.load(reader)
        
        extractor_cache_path = "{}/extractor.pickle".format(os.path.dirname(extractor_file))
        if not os.path.exists(extractor_cache_path):
            self.extractor = ivectorExtractor(self.extractor_file)
            with open(extractor_cache_path, "wb") as writer:
                pickle.dump(self.extractor, writer, -1)
        else:
            with open(extractor_cache_path, "rb") as reader:
                self.extractor = pickle.load(reader)
        
        plda_cache_path = "{}/plda.pickle".format(os.path.dirname(plda_file))
        if not os.path.exists(plda_cache_path):
            self.plda = PLDA(self.plda_file)
            with open(plda_cache_path, "wb") as writer:
                pickle.dump(self.plda, writer, -1)
        else:
            with open(plda_cache_path, "rb") as reader:
                self.plda = pickle.load(reader)
                
        if self.fgmm.device != self.device:
            self.fgmm.to(self.device)
        if self.extractor.device != self.device:
            self.extractor.to(self.device)
        if self.plda.device != self.device:
            self.plda.to(self.device)

        self.emb_mean = parse_mean_file(mean_file, self.device)
        self.transform_mat = parse_transform_mat_file(transform_mat_file, self.device)
        
        if model_file is not None:
            self.num_spks, self.spk_ids, self.z_norm_means, self.z_norm_stds, self.enroll_embs = \
                parse_enroll_model_file(model_file, self.device)

        # If you need SV or OSI, must input threshold
        self.threshold = threshold if threshold else -np.infty # Valid for SV and OSI tasks; CSI: -infty

        self.allowed_flags = sorted([
            0, 1, 2, 3
        ]) # 0: wav; 1: raw feat; 2: delta feat; 3: cmvn feat
        self.range_type = 'origin'
        
        # how many frames to be processed in one batch when calculating the 0-th and 1-th stats in GMM; 
	# setting > 1 to speed up the computation
        # adjust it according to your GPU memory
        self.gmm_frame_bs = gmm_frame_bs

    
    def compute_feat(self, x, flag=1):
        """
        x: wav with shape [B, 1, T]
        flag: the flag indicating to compute what type of features (1: raw feat; 2: delta feat; 3: cmvn feat)
        return: feats with shape [B, T, F] (T: #Frames, F: #feature_dim)
        """
        assert flag in [f for f in self.allowed_flags if f != 0]
        x = check_input_range(x, range_type=self.range_type)

        feats = self.raw(x) # (B, T, F)
        if flag == 1: # calulate ori feat
            pass
        else:
            feats = self.comput_feat_from_feat(feats, ori_flag=1, des_flag=2)
            if flag == 2:
                pass
            else:
                feats = self.comput_feat_from_feat(feats, ori_flag=2, des_flag=3)
                if flag == 3:
                    pass
                else: # will not enter this branch
                    pass 
            
        return feats

    
    def comput_feat_from_feat(self, feats, ori_flag=1, des_flag=2):
        """
        transfer function between different levels of acoustic features
        x: feature with shape [B, T, F]
        ori_flag: the level of input feature x
        des_flag: the level of the target feature
        """
        assert ori_flag in [f for f in self.allowed_flags if f != 0]
        assert des_flag in [f for f in self.allowed_flags if f != 0]
        assert des_flag > ori_flag

        if ori_flag == 1:
            if des_flag == 2:
                return self.add_delta(feats)
            if des_flag == 3:
                return self.cmvn(self.add_delta(feats))
        if ori_flag == 2:
            if des_flag == 3:
                return self.cmvn(feats)

    
    def embedding(self, x, flag=0):
        """
        x: wav or acoustic features (raw/delta/cmvn)
        flag: indicating the type of x (0: wav; 1: raw feat; 2: delta feat; 3: cmvn feat)
        """
        assert flag in self.allowed_flags
        if flag == 0:
            # x = check_input_range(x, range_type=self.range_type) # no need since compute_feat will check
            feats = self.compute_feat(x, flag=self.allowed_flags[-1])
        elif flag == 1:
            feats = self.comput_feat_from_feat(x, ori_flag=1, des_flag=self.allowed_flags[-1])
        elif flag == 2:
            feats = self.comput_feat_from_feat(x, ori_flag=2, des_flag=self.allowed_flags[-1])
        elif flag == 3:
            feats = x
        else:
            pass
        emb = self.extract_emb(feats)
        # return emb - self.emb_mean # [B, D]
        return emb # already subtract emb mean in self.extract_emb(feats)
    

    def forward(self, x, flag=0, return_emb=False, enroll_embs=None):
        """
        x: wav or acoustic features (raw/delta/cmvn)
        flag: indicating the type of x (0: wav; 1: raw feat; 2: delta feat; 3: cmvn feat)
        """
        embedding = self.embedding(x, flag=flag)
        
        if not hasattr(self, 'enroll_embs'):
            assert enroll_embs is not None
        enroll_embs = enroll_embs if enroll_embs is not None else self.enroll_embs
        scores = self.scoring_trials(enroll_embs=enroll_embs, embs=embedding)
        if not return_emb:
            return scores
        else:
            return scores, embedding

    
    def score(self, x, flag=0, enroll_embs=None):
        """
        x: wav or acoustic features (raw/delta/cmvn)
        flag: indicating the type of x (0: wav; 1: raw feat; 2: delta feat; 3: cmvn feat)
        """
        logits = self.forward(x, flag=flag, enroll_embs=enroll_embs)
        scores = logits
        return scores
    

    def make_decision(self, x, flag=0, enroll_embs=None):
        """
        x: wav or acoustic features (raw/delta/cmvn)
        flag: indicating the type of x (0: wav; 1: raw feat; 2: delta feat; 3: cmvn feat)
        """
        scores = self.score(x, flag=flag, enroll_embs=enroll_embs)

        decisions = torch.argmax(scores, dim=1)
        max_scores = torch.max(scores, dim=1)[0]
        decisions = torch.where(max_scores > self.threshold, decisions,
                        torch.tensor([-1] * decisions.shape[0], dtype=torch.int64, device=decisions.device)) # -1 means reject

        return decisions, scores

    
    def raw(self, x):
        """
        x: (B, 1, T)
        """
        batch_raw_feat = None
        for audio in x:
            raw_feat = torchaudio.compliance.kaldi.mfcc(
                                                audio,

                                                sample_frequency=16000, 
                                                frame_shift=10,
                                                frame_length=25,
                                                dither=1.0,  
                                                # dither=0.0, 
                                                preemphasis_coefficient=0.97,
                                                remove_dc_offset=True,
                                                window_type="povey",
                                                round_to_power_of_two=True,
                                                blackman_coeff=0.42,
                                                snip_edges=False,
                                                # allow_downsample=False,
                                                # allow_upsample=False,
                                                # max_feature_vectors=-1,

                                                num_mel_bins=30,
                                                low_freq=20,
                                                high_freq=7600,
                                                vtln_low=100,
                                                vtln_high=-500,
                                                vtln_warp=1.0,
                                                # debug_mel=False,
                                                # htk_mode=False,

                                                num_ceps=24, 
                                                use_energy=True,
                                                energy_floor=0.0, 
                                                # energy_floor=1.0, 
                                                # energy_floor=0.1,   
                                                raw_energy=True, 
                                                cepstral_lifter=22.0,
                                                htk_compat=False)

            raw_feat = raw_feat.unsqueeze(0) # (T, F) --> (1, T, F)
            if batch_raw_feat is None:
                batch_raw_feat = raw_feat
            else:
                batch_raw_feat = torch.cat((batch_raw_feat, raw_feat), dim=0)

        return batch_raw_feat # (B, T, F)
    

    def add_delta(self, batch_raw_feat, window=3, order=2, mode="replicate"):
        '''
        batch_raw_feat: (B, T, F)
        '''
        batch_delta_feat = None
        for raw_feat in batch_raw_feat:
            # get scales
            scales_ = self.get_scales(window, order, mode)

            num_frames, feat_dim = raw_feat.size()
            delta_feat = torch.zeros((num_frames,feat_dim * (order + 1)), dtype=torch.float, device=self.device)

            for i in range(0, order + 1):
                scales = scales_[i].to(raw_feat.device)
                max_offset = int((scales.size()[0] - 1) / 2) 
                j = torch.arange(-1*max_offset, max_offset+1, device=raw_feat.device)
                offset_frame = torch.clamp(j.repeat(num_frames, 1) + torch.arange(num_frames, device=raw_feat.device).view(-1, 1), min=0, max=num_frames-1)
                part_feat =  delta_feat[:, i*feat_dim:(i+1)*feat_dim]
                part_feat.add_(torch.sum(raw_feat[offset_frame.view(-1, ), :].view(num_frames, -1, feat_dim) * scales.view(-1, 1).expand(num_frames, scales.shape[0], 1), dim=1))

            delta_feat = delta_feat.unsqueeze(0)
            if batch_delta_feat is None:
                batch_delta_feat = delta_feat
            else:
                batch_delta_feat = torch.cat((batch_delta_feat, delta_feat), dim=0)

        return batch_delta_feat


    def get_scales(self, window, order, mode):
        scales = [torch.tensor([1.0], dtype=torch.float)]
        for i in range(1, order + 1):
            prev_scales = scales[i-1]
            assert window != 0 
            prev_offset = int((prev_scales.size()[0] - 1) / 2)
            cur_offset = int(prev_offset + window)
            cur_scales = torch.zeros((prev_scales.size()[0] + 2 * window, ), dtype=torch.float)
            normalizer = 0.0
            for j in range(int(-1 * window), int(window + 1)):
                normalizer += j * j
                for k in range(int(-1 * prev_offset), int(prev_offset + 1)):
                    cur_scales[j+k+cur_offset] += j * prev_scales[k+prev_offset]
            cur_scales = cur_scales * (1. / normalizer)
            scales.append(cur_scales)
        
        return scales


    def cmvn(self, batch_delta_feat):
        '''
        batch_delta_feat: (B, T, F)
        '''
        batch_cmvn_feat = None

        for delta_feat in batch_delta_feat:

            opts = {}
            CENTER = "center"
            NORMALIZE_VARIANCE = "normalize_variance"
            CMN_WINDOW = "cmn_window"
            opts[CENTER] = True
            opts[NORMALIZE_VARIANCE] = False
            opts[CMN_WINDOW] = 300

            num_frames, dim = delta_feat.size()
            last_window_start = -1
            last_window_end = -1
            cur_sum = torch.zeros((dim, ), device=self.device)
            cur_sumsq = torch.zeros((dim, ), device=self.device)

            cmvn_feat = delta_feat.clone()
            for t in range(num_frames):

                window_start = 0
                window_end = 0
                if opts[CENTER]:
                    window_start = t - (opts[CMN_WINDOW] / 2)
                    window_end = window_start + opts[CMN_WINDOW]
                else:
                    pass
                if window_start < 0:
                    window_end -= window_start
                    window_start = 0
                if not opts[CENTER]:
                    pass
                if window_end > num_frames:
                    window_start -= (window_end - num_frames)
                    window_end = num_frames
                    if window_start < 0:
                        window_start = 0
                if last_window_start == -1:
                    delta_feat_part = delta_feat[int(window_start):int(window_end), :]
                    cur_sum.fill_(0.)
                    cur_sum.add_(torch.sum(delta_feat_part, 0, keepdim=False), alpha=1.)
                    if opts[NORMALIZE_VARIANCE]:
                        pass
                else:
                    if window_start > last_window_start:
                        assert window_start == last_window_start + 1
                        frame_to_remove = delta_feat[int(last_window_start), :]
                        cur_sum.add_(frame_to_remove, alpha=-1.)

                        if opts[NORMALIZE_VARIANCE]:
                            pass
                    
                    if window_end > last_window_end:
                        assert window_end == last_window_end + 1
                        frame_to_add = delta_feat[int(last_window_end), :] 
                        cur_sum.add_(frame_to_add, alpha=1.)

                        if opts[NORMALIZE_VARIANCE]:
                            pass
                
                window_frames = window_end - window_start
                last_window_start = window_start
                last_window_end = window_end

                assert window_frames > 0 
                cmvn_feat[t].add_(cur_sum, alpha=-1. / window_frames)

                if opts[NORMALIZE_VARIANCE]:
                    pass

            cmvn_feat = cmvn_feat.unsqueeze(0)
            if batch_cmvn_feat is None:
                batch_cmvn_feat = cmvn_feat
            else:
                batch_cmvn_feat = torch.cat((batch_cmvn_feat, cmvn_feat), dim=0)
        
        return batch_cmvn_feat


    def extract_emb(self, x):
        '''
        x: (B, T, F)
        '''
        batch_emb = None
        for mfcc in x:
            zeroth_stats, first_stats = self.fgmm.Zeroth_First_Stats(mfcc, self.gmm_frame_bs)
            emb, _, _ = self.extractor.Extract(zeroth_stats, first_stats)
            emb = self.process_emb(emb, num_utt=1, simple_length_norm=False, normalize_length=True)

            emb = emb.unsqueeze(0)
            if batch_emb is None:
                batch_emb = emb
            else:
                batch_emb = torch.cat((batch_emb, emb), dim=0)

        return batch_emb

    
    def scoring_trials(self, enroll_embs, embs):
        scores = None
        for emb in embs:
            score = self.plda.ComputeScores(enroll_embs, 1, emb)
            score = score.unsqueeze(0)
            if scores is None:
                scores = score
            else:
                scores = torch.cat((scores, score), dim=0)
        return scores


    def process_emb(self, emb, num_utt=1, simple_length_norm=False, normalize_length=True):
        emb = self.SubtractGlobalMean(emb)
        emb = self.lda_reduce_dim(emb)
        emb = self.LengthNormalization(emb)
        emb = self.TransformEmb(emb, num_utt, simple_length_norm, normalize_length)
        return emb


    def SubtractGlobalMean(self, emb):
        return self.extractor.SubtractGlobalMean(emb, self.emb_mean)
    

    def lda_reduce_dim(self, emb):
        _, transform_cols = self.transform_mat.size()
        vec_dim = emb.size()[0]
        reduced_dim_vec = None
        if transform_cols == vec_dim:
            pass # not our case, just skip
        else:
            assert transform_cols == vec_dim + 1
            reduced_dim_vec = self.transform_mat[:, vec_dim:vec_dim+1]
            reduced_dim_vec = reduced_dim_vec.clone() # avoid influcing self.transform_mat  
            reduced_dim_vec.add_(1. * torch.matmul(self.transform_mat[:, :-1], torch.unsqueeze(emb, 1)))

        return torch.squeeze(reduced_dim_vec)
    
    
    def LengthNormalization(self, emb):
        return self.extractor.LengthNormalization(emb, torch.sqrt(torch.tensor(emb.size()[0], dtype=torch.float, device=self.device)))
    

    def TransformEmb(self, emb, num_utt=1, simple_length_norm=False, normalize_length=True):
        return self.plda.Transform(emb, num_utt, simple_length_norm, normalize_length)

    
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%% Deprecated  Deprecated Deprecated Deprecated %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # def forward(self, audios, return_emb=False, wav=0, enroll_embs=None):
    #     """
    #         audios: (batch, 1, T)
    #     """
    #     if audios.max() <= 1 and audios.min() >= -1:
    #         audios = audios * (2 ** (BITS-1))
    #     n_audios = audios.size()[0]
    #     n_spks = enroll_ivectors.size()[0]
    #     scores = torch.zeros((n_audios, n_spks), device=self.device)
    #     embedding = None
    #     if self.transform_layer is not None:
    #         #using squeeze then unsqueeze since self.transform_layer takes (batch, T) input
    #         audios = self.transform_layer(audios.squeeze(1), param=self.param).unsqueeze(1)
    #     for index in range(n_audios):
    #         ivector = self.compute_ivector(audios[index])
    #         enroll_embs = enroll_embs if enroll_embs is not None else self.enroll_embs
    #         # scores[index] = self.ComputeScores(enroll_ivectors, ivector)
    #         scores[index] = self.ComputeScores(enroll_embs, ivector)

    #         if index == 0:
    #             embedding = ivector
    #         else:
    #             embedding = torch.cat((embedding, ivector), dim=0)

    #     if not return_emb:
    #         return scores
    #     else:
    #         return scores, embedding    
    #     return scores
    
    # def score(self, audios, wav=0, enroll_embs=None):
    #     scores = self.forward(audios, wav=wav, enroll_embs=enroll_embs)
    #     scores = (scores - self.z_norm_means) / self.z_norm_stds
    #     return scores
    
    # def make_decision(self, audios, wav=0, enroll_embs=None): # -1: reject
    #     scores = self.score(audios, wav=wav, enroll_embs=enroll_embs)
    #     decisions = torch.argmax(scores, dim=1)
    #     max_scores = torch.max(scores, dim=1)[0]
    #     decisions = torch.where(max_scores > self.threshold , decisions, 
    #                     torch.tensor([-1] * decisions.shape[0], dtype=torch.int64, device=decisions.device))

    #     return decisions, scores 

    # def compute_ivector(self, audio, num_utt=1, simple_length_norm=False, normalize_length=True):
    #     ivector = self.Extract(audio)
    #     return self.process_ivector(ivector, num_utt=num_utt, 
    #                     simple_length_norm=simple_length_norm, normalize_length=normalize_length)

    # def Extract(self, audio):
    #     if len(audio.shape) == 1:
    #         audio = audio.unsqueeze(0)
    #     mfcc = self.mfcc(audio)
    #     zeroth_stats, first_stats = self.fgmm.Zeroth_First_Stats(mfcc)
    #     ivector, _, _ = self.extractor.Extractivector(zeroth_stats, first_stats)
    #     return ivector
    
    # def ComputeScores(self, enroll_ivectors, ivector):
    #     return self.plda.ComputeScores(enroll_ivectors, 1, ivector)

    # def mfcc(self, audio):
    #     if audio.device != self.device:
    #         audio = audio.to(self.device)
    #     raw_feat = self.raw(audio)

    #     if not self.feat_transform:
    #         delta_feat = self.add_delta(raw_feat)
    #         cmvn_feat = self.cmvn(delta_feat)
    #         vad_result = self.vad(raw_feat)
    #         final_feat = self.select_voiced_frames(cmvn_feat, vad_result)
    #     elif self.feat_point == 'raw':
    #         raw_feat = self.transform_layer(raw_feat, param=self.param, other_param=self.other_param)
    #         delta_feat = self.add_delta(raw_feat)
    #         cmvn_feat = self.cmvn(delta_feat)
    #         vad_result = self.vad(raw_feat)
    #         final_feat = self.select_voiced_frames(cmvn_feat, vad_result)
    #     elif self.feat_point == 'delta':
    #         delta_feat = self.transform_layer(self.add_delta(raw_feat), param=self.param, other_param=self.other_param)
    #         cmvn_feat = self.cmvn(delta_feat)
    #         raw_feat = self.transform_layer(raw_feat, param=self.param, other_param=self.other_param)
    #         vad_result = self.vad(raw_feat)
    #         final_feat = self.select_voiced_frames(cmvn_feat, vad_result)
    #     elif self.feat_point == 'cmvn':
    #         delta_feat = self.add_delta(raw_feat)
    #         cmvn_feat = self.transform_layer(self.cmvn(delta_feat), param=self.param, other_param=self.other_param)
    #         raw_feat = self.transform_layer(raw_feat, param=self.param, other_param=self.other_param)
    #         vad_result = self.vad(raw_feat)
    #         final_feat = self.select_voiced_frames(cmvn_feat, vad_result)
    #     elif self.feat_point == 'final':
    #          delta_feat = self.add_delta(raw_feat)
    #          cmvn_feat = self.cmvn(delta_feat)
    #          vad_result = self.vad(raw_feat)
    #          final_feat = self.transform_layer(self.select_voiced_frames(cmvn_feat, vad_result), 
    #                         param=self.param, other_param=self.other_param)
    #     else:
    #         raise NotImplementedError('Not Supported Feat Point')
        
    #     return final_feat


    


    # def vad(self, raw_feat):
    #     r""" Only appliable to the case when vad_frames_context = 0
    #     """

    #     vad_energy_threshold = 5.5 # default 5.0
    #     vad_energy_mean_scale = 0.5 # default 0.5
    #     vad_frames_context = 0 # default 0 
    #     vad_proportion_threshold = 0.6 # default 0.6

    #     assert vad_energy_mean_scale >= 0.0
    #     assert vad_frames_context >= 0
    #     assert vad_proportion_threshold > 0.0
    #     assert vad_proportion_threshold < 1.0

    #     T = raw_feat.size()[0]
    #     log_energy = raw_feat[:, 0]

    #     energy_threshold = vad_energy_threshold
    #     energy_threshold += vad_energy_mean_scale * torch.mean(log_energy)

    #     vad_result = (log_energy > energy_threshold).float()

    #     return vad_result
    
    # def vad_2(self, raw_feat):

    #     vad_energy_threshold = 5.5 # default 5.0
    #     vad_energy_mean_scale = 0.5 # default 0.5
    #     vad_frames_context = 0 # default 0 
    #     vad_proportion_threshold = 0.6 # default 0.6

    #     assert vad_energy_mean_scale >= 0.0
    #     assert vad_frames_context >= 0
    #     assert vad_proportion_threshold > 0.0
    #     assert vad_proportion_threshold < 1.0

    #     T = raw_feat.size()[0]
    #     log_energy = raw_feat[:, 0]

    #     energy_threshold = vad_energy_threshold
    #     energy_threshold += vad_energy_mean_scale * torch.mean(log_energy)

    #     binary = (log_energy > energy_threshold).float()
    #     vad_result = torch.zeros_like(log_energy)
    #     for i in range(T):
    #         binary_part = binary[max(0, i-vad_frames_context):min(T, i+vad_frames_context+1)]
    #         if torch.mean(binary_part) >= vad_proportion_threshold:
    #             vad_result[i] = 1.0

    #     return vad_result

    # def select_voiced_frames(self, cmvn_feat, vad_result):
    #     num_frames = cmvn_feat.size()[0]
    #     assert num_frames == vad_result.size()[0]

    #     voiced_index = (vad_result == 1.0).nonzero().view(-1)
    #     assert voiced_index.shape[0] > 0
    #     final_feats = cmvn_feat[voiced_index, :]
    #     return final_feats

    # def to(self, device):
    #     if device == self.device:
    #         return
    #     self.device = device
    #     self.fgmm.to(self.device)
    #     self.extractor.to(self.device)
    #     self.plda.to(self.device)
    #     self.ivector_mean = self.ivector_mean.to(self.device)
    #     self.transform_mat = self.transform_mat.to(self.device)

    # def copy(self, device):
    #     copy_helper = copy.deepcopy(self)  
    #     copy_helper.to(device)
    #     return copy_helper
