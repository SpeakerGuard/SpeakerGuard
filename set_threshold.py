
'''
Part of the code is drawn from https://github.com/FAKEBOB-adversarial-attack/FAKEBOB
Paper: Who is Real Bob? Adversarial Attacks on Speaker Recognition Systems (IEEE S&P 2021)
'''

import torch
from torch.utils.data import DataLoader
import numpy as np
from defense.defense import parser_defense

from model.iv_plda import iv_plda
from model.xv_plda import xv_plda

from model.defended_model import defended_model

from dataset.Spk10_test import Spk10_test
from dataset.Spk10_imposter import Spk10_imposter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_threshold(score_target, score_untarget):

    if not isinstance(score_target, np.ndarray):
        score_target = np.array(score_target)
    if not isinstance(score_untarget, np.ndarray):
        score_untarget = np.array(score_untarget)

    n_target = score_target.size
    n_untarget = score_untarget.size

    final_threshold = 0.
    min_difference = np.infty
    final_far = 0.
    final_frr = 0.
    for candidate_threshold in score_target:

        frr = np.argwhere(score_target < candidate_threshold).flatten().size * 100 / n_target
        far = np.argwhere(score_untarget >= candidate_threshold).flatten().size * 100 / n_untarget
        difference = np.abs(frr - far)
        if difference < min_difference:
            final_threshold = candidate_threshold
            final_far = far
            final_frr = frr
            min_difference = difference

    return final_threshold, final_frr, final_far

def main(args):

    #Step 1: set up base_model
    if args.system_type == 'iv_plda':
        base_model = iv_plda(args.gmm, args.extractor, args.plda, args.mean, args.transform, device=device, model_file=args.model_file)
    elif args.system_type == 'xv_plda':
        base_model = xv_plda(args.extractor, args.plda, args.mean, args.transform, device=device, model_file=args.model_file)
    else:
        raise NotImplementedError('Unsupported System Type')
    
    defense, defense_name = parser_defense(args.defense, args.defense_param, args.defense_flag, args.defense_order)
    model = defended_model(base_model=base_model, defense=defense, order=args.defense_order)
    
    #Step2: load dataset
    test_dataset = Spk10_test(base_model.spk_ids, args.root, return_file_name=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)
    imposter_dataset = Spk10_imposter(base_model.spk_ids, args.root, return_file_name=True)
    imposter_loader = DataLoader(imposter_dataset, batch_size=1, num_workers=0)

    #Step3: scoring
    # score_target = []
    # score_untarget = []
    score_target_sv = []
    score_untarget_sv = []
    score_target_osi = []
    score_untarget_osi = []
    trues = [] # used to calculate IER for OSI
    max_scores = [] # used to calculate IER for OSI
    decisions = [] # used to calculate IER for OSI

    acc_cnt = 0
    with torch.no_grad():
        for index, (origin, true, file_name) in enumerate(test_loader):
            origin = origin.to(device)
            true = true.cpu().item()
            # print(origin.shape)
            decision, scores = model.make_decision(origin)
            decision = decision.cpu().item()
            scores = scores.cpu().numpy().flatten() # (n_spks,)
            print(index, file_name[0], scores, true, decision)
            score_target_sv.append(scores[true])
            score_untarget_sv += np.delete(scores, true).tolist()
            if decision == true:
                score_target_osi.append(scores[true])
            trues.append(true)
            max_scores.append(np.max(scores))
            decisions.append(decision)
            
            if decision == true:
                acc_cnt += 1

        for index, (origin, true, file_name) in enumerate(imposter_loader):
            origin = origin.to(device)
            true = true.cpu().item()
            decision, scores = model.make_decision(origin)
            decision = decision.cpu().item()
            scores = scores.cpu().numpy().flatten() # (n_spks,)
            print(index, file_name[0], scores, true, decision)
            score_untarget_sv += scores.tolist()
            score_untarget_osi.append(np.max(scores))
    
    task = 'SV'
    threshold, frr, far = set_threshold(score_target_sv, score_untarget_sv)
    print("----- Test of {}-based {}, result ---> threshold: {:.2f} EER: {:.2f}".format(args.system_type, 
    task, threshold, max(frr, far)))

    task = 'OSI'
    threshold, frr, far = set_threshold(score_target_osi, score_untarget_osi)
    IER_cnt = np.intersect1d(np.argwhere(max_scores >= threshold).flatten(),
                np.argwhere(decisions != trues).flatten()).flatten().size
    # # IER: Identification Error, 
    # for detail, refer to 'Who is Real Bob? Adversarial Attacks on Speaker Recognition Systems'
    IER = IER_cnt * 100 / len(trues) 
    print("----- Test of {}-based {}, result ---> threshold: {:.2f}, EER: {:.2f}, IER: {:.2f} -----".format(
        args.system_type, task, threshold, max(frr, far), IER))

    # CSI-E accuracy
    print('CSI ACC:', acc_cnt * 100 / len(test_loader))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-root', default='./data')
    parser.add_argument('-defense', nargs='+', default=None)
    parser.add_argument('-defense_param', nargs='+', default=None)
    parser.add_argument('-defense_flag', nargs='+', default=None, type=int)
    parser.add_argument('-defense_order', default='sequential', choices=['sequential', 'average'])

    subparser = parser.add_subparsers(dest='system_type') # either iv (ivector-PLDA) or xv (xvector-PLDA)

    iv_parser = subparser.add_parser("iv_plda")
    iv_parser.add_argument('-gmm', default='pre-trained-models/iv_plda/final_ubm.txt')
    iv_parser.add_argument('-extractor', default='pre-trained-models/iv_plda/final_ie.txt')
    iv_parser.add_argument('-plda', default='pre-trained-models/iv_plda/plda.txt')
    iv_parser.add_argument('-mean', default='pre-trained-models/iv_plda/mean.vec')
    iv_parser.add_argument('-transform', default='pre-trained-models/iv_plda/transform.txt')
    iv_parser.add_argument('-model_file', default='model_file/iv_plda/speaker_model_iv_plda')
    
    xv_parser = subparser.add_parser("xv_plda")
    xv_parser.add_argument('-extractor', default='pre-trained-models/xv_plda/xvecTDNN_origin.ckpt')
    xv_parser.add_argument('-plda', default='pre-trained-models/xv_plda/plda.txt')
    xv_parser.add_argument('-mean', default='pre-trained-models/xv_plda/mean.vec')
    xv_parser.add_argument('-transform', default='pre-trained-models/xv_plda/transform.txt')
    xv_parser.add_argument('-model_file', default='model_file/xv_plda/speaker_model_xv_plda')

    args = parser.parse_args()
    main(args)