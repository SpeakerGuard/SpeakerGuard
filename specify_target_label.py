
import torch
import numpy as np
import pickle
from torch.utils.data.dataloader import DataLoader

from defense.defense import *

from model.iv_plda import iv_plda
from model.xv_plda import xv_plda
from model.audionet_csine import audionet_csine

from model.defended_model import defended_model


from dataset.Dataset import Dataset

import warnings

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    
    #Step1: load the model
     # set up model
    if args.system_type == 'iv_plda':
        base_model = iv_plda(args.gmm, args.extractor, args.plda, args.mean, args.transform, device=device, model_file=args.model_file, threshold=args.threshold)
    elif args.system_type == 'xv_plda':
        base_model = xv_plda(args.extractor, args.plda, args.mean, args.transform, device=device, model_file=args.model_file, threshold=args.threshold)
    elif args.system_type == 'audionet_csine':
        base_model = audionet_csine(args.extractor, label_encoder=args.label_encoder, device=device)
    else:
        raise NotImplementedError('Unsupported System Type')
    
    defense, defense_name = parser_defense(args.defense, args.defense_param, args.defense_flag, args.defense_order)
    model = defended_model(base_model=base_model, defense=defense, order=args.defense_order)
    spk_ids = base_model.spk_ids

    possible_decisions = list(range(len(spk_ids)))
    if args.task == 'SV' or args.task == 'OSI':
        possible_decisions.append(-1) # -1: rejecting

    #Step2: load the dataset
    dataset = Dataset(spk_ids, args.root, args.name, return_file_name=True)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    if args.task == 'SV':
        args.hardest = False # hardest is meaningless for SV task
    if args.hardest and args.simplest:
        args.hardest = False
        args.simplest = False
        warnings.warn('You set both hardest and simplest to true, will roll back to random!!')
        
    #Step3: start
    name2target = {}
    with torch.no_grad():
        for index, (origin, true, file_name) in enumerate(loader):
            origin = origin.to(device)
            true = true.cpu().item()
            decision, scores = model.make_decision(origin)
            decision = decision.cpu().item()
            scores = scores.cpu().numpy().flatten() # (n_spks,)
            candidate_target_labels = [ii for ii in possible_decisions if ii != true and ii != decision]
            candidate_scores = [score for ii, score in enumerate(scores) if ii != true and ii != decision]
            if len(candidate_target_labels) == 0:
                candidate_target_labels = [ii for ii in possible_decisions if ii != decision]
            if len(candidate_scores) == 0:
                candidate_scores = [score for ii, score in enumerate(scores) if ii != decision]
            if not args.hardest and not args.simplest:
                target_label = np.random.choice(candidate_target_labels)
            else:
                if -1 in candidate_target_labels:
                    candidate_target_labels.remove(-1) # reject decision has no score, so remove it
                target_label = candidate_target_labels[np.argmin(candidate_scores)] if args.hardest else \
                    candidate_target_labels[np.argmax(candidate_scores)]
            name2target[file_name[0]] = target_label
            print(index, file_name[0], scores, true, decision, target_label) 
    # Step4: save
    save_path = args.save_path if args.save_path else \
        '{}-{}-{}-{}-{}.target_label'.format(args.system_type, args.task, 
        defense_name, args.name, args.hardest)
    with open(save_path, 'wb') as writer:
        pickle.dump(name2target, writer, -1)
    print('save file name and target label pair in {}'.format(save_path))

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-root', required=True) # the directory where the dataset locates
    parser.add_argument('-name', required=True) # the dataset name we specify target label for

    parser.add_argument('-save_path', default=None) # the path to store the file name and target label pair

    # whether setting the target label such that the attack is the hardest (simplest)
    # When both set to False or both true, will setting the target label randomly
    parser.add_argument('-hardest', action='store_true', default=False)
    parser.add_argument('-simplest', action='store_true', default=False)

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
    
    audionet_c_parser = subparser.add_parser("audionet_csine")
    audionet_c_parser.add_argument('-extractor', 
                default='pre-trained-models/audionet/cnn-natural-model-noise-0-002-50-epoch.pt.tmp8540_ckpt')
    audionet_c_parser.add_argument('-label_encoder', default='./label-encoder-audionet-Spk251_test.txt')

    parser.add_argument('-threshold', type=float, default=None) # for SV/OSI task
    parser.add_argument('-task', type=str, default='CSI', choices=['CSI', 'SV', 'OSI'])

    #### Note that for white-box attack, the defense method needs to be differentiable
    parser.add_argument('-defense', nargs='+', default=None)
    parser.add_argument('-defense_param', nargs='+', default=None)
    parser.add_argument('-defense_flag', nargs='+', default=None, type=int)
    parser.add_argument('-defense_order', default='sequential', choices=['sequential', 'average'])    

    args = parser.parse_args()
    main(args)