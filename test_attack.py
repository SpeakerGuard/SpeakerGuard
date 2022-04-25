
from torch.utils.data import DataLoader
import torch
import pickle
import numpy as np

from metric.metric import get_all_metric

from defense.defense import parser_defense

from model.iv_plda import iv_plda
from model.xv_plda import xv_plda
from model.audionet_csine import audionet_csine

from model.defended_model import defended_model

from dataset.Dataset import Dataset

import warnings

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
bits = 16

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

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

    #### add a defense layer in the model
    #### Note that for white-box attack, the defense method needs to be differentiable
    parser.add_argument('-defense', nargs='+', default=None)
    parser.add_argument('-defense_param', nargs='+', default=None)
    parser.add_argument('-defense_flag', nargs='+', default=None, type=int)
    parser.add_argument('-defense_order', default='sequential', choices=['sequential', 'average'])

    parser.add_argument('-root', type=str, required=True)
    parser.add_argument('-name', type=str, required=True)
    parser.add_argument('-root_ori', type=str, default=None) # directory where the name_ori locates
    parser.add_argument('-name_ori', type=str, default=None) # used to calculate imperceptibility
    parser.add_argument('-wav_length', type=int, default=None)

    ## common attack parameters
    # parser.add_argument('-targeted', action='store_true', default=False)
    parser.add_argument('-batch_size', type=int, default=1)

    parser.add_argument('-target_label_file', default=None) # used to test the targeted attack success rate

    args = parser.parse_args()
    return args

def main(args):

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
    
    wav_length = None if args.batch_size == 1 else args.wav_length
    # If you want to test the distance between ori and adv voices, you must make sure
    # the ori voice is not padded or cutted during adv voice generation, i.e., 
    # no batch attack (batch_size=1) and wav_length=None in attackMain.py
    # The reason is that if ori voice is not padded or cutted, the ori and adv voices will no longer align with each other,
    # and the impercpetibility result will be wrong
    if args.root_ori is not None and args.name_ori is not None:
        wav_length = None
        args.batch_size = 1 # force set args.batch_size to 1
        warnings.warn('You want to test the imperceptibility. \
        Make sure you set batch_size to 1 and wav_length to None for attackMain.py when generating adv. voices \
            Otherwise, the adv. and ori. voices will not align with each other. \
                and the imperceptibility result is wrong.')

    dataset = Dataset(spk_ids, args.root, args.name, return_file_name=True, wav_length=wav_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    if args.root_ori is not None and args.name_ori is not None:
        ori_dataset = Dataset(spk_ids, args.root_ori, args.name_ori, return_file_name=True, wav_length=wav_length)
        ori_loader = DataLoader(ori_dataset, batch_size=args.batch_size, num_workers=0)
        name2ori = {}
        for index, (origin, _, file_name) in enumerate(ori_loader):
            origin = origin.to(device)
            name2ori[file_name[0]] = origin # single audio, since args.batch_size=1

    if args.target_label_file is not None:
        with open(args.target_label_file, 'rb') as reader:
            name2target = pickle.load(reader)

    right_cnt = 0
    target_success_cnt = 0
    imper = [] # imperceptibilty results
    with torch.no_grad():
        for index, (adver, true, file_name) in enumerate(loader):
            adver = adver.to(device)
            true = true.to(device)
            decisions, _ = model.make_decision(adver)
            right_cnt += torch.where(true == decisions)[0].shape[0]
            # print('*' * 10, index, '*' * 10)
            # print(true, decisions) 

            # get target label
            target = None
            if args.target_label_file is not None:
                target = true.clone()
                for ii, name in enumerate(file_name):
                    if name in name2target.keys():
                        target[ii] = name2target[name]
                    else:
                        raise NotImplementedError('Wrong target label file')
                # print(target)
                target_success_cnt += torch.where(target == decisions)[0].shape[0]

            # get original audios
            if args.root_ori is not None and args.name_ori is not None:
                imper_ = get_all_metric(name2ori[file_name[0]], adver)
                imper.append(imper_)
                # print(imper_)
            
            print((f"index: {index} true: {true} target: {target} decision: {decisions}"), end='\r')
    
    print()
    total_cnt = len(dataset)
    ACC = right_cnt * 100 / total_cnt
    print('Acc:', ACC)
    untar_ASR = 100. - ACC
    print('Untargeted Attack Success Rate:', untar_ASR)
    if args.target_label_file is not None:
        target_ASR = target_success_cnt * 100 / total_cnt
        print('Targeted Attack Success Rate:', target_ASR)
    if args.root_ori is not None and args.name_ori is not None:
        # check for abnormal imper
        imper = [imper_ for imper_ in imper if imper_[4] != np.infty]
        imper = np.mean(np.array(imper), axis=0)
        # print('L2, L0, L1, Linf, SNR, PESQ, STOI', imper)
        print('L2, SNR, PESQ: {:.3f} {:.2f} {:.2f}'.format(imper[0], imper[4], imper[5]))
    

if __name__ == "__main__":

    main(parse_args())