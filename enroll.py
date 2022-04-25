
import torch
import os
import numpy as np
import torchaudio
from defense.defense import parser_defense

from model.iv_plda import iv_plda
from model.xv_plda import xv_plda

from model.defended_model import defended_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):

    #Step 1: set up base_model
    if args.system_type == 'iv_plda':
        base_model = iv_plda(args.gmm, args.extractor, args.plda, args.mean, args.transform, device=device)
    elif args.system_type == 'xv_plda':
        base_model = xv_plda(args.extractor, args.plda, args.mean, args.transform, device=device)
    else:
        raise NotImplementedError('Unsupported System Type')
    
    defense, defense_name = parser_defense(args.defense, args.defense_param, args.defense_flag, args.defense_order)
    model = defended_model(base_model=base_model, defense=defense, order=args.defense_order)

    #Step2: scoring
    des_dir = args.model_dir
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    model_info = []

    # import pickle
    # with open('vox1-dev.pickle', 'rb') as reader:
    #     print('begin loading data')
    #     loader = pickle.load(reader)
    #     print('load data done')
    
    with torch.no_grad():
        root = args.root
        enroll_dir = os.path.join(root, 'Spk10_enroll')
        spk_iter = os.listdir(enroll_dir)
        for spk_id in spk_iter:

            spk_dir = os.path.join(enroll_dir, spk_id)
            audio_iter = os.listdir(spk_dir)
            num_enroll_utt = 0
            for ii, audio_name in enumerate(audio_iter):
                audio_path = os.path.join(spk_dir, audio_name)
                audio, _ = torchaudio.load(audio_path)
                audio = audio.to(device) * (2 ** (16-1))
                audio = audio.unsqueeze(0)
                emb_ = model.embedding(audio) # (1, dim)

                if ii == 0:
                    emb = emb_
                else:
                    emb.data += emb_
                
                num_enroll_utt += 1

            emb.data = emb / num_enroll_utt
            des_path = os.path.join(des_dir, args.system_type)
            if not os.path.exists(des_path):
                os.makedirs(des_path)
            emb_path = '{}/{}.{}'.format(des_path, spk_id, args.system_type) if defense is None else \
                '{}/{}.{}-{}'.format(des_path, spk_id, args.system_type, defense_name)
            torch.save(emb, emb_path)

            spk_nontarget_scores = []
            test_dir = os.path.join(root, 'Spk10_test')
            # test_dir = os.path.join(root, 'Spk10_imposter')
            test_spk_iter = os.listdir(test_dir)
            for test_spk_id in test_spk_iter:

                if test_spk_id == spk_id:
                    continue
                
                test_spk_dir = os.path.join(test_dir, test_spk_id)
                test_audio_iter = os.listdir(test_spk_dir)
                for name in test_audio_iter:
                    test_audio_path = os.path.join(test_spk_dir, name)
                    test_audio, _ = torchaudio.load(test_audio_path)
                    test_audio = (test_audio.to(device) * (2 ** (16-1))).unsqueeze(0)
                    scores = model.score(test_audio, enroll_embs=emb).flatten().detach().cpu().item()
                    spk_nontarget_scores.append(scores)
                    print(spk_id, name, scores)
            # spk_nontarget_scores = []
            # for index, (origin, _, file_name, lens) in enumerate(loader):
            #     origin = origin.to(device)
            #     scores = model.score(origin, enroll_embs=emb).flatten().detach().cpu().item()
            #     spk_nontarget_scores.append(scores)
            #     print(spk_id, file_name, scores)

            z_norm_mean = np.mean(spk_nontarget_scores)
            z_norm_std = np.std(spk_nontarget_scores) 
            
            spk_model_info = "{} {} {} {}".format(spk_id, emb_path, z_norm_mean, z_norm_std)
            model_info.append(spk_model_info)
            if defense is not None:
                spk_model_file = os.path.join(des_path, 'speaker_model_{}_{}_{}'.format(args.system_type, defense_name, spk_id))
            else:
                spk_model_file = os.path.join(des_path, 'speaker_model_{}_{}'.format(args.system_type, spk_id))
            np.savetxt(spk_model_file, [spk_model_info], fmt='%s')
        
        if defense is not None:
            model_file = os.path.join(des_path, 'speaker_model_{}_{}'.format(args.system_type, defense_name))
        else:
            model_file = os.path.join(des_path, 'speaker_model_{}'.format(args.system_type))
        np.savetxt(model_file, model_info, fmt='%s')


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-model_dir', default='model_file') # path to store speaker model file
    parser.add_argument('-root', default='data') # path where the Spk10_enroll and Spk10_test locate
    parser.add_argument('-defense', nargs='+', default=None)
    parser.add_argument('-defense_param', nargs='+', default=None)
    parser.add_argument('-defense_flag', nargs='+', default=None, type=int)
    parser.add_argument('-defense_order', default='sequential', choices=['sequential', 'average'])

    subparser = parser.add_subparsers(dest='system_type')

    iv_parser = subparser.add_parser("iv_plda")
    iv_parser.add_argument('-gmm', default='pre-trained-models/iv_plda/final_ubm.txt')
    iv_parser.add_argument('-extractor', default='pre-trained-models/iv_plda/final_ie.txt')
    iv_parser.add_argument('-plda', default='pre-trained-models/iv_plda/plda.txt')
    iv_parser.add_argument('-mean', default='pre-trained-models/iv_plda/mean.vec')
    iv_parser.add_argument('-transform', default='pre-trained-models/iv_plda/transform.txt')
    
    xv_parser = subparser.add_parser("xv_plda")
    xv_parser.add_argument('-extractor', default='pre-trained-models/xv_plda/xvecTDNN_origin.ckpt')
    xv_parser.add_argument('-plda', default='pre-trained-models/xv_plda/plda.txt')
    xv_parser.add_argument('-mean', default='pre-trained-models/xv_plda/mean.vec')
    xv_parser.add_argument('-transform', default='pre-trained-models/xv_plda/transform.txt')

    args = parser.parse_args()
    main(args)
