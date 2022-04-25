
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import logging

from attack.FGSM import FGSM
from attack.PGD import PGD
from dataset.Spk251_train import Spk251_train
from dataset.Spk251_test import Spk251_test

from defense.defense import parser_defense

from model.audionet_csine import audionet_csine
from model.defended_model import defended_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parser_args(): 
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-defense', nargs='+', default=None)
    parser.add_argument('-defense_param', nargs='+', default=None)
    parser.add_argument('-defense_flag', nargs='+', default=None, type=int)
    parser.add_argument('-defense_order', default='sequential', choices=['sequential', 'average'])

    parser.add_argument('-label_encoder', default='./label-encoder-audionet-Spk251_test.txt')

    # parser.add_argument('-aug_eps', type=float, default=0.002)
    # unlike natural_train.py, we don't apply noise augmentation to normal examples 
    # since adversarial examples already augment the training data.
    parser.add_argument('-aug_eps', type=float, default=0.)


    parser.add_argument('-attacker', type=str, choices=['PGD', 'FGSM'], default='PGD')
    parser.add_argument('-epsilon', type=float, default=0.002) 
    parser.add_argument('-step_size', type=float, default=0.0004) # recommend: epsilon / 5
    parser.add_argument('-max_iter', type=int, default=10) # PGD-10 default
    parser.add_argument('-num_random_init', type=int, default=0)
    parser.add_argument('-EOT_size', type=int, default=1)
    parser.add_argument('-EOT_batch_size', type=int, default=1)
    
    # using root/Spk251_train as training data
    # using root/Spk251_test as validation data
    parser.add_argument('-root', type=str, default='./data') # directory where Spk251_train and Spk251_test locates
    parser.add_argument('-num_epoches', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-wav_length', type=int, default=80_000)

    parser.add_argument('-ratio', type=float, default=0.5) # the ratio of adversarial examples in each minibatch

    parser.add_argument('-model_ckpt', type=str)
    parser.add_argument('-log', type=str)
    parser.add_argument('-ori_model_ckpt', type=str)
    parser.add_argument('-ori_opt_ckpt', type=str)
    parser.add_argument('-start_epoch', type=int, default=0)

    parser.add_argument('-evaluate_per_epoch', type=int, default=1)
    parser.add_argument('-evaluate_adver', action='store_true', default=False) 

    args = parser.parse_args()
    return args


def validation(args, model, val_data, attacker):
    model.eval()
    val_normal_acc = None
    val_adver_acc = None
    with torch.no_grad():
        total_cnt = len(val_data)
        right_cnt = 0
        for index, (origin, true, file_name) in enumerate(val_data):
            origin = origin.to(device)
            true = true.to(device)
            decision, _ = model.make_decision(origin)
            print((f'[{index}/{total_cnt}], name:{file_name[0]}, true:{true.cpu().item():.0f}, predict:{decision.cpu().item():.0f}'), 
                end='\r')
            if decision == true:
                right_cnt += 1
        val_normal_acc = right_cnt / total_cnt
    print()
    if args.evaluate_adver:
        n_select = len(val_data)
        val_adver_cnt = 0
        for index, (origin, true, file_name) in enumerate(val_data):
            origin = origin.to(device)
            true = true.to(device)
            adver, success = attacker.attack(origin, true)
            decision, _ = model.make_decision(adver) 
            print((f'[{index}/{n_select}], name:{file_name[0]}, true:{true.cpu().item():.0f}, predict:{decision.cpu().item():.0f}'), 
                end='\r')
            if decision == true: 
                val_adver_cnt += 1 
        val_adver_acc = val_adver_cnt / n_select
        print()
    else:
        val_adver_acc = 0.0
    return val_normal_acc, val_adver_acc
    
def main(args):

    # load model
    if args.ori_model_ckpt:
        print(args.ori_model_ckpt)
        base_model = audionet_csine(extractor_file=args.ori_model_ckpt, label_encoder=args.label_encoder, device=device)
        base_model.train() # important!! since audionet_csine() will set to eval() if extractor_file is not None
    else:
        base_model = audionet_csine(label_encoder=args.label_encoder, device=device)
    spk_ids = base_model.spk_ids
    defense, defense_name = parser_defense(args.defense, args.defense_param, args.defense_flag, args.defense_order)
    model = defended_model(base_model=base_model, defense=defense, order=args.defense_order)
    print('load model done')

    # load optimizer
    optimizer = torch.optim.Adam(model.parameters())
    if args.ori_opt_ckpt:
        print(args.ori_opt_ckpt)
        # optimizer_state_dict = torch.load(args.ori_opt_ckpt).state_dict()
        optimizer_state_dict = torch.load(args.ori_opt_ckpt)
        optimizer.load_state_dict(optimizer_state_dict)
    print('set optimizer done')

    # load val data
    val_dataset = None
    val_loader = None
    if args.evaluate_per_epoch > 0:
        val_dataset = Spk251_test(spk_ids, args.root, return_file_name=True, wav_length=None)
        test_loader_params = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': True
        }
        val_loader = DataLoader(val_dataset, **test_loader_params)

    # load train data
    train_dataset = Spk251_train(spk_ids, args.root, wav_length=args.wav_length)
    train_loader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': True 
    }
    train_loader = DataLoader(train_dataset, **train_loader_params)
    print('load train data done', len(train_dataset))

    # attacker
    attacker = None
    if args.attacker == 'FGSM':
        attacker = FGSM(model, epsilon=args.epsilon, loss='Entropy', targeted=False, 
                        batch_size=int(args.batch_size * args.ratio), EOT_size=args.EOT_size, 
                        EOT_batch_size=args.EOT_batch_size, verbose=0)
    elif args.attacker == 'PGD':
        attacker = PGD(model, targeted=False, step_size=args.step_size,
                       epsilon=args.epsilon, max_iter=args.max_iter,
                       batch_size=int(args.batch_size * args.ratio), num_random_init=args.num_random_init,
                       loss='Entropy', EOT_size=args.EOT_size, EOT_batch_size=args.EOT_batch_size, 
                       verbose=0)
    else:
        raise NotImplementedError('Not Supported Attack Algorithm for Adversarial Training') 

    # loss
    criterion = torch.nn.CrossEntropyLoss()

    # 
    log = args.log if args.log else ('./model_file/audionet-adver-{}.log'.format(defense_name) if defense is not None else \
                                    './model_file/audionet-adver.log')
    logging.basicConfig(filename=log, level=logging.DEBUG)
    model_ckpt = args.model_ckpt if args.model_ckpt else \
        ('./model_file/audionet-adver-{}'.format(defense_name) if defense is not None else \
                                                                './model_file/audionet-adver') 
    print(log, model_ckpt)

    num_batches = len(train_dataset) // args.batch_size
    for i_epoch in range(args.num_epoches):
        all_accuracies = []
        all_accuracies_normal = []
        all_accuracies_adv = []
        ASR = []
        model.train()
        for batch_id, (x_batch, y_batch) in enumerate(train_loader):
            start_t = time.time()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Choose indices to replace with adversarial samples
            nb_adv = int(np.ceil(args.ratio * x_batch.shape[0]))
            if args.ratio < 1:
                adv_ids = np.random.choice(x_batch.shape[0], size=nb_adv, replace=False)
            else:
                adv_ids = list(range(x_batch.shape[0])) 
                np.random.shuffle(adv_ids)
                x_batch_clone = x_batch.clone()

            x_batch[adv_ids], success = attacker.attack(x_batch[adv_ids], y_batch[adv_ids])
            success_cnt = np.argwhere(success).flatten().size
            success_rate = success_cnt * 100 / len(success)
            ASR.append(success_rate)

            #Noise augmentation to normal samples
            all_ids = range(x_batch.shape[0])
            normal_ids = [i_ for i_ in all_ids if i_ not in adv_ids]
            if len(normal_ids) > 0 and args.aug_eps > 0.:
                x_batch_normal = x_batch[normal_ids, ...]
                y_batch_normal = y_batch[normal_ids, ...]

                a = np.random.rand()
                noise = torch.rand_like(x_batch_normal, dtype=x_batch_normal.dtype, device=device)
                epsilon = args.aug_eps
                noise = 2 * a * epsilon * noise - a * epsilon 
                x_batch_normal_noisy = x_batch_normal + noise
                x_batch = torch.cat((x_batch, x_batch_normal_noisy), dim=0)
                y_batch = torch.cat((y_batch, y_batch_normal))

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions, _ = model.make_decision(x_batch)
            acc = torch.where(predictions == y_batch)[0].size()[0] / predictions.size()[0]
            predictions_adv = predictions[adv_ids]
            acc_adv = torch.where(predictions_adv == y_batch[adv_ids])[0].size()[0] / predictions_adv.size()[0]
            predictions_normal = predictions[normal_ids]
            acc_normal= None
            if predictions_normal.size()[0] > 0:
                acc_normal = torch.where(predictions_normal == y_batch[normal_ids])[0].size()[0] / predictions_normal.size()[0]
            else:
                predictions_normal, _ = model.make_decision(x_batch_clone)
                acc_normal = torch.where(predictions_normal == y_batch)[0].size()[0] / predictions_normal.size()[0]
            
            end_t = time.time()
            print("Batch", batch_id, "/", num_batches, " : ASR = ", round(success_rate, 4), 
                    "\tAcc = ", round(acc,4), "\tAcc adv =", round(acc_adv,4), 
                    "\tAcc normal =", round(acc_normal,4), "\t batch time =", round(end_t-start_t, 6))

            all_accuracies.append(acc)
            all_accuracies_normal.append(acc_normal)
            all_accuracies_adv.append(acc_adv)

        print()
        print('--------------------------------------') 
        print("EPOCH", i_epoch + args.start_epoch, "/", args.num_epoches + args.start_epoch, 
                ": ASR = ", round(np.mean(ASR), 4), 
                "\tAcc = ", round(np.mean(all_accuracies),4), 
                "\tAcc adv =", round(np.mean(all_accuracies_adv),4), 
                "\tAcc normal =", round(np.mean(all_accuracies_normal),4))
        print('--------------------------------------') 
        print()
        logging.info("EPOCH {}/{}: ASR = {:.6f}\tAcc = {:.6f}\tAcc adv = {:.6f}\tAcc normal = {:.6f}".format(i_epoch + args.start_epoch, args.num_epoches + args.start_epoch, np.mean(ASR), np.mean(all_accuracies), np.mean(all_accuracies_adv), np.mean(all_accuracies_normal)))

        ### save ckpt
        ckpt = model_ckpt + "_{}".format(i_epoch + args.start_epoch)
        ckpt_optim = ckpt + '.opt'
        # torch.save(model, ckpt)
        # torch.save(optimizer, ckpt_optim)
        # torch.save(model.state_dict(), ckpt) # DO NOT save the whole defended_model
        torch.save(model.base_model.state_dict(), ckpt) # save the base_model
        torch.save(optimizer.state_dict(), ckpt_optim)
        print()
        print("Save epoch ckpt in %s" % ckpt)
        print()

        ### evaluate
        if args.evaluate_per_epoch > 0 and i_epoch % args.evaluate_per_epoch == 0:
            val_acc, val_adver_acc = validation(args, model, val_loader, attacker) 
            print()
            print('Val Acc: %f, Val Adver Acc: %f' % (val_acc, val_adver_acc))
            print()
            logging.info('Val Acc: {:.6f} Val Adver Acc: {:.6f}'.format(val_acc, val_adver_acc))
        
    # torch.save(model, model_ckpt)
    # torch.save(model.state_dict(), model_ckpt) # DO NOT save the whole defended_model
    torch.save(model.base_model.state_dict(), model_ckpt) # save the base_model

if __name__ == '__main__':

    main(parser_args())
