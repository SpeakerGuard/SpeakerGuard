
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import logging

from dataset.Spk251_train import Spk251_train
from dataset.Spk251_test import Spk251_test

from model.audionet_csine import audionet_csine
from model.defended_model import defended_model
from defense.defense import parser_defense 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parser_args():
    import argparse 

    parser = argparse.ArgumentParser()

    parser.add_argument('-defense', nargs='+', default=None)
    parser.add_argument('-defense_param', nargs='+', default=None)
    parser.add_argument('-defense_flag', nargs='+', default=None, type=int)
    parser.add_argument('-defense_order', default='sequential', choices=['sequential', 'average'])

    parser.add_argument('-label_encoder', default='./label-encoder-audionet-Spk251_test.txt')

    parser.add_argument('-aug_eps', type=float, default=0.002)
    
    parser.add_argument('-root', default='./data') # directory where Spk251_train and Spk251_test locates
    parser.add_argument('-num_epoches', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-wav_length', type=int, default=80_000)

    parser.add_argument('-model_ckpt', type=str)
    parser.add_argument('-log', type=str)
    parser.add_argument('-ori_model_ckpt', type=str)
    parser.add_argument('-ori_opt_ckpt', type=str)
    parser.add_argument('-start_epoch', type=int, default=0)

    parser.add_argument('-evaluate_per_epoch', type=int, default=1)

    args = parser.parse_args()
    return args


def validation(model, val_data):
    model.eval()
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
    return right_cnt / total_cnt 

def main(args):

    # load model
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

    # loss
    criterion = torch.nn.CrossEntropyLoss()

    # 
    log = args.log if args.log else ('./model_file/audionet-natural-{}.log'.format(defense_name) if defense is not None else \
                                    './model_file/audionet-natural.log')
    logging.basicConfig(filename=log, level=logging.DEBUG)
    model_ckpt = args.model_ckpt if args.model_ckpt else \
        ('./model_file/audionet-natural-{}'.format(defense_name) if defense is not None else \
                                                                './model_file/audionet-natural') 
    print(log, model_ckpt)

    num_batches = len(train_dataset) // args.batch_size
    for i_epoch in range(args.num_epoches):
        all_accuracies = []
        model.train()
        for batch_id, (x_batch, y_batch) in enumerate(train_loader):
            start_t = time.time()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # print(x_batch.min(), x_batch.max())

            #Noise augmentation to normal samples
            all_ids = range(x_batch.shape[0])
            normal_ids = all_ids

            if args.aug_eps > 0.:
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

            # print('main:', x_batch.min(), x_batch.max())

            predictions, _ = model.make_decision(x_batch)
            acc = torch.where(predictions == y_batch)[0].size()[0] / predictions.size()[0]

            end_t = time.time() 
            print("Batch", batch_id, "/", num_batches, ": Acc = ", round(acc,4), "\t batch time =", end_t-start_t)

            all_accuracies.append(acc)

        print()
        print('--------------------------------------') 
        print("EPOCH", i_epoch + args.start_epoch, "/", args.num_epoches + args.start_epoch, ": Acc = ", round(np.mean(all_accuracies),4))
        print('--------------------------------------') 
        print()
        logging.info("EPOCH {}/{}: Acc = {:.6f}".format(i_epoch + args.start_epoch, args.num_epoches + args.start_epoch, np.mean(all_accuracies)))

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
            val_acc = validation(model, val_loader) 
            print()
            print('Val Acc: %f' % (val_acc))
            print()
            logging.info('Val Acc: {:.6f}'.format(val_acc))
    
    # torch.save(model, model_ckpt)
    # torch.save(model.state_dict(), model_ckpt) # DO NOT save the whole defended_model
    torch.save(model.base_model.state_dict(), model_ckpt) # save the base_model

if __name__ == '__main__':

    main(parser_args())
