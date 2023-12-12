from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics
import scipy.io as sio

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    # Load data
    data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape
    print(data)
    print(len(data['idx_train']))
    print(len(data['idx_val']))
    print(len(data['idx_test']))
    print('data')


    #noise = torch.tensor(np.random.normal(0,0.01,size=data['features'].shape),dtype=torch.float)
    #noise1 = torch.normal(0,1,size=data['features'].shape)
    #noise01 = torch.normal(0,0.1,size=data['features'].shape)
    #noise001 = torch.normal(0,0.01,size=data['features'].shape)
    #noise0001 = torch.normal(0,0.001,size=data['features'].shape)
    #noise0005 = torch.normal(0,0.005,size=data['features'].shape)
    #noise=noise0001
    #print(noise)
    #np.savetxt('./noise0001_corn.csv',noise.detach().numpy(),fmt='%.4f',delimiter=',')
    #for i in data['idx_test']:
    #    if i not in data['idx_val']:
    #       data['features'][i]= (data['features']+noise)[i]
    

    if  args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel
        else:
            Model = RECModel
            # No validation for reconstruction task
            args.eval_freq = args.epochs + 1

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model = Model(args)
    logging.info(str(model))
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None

    lip=[]
    val=[]
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings, drop_weight,x_hyp = model.encode(data['features'], data['adj_train_norm'])
        #np.savetxt('./pubmed_features.csv',embeddings.detach().numpy(),fmt='%.4f',delimiter=',')
        print(embeddings.shape)
        print(embeddings)
        print('11')
        train_metrics = model.compute_metrics(embeddings, drop_weight,x_hyp, data, 'train')
        #print(debug)
        train_metrics['loss'].backward()
        lip.append(train_metrics['lip'].cpu().detach().numpy())
        #for p in model.parameters():
        #    print(p.grad.norm())
        #    torch.nn.utils.clip_grad_norm(p,10)
        #optimizer.step
        
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()
        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {}'.format(lr_scheduler.get_lr()[0]),
                                   format_metrics(train_metrics, 'train'),
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings,drop_weight,x_hyp = model.encode(data['features'], data['adj_train_norm'])
            #np.savetxt('./cha1_features.csv',embeddings.detach().numpy(),fmt='%.6f',delimiter=',')
            print(embeddings.shape)
            print(embeddings)
            print('22')
            val_metrics = model.compute_metrics(embeddings, drop_weight, x_hyp,data, 'val')
            val.append(train_metrics['acc'])
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, drop_weight, x_hyp,data, 'test')
                best_emb = embeddings.cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    logging.info("Early stopping")
                    break
                #print(embeddings.shape)
        #torch.save(model.state_dict(),'act_loss8.pt')

    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        best_emb, drop_weight,x_hyp = model.encode(data['features'], data['adj_train_norm'])
        #np.savetxt('./corn2_features.csv',embeddings.detach().numpy(),fmt='%.6f',delimiter=',')
        best_test_metrics = model.compute_metrics(best_emb, drop_weight, x_hyp,data, 'test')
    logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
    if args.save:
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logging.info(f"Saved model in {save_dir}")
    sio.savemat('act_l2_lip.mat', mdict={'lip':lip})  
    sio.savemat('act_l2_val.mat', mdict={'val':val})

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
