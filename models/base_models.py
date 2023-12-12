"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import FermiDiracDecoder
import layers.hyp_layers as hyp_layers
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import acc_f1
import scipy.io as sio


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)

    def compute_metrics(self, embeddings,drop_weight, x_hyp,data, split):
        #print(self.weights)
        idx = data[f'idx_{split}']
        #print(idx)
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        #print(output)
        #print(output.shape[0])
        #print('output')
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        print(loss)  
        #print(embeddings.shape)
        #print(data['features'].shape)
        #print(embeddings.shape)
        #print(self.weights) 
        #print(self.weights.shape)
        #print(self.f1_average.shape)
        #print('loss')
        #b = np.tanh(1.0)
    #0    
        b=torch.norm(torch.tanh(torch.tensor(1)))
        a = 1 / (b**2) + 1 / (b * (1 - b**2))
        print("a:", a)
        print(drop_weight.shape)
        print(x_hyp.shape)
        x=x_hyp.t()
        M=drop_weight
        print(x.shape[1])
        #norm_M = torch.norm(M).detach().numpy()  
        norm_M = torch.norm(M)
        print(norm_M)
        #Mx = torch.matmul(M, x).detach().numpy()
        Mx = torch.matmul(M, x)
        print(Mx.shape)
        #norm_Mx=np.linalg.norm(Mx,axis=0)
        norm_Mx=torch.norm(Mx,dim=0)
        print(norm_Mx.shape)
        norm_x = torch.norm(x,dim=0)
        print(norm_x.shape)
        print('norm')

# a
        tanh_arg = (norm_Mx / norm_x) * torch.atanh(norm_x)
        tanh_term = torch.tanh(tanh_arg)
        print(tanh_term.shape)
        sech_term = 1 / torch.cosh(tanh_arg)**2
        print(sech_term.shape)

        part1 =2*norm_M/ norm_Mx 
        atanh_x = torch.atanh(norm_x)
        #print(atanh_x.shape)
        #print(torch.norm(M.t().mm(M)))
        #print(torch.norm(torch.matmul(M.t(),M)))
        #print(norm_M**2)
        part2 = atanh_x*torch.norm(M.t().mm(M))/ (norm_Mx**2) + (atanh_x) / (norm_x**2) + 1 / (norm_x* (1 - norm_x**2))
        result = tanh_term*part1 + norm_Mx*sech_term*part2
        print(result.shape)
        lipsum=torch.sum(result)
        print(lipsum)

#b
        # tanh_arg = (norm_Mx / norm_x) 
        # tanh_term = torch.tanh(tanh_arg)
        # #print(tanh_term.shape)
        # sech_term = 1 / torch.cosh(tanh_arg)**2
        # # print(sech_term.shape)

        # part1 =2*norm_M/ norm_Mx 
        # # #print(atanh_x.shape)
        # # #print(torch.norm(M.t().mm(M)))
        # # #print(torch.norm(torch.matmul(M.t(),M)))
        # # #print(norm_M**2)
        # part2 = torch.norm(M.t().mm(M))/ (norm_Mx**2) + 1/ (norm_x**2) + 1 / (norm_x* (1 - norm_x**2))
        # result = tanh_term*part1 + norm_Mx*sech_term*part2
        # print(result.shape)
        # lipsum=torch.sum(result)
        # print(lipsum)
# c

        # # part1 =2*norm_M/ norm_Mx 
        # # part2 = torch.norm(M.t().mm(M))/ (norm_Mx) + norm_Mx*a
        # result2 = 2*norm_M/ norm_Mx+torch.norm(M.t().mm(M))/ (norm_Mx) + norm_Mx*a
        # # print(result.shape)
        # lipsum2=torch.sum(result2)
        # # print(lipsum)


        #part1=(2 * norm_M) /norm_Mx
        #print(part1.shape)
        #part2=(norm_M ** 2) /norm_Mx
        #print(part2.shape)
        #part3=a * norm_Mx
        #print(part3.shape)

        #lip2= (2 * norm_M) /norm_Mx+ (norm_M ** 2) *norm_Mx + a * norm_Mx
        #print(lip2.shape)
        #print('lip')
        # lip = np.array(lip)
        # lip = torch.from_numpy(lip)
        # #print(lip)
        #lipsum2=torch.sum(lip2)
        #print(lipsum)
        
        #sio.savemat('lip'.mat', mdict={'lip': lipsum})  

        #loss=loss+(10e-7)*lipsum
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1, 'lip': lipsum}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges

    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return probs

    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

