import time

from torch_geometric.datasets import Planetoid
import random
import numpy as np
import torch
from src import GOLF as lt
from src.DeLaLA_select import DeLaLA_select
from torch_geometric.transforms import GCNNorm
from torch_sparse import SparseTensor
from copy import deepcopy
from torch_geometric.datasets import Amazon
import torch_geometric.transforms as T
from DI.gcn.utils import preprocess_model_config
import pprint


def load_data(dataset, num_train_per_class, num_test):
    if dataset == 'photo':
        return Amazon(root='./data/{}'.format(dataset), name=dataset, transform=T.NormalizeFeatures())
    else:
        return Planetoid(root='./data/{}'.format(dataset), name=dataset, split="random",
                         num_train_per_class=num_train_per_class,
                         num_val=0, num_test=num_test)


def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def Determinate_Node_Selection(data, y, dc, lt_num, k, l, rootWeight):
    start = time.time()
    g = GCNNorm(add_self_loops=True)
    agg = g(deepcopy(data))
    adj = SparseTensor(row=agg.edge_index[0], col=agg.edge_index[1], value=agg.edge_weight)
    rep = adj.matmul(agg.x)
    ldt = lt.LeadingTree(data, dc, lt_num, rep)
    ldt.fit()
    LTgammaPara = ldt.gamma
    idx_train = DeLaLA_select(LTgammaPara, ldt.density, ldt.layer, y, k, l,
                              rootWeight)  # get train data index
    idx = np.arange(data.x.shape[0])
    remain = np.delete(idx, idx_train)
    print('DNS time cost: {:.4f}s'.format(time.time() - start))

    return idx_train, remain


def DNS_parameters(dataset, model, label_rate):
    if dataset == 'cora':
        if label_rate == 0.5:
            if model == 'GCN':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 2, 14, 2, 2
            elif model == 'lp':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 2, 14, 2, 2
            elif model == 'cotraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.5, 25, 1, 14, 2, 2
            elif model == 'selftraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.5, 25, 2, 14, 2, 2
            elif model == 'union':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 1, 14, 2, 2
            elif model == 'intersection':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 1, 14, 2, 2
        elif label_rate == 1:
            if model == 'GCN':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 3, 28, 2, 4
            elif model == 'lp':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 1, 28, 2, 4
            elif model == 'cotraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 2.3, 25, 4, 28, 0.92, 4
            elif model == 'selftraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 1, 28, 2, 4
            elif model == 'union':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 1, 28, 2, 4
            elif model == 'intersection':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 1, 28, 2, 4
        elif label_rate == 2:
            if model == 'GCN':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 6, 56, 2, 8
            elif model == 'lp':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 2.3, 25, 7, 56, 0.92, 8
            elif model == 'cotraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 2.3, 25, 7, 56, 0.92, 8
            elif model == 'selftraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 6, 56, 2, 8
            elif model == 'union':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 6, 56, 2, 8
            elif model == 'intersection':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 6, 56, 2, 8
        elif label_rate == 3:
            if model == 'GCN':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 2.3, 25, 10, 84, 0.92, 12
            elif model == 'lp':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 2.3, 25, 10, 84, 0.92, 12
            elif model == 'cotraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 2.3, 25, 11, 84, 0.92, 12
            elif model == 'selftraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 10, 84, 2, 12
            elif model == 'union':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 10, 84, 2, 12
            elif model == 'intersection':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 10, 84, 2, 12
        elif label_rate == 4:
            if model == 'GCN':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 99.1, 8, 12, 112, 1.02, 16
            elif model == 'lp':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 99.1, 8, 12, 112, 1.02, 16
            elif model == 'cotraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 11, 112, 2, 16
            elif model == 'selftraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 11, 112, 2, 16
            elif model == 'union':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 11, 112, 2, 16
            elif model == 'intersection':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 50, 11, 112, 2, 16
    elif dataset == 'citeseer':
        if label_rate == 0.5:
            if model == 'GCN':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 11, 1, 18, 0.5, 3
            elif model == 'lp':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 11, 1, 18, 0.5, 3
            elif model == 'cotraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 11, 2, 18, 0.5, 3
            elif model == 'selftraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 11, 1, 18, 0.5, 3
            elif model == 'union':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 11, 2, 18, 0.5, 3
            elif model == 'intersection':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 100, 110, 3, 18, 2.5, 3
        elif label_rate == 1:
            if model == 'GCN':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 11, 5, 36, 0.5, 6
            elif model == 'lp':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 11, 5, 36, 0.5, 6
            elif model == 'cotraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 11, 5, 36, 0.5, 6
            elif model == 'selftraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 11, 5, 36, 0.5, 6
            elif model == 'union':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 11, 4, 36, 0.5, 6
            elif model == 'intersection':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 11, 4, 36, 0.5, 6
        elif label_rate == 2:
            if model == 'GCN':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 11, 11, 66, 0.5, 11
            elif model == 'lp':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 2.49, 9, 11, 66, 1.17, 11
            elif model == 'cotraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 11, 10, 66, 0.5, 11
            elif model == 'selftraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 11, 6, 66, 0.5, 11
            elif model == 'union':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 11, 6, 66, 0.5, 11
            elif model == 'intersection':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 11, 6, 66, 0.5, 11
        elif label_rate == 3:
            if model == 'GCN':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 10, 15, 108, 0.5, 18
            elif model == 'lp':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 2.49, 9, 18, 108, 1.19, 18
            elif model == 'cotraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 10, 14, 108, 0.5, 18
            elif model == 'selftraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 2.49, 9, 18, 108, 1.19, 18
            elif model == 'union':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 10, 14, 108, 0.5, 18
            elif model == 'intersection':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 2.49, 9, 18, 108, 1.19, 18
        elif label_rate == 4:
            if model == 'GCN':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 10, 14, 132, 0.5, 22
            elif model == 'lp':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 2.49, 9, 16, 132, 1.19, 22
            elif model == 'cotraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 10, 14, 132, 0.5, 22
            elif model == 'selftraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 10, 14, 132, 0.5, 22
            elif model == 'union':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 10, 14, 132, 0.5, 22
            elif model == 'intersection':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.01, 10, 22, 132, 0.5, 22
    elif dataset == 'pubmed':
        if label_rate == 0.03:
            if model == 'GCN':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 15, 2, 6, 1.19, 2
            elif model == 'lp':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 15, 2, 6, 1.19, 2
            elif model == 'cotraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 15, 2, 6, 1.19, 2
            elif model == 'selftraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 15, 2, 6, 1.19, 2
            elif model == 'union':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 15, 2, 6, 1.19, 2
            elif model == 'intersection':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 15, 1, 6, 0.01, 2
        elif label_rate == 0.05:
            if model == 'GCN':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 15, 3, 9, 1.19, 3
            elif model == 'lp':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 15, 2, 9, 1.19, 3
            elif model == 'cotraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 15, 3, 9, 1.19, 3
            elif model == 'selftraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 15, 3, 9, 1.19, 3
            elif model == 'union':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 15, 2, 9, 1.19, 3
            elif model == 'intersection':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 15, 2, 9, 1.19, 3
        elif label_rate == 0.1:
            if model == 'GCN':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 15, 6, 21, 1.19, 7
            elif model == 'lp':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 15, 6, 21, 1.19, 7
            elif model == 'cotraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 15, 6, 21, 1.19, 7
            elif model == 'selftraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 15, 6, 21, 1.19, 7
            elif model == 'union':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 15, 6, 21, 1.19, 7
            elif model == 'intersection':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 15, 4, 21, 1.19, 7
    elif dataset == 'photo':
        if label_rate == 0.2:
            if model == 'GCN':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.5, 5, 2, 16, 0.99, 2
            elif model == 'lp':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.5, 5, 1, 16, 0.99, 2
            elif model == 'cotraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.5, 5, 2, 16, 0.99, 2
            elif model == 'selftraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.5, 5, 2, 16, 0.99, 2
            elif model == 'union':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.5, 5, 2, 16, 0.99, 2
            elif model == 'intersection':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.5, 5, 2, 16, 0.99, 2
        elif label_rate == 0.5:
            if model == 'GCN':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 99.1, 10, 3, 40, 1.99, 5
            elif model == 'lp':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 99.1, 10, 2, 40, 1.99, 5
            elif model == 'cotraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 99.1, 10, 3, 40, 1.99, 5
            elif model == 'selftraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 99.1, 10, 3, 40, 1.99, 5
            elif model == 'union':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 99.1, 10, 3, 40, 1.99, 5
            elif model == 'intersection':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 99.1, 10, 3, 40, 1.99, 5
        elif label_rate == 1:
            if model == 'GCN':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 99.1, 980, 6, 80, 0.8, 10
            elif model == 'lp':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 980, 6, 80, 0.77, 10
            elif model == 'cotraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 99.1, 980, 7, 80, 0.8, 10
            elif model == 'selftraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 99.1, 980, 6, 80, 0.8, 10
            elif model == 'union':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 99.1, 980, 6, 80, 0.8, 10
            elif model == 'intersection':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 99.1, 980, 6, 80, 0.8, 10
        elif label_rate == 1.5:
            if model == 'GCN':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 99.1, 980, 11, 112, 0.8, 14
            elif model == 'lp':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 0.1, 980, 10, 112, 0.77, 14
            elif model == 'cotraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 99.1, 980, 11, 112, 0.8, 14
            elif model == 'selftraining':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 99.1, 980, 11, 112, 0.8, 14
            elif model == 'union':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 99.1, 980, 11, 112, 0.8, 14
            elif model == 'intersection':
                seed, dc, lt_num, k, l, rootWeight, num_train_per_class = int(time.time()), 99.1, 980, 11, 112, 0.8, 14

    dns_params = {'seed': seed, 'label_rate': label_rate, 'dc': dc, 'lt_num': lt_num, 'k': k, 'l': l,
                  'rootWeight': rootWeight, 'num_train_per_class': num_train_per_class}
    print(dns_params)
    return dns_params


def model_parameters(dataset, model, args, dns_params):
    args = args
    args.seed = dns_params['seed']
    connection = None
    layer_size = None
    if dataset == 'cora':
        if dns_params['label_rate'] == 0.5:
            if model == 'GCN':
                args.layers = [16, 16, 16]
            elif model in ['lp', 'cotraining', 'selftraining', 'union', 'intersection']:
                connection = 'cccc'
                layer_size = [16, 16, 16]
        elif dns_params['label_rate'] == 1 or dns_params['label_rate'] == 2:
            if model == 'GCN':
                args.layers = [16, 16]
            elif model in ['lp', 'cotraining', 'selftraining', 'union', 'intersection']:
                connection = 'ccc'
                layer_size = [16, 16]
        else:
            if model == 'GCN':
                args.layers = [16]
            elif model in ['lp', 'cotraining', 'selftraining', 'union', 'intersection']:
                connection = 'cc'
                layer_size = [16]
    if dataset == 'citeseer':
        if dns_params['label_rate'] == 0.5 or dns_params['label_rate'] == 1 or dns_params['label_rate'] == 2:
            if model == 'GCN':
                args.layers = [16, 16]
            elif model in ['lp', 'cotraining', 'selftraining', 'union', 'intersection']:
                connection = 'ccc'
                layer_size = [16, 16]
        else:
            if model == 'GCN':
                args.layers = [16]
            elif model in ['lp', 'cotraining', 'selftraining', 'union', 'intersection']:
                connection = 'cc'
                layer_size = [16]
    if dataset == 'pubmed':
        if dns_params['label_rate'] < 0.5:
            if model == 'GCN':
                args.layers = [16, 16, 16]
            elif model in ['lp', 'cotraining', 'selftraining', 'union', 'intersection']:
                connection = 'cccc'
                layer_size = [16, 16, 16]
        else:
            args.layers = [16]
            connection = 'cccc'
            layer_size = [16, 16, 16]
    if dataset == 'photo':
        connection = 'cc'
        layer_size = [64]

    # model_config = {'repeating': args.runs,
    #                 'default': {
    #                     'dataset': dataset,
    #                     'shuffle': True,
    #                     'train_size': dns_params['num_train_per_class'],
    #                     'validation_size': 0,
    #                     'validate': False,
    #                     'test_size': 1000,
    #                     'conv': 'gcn',
    #                     'max_degree': 2,
    #                     'learning_rate': args.lr,
    #                     'epochs': 200,
    #                     'Model': model,
    #                     'Model_to_add_label': {'Model': 0},
    #                     'Model_to_predict': {'Model': 0},
    #                     'Model19': 'union',
    #                     'alpha': 1e-06,
    #                     'connection': connection,
    #                     'layer_size': layer_size,
    #                     'dropout': 0.5,
    #                     'weight_decay': args.decay,
    #                     'random_seed': 0,
    #                     'feature': 'bow',
    #                     'model_dir': './model/',
    #                     'name': None,
    #                     'threads': 8,
    #                     'train': True
    #                     }
    #                 }
    # model_config['model_list'] = model_config['default']
    # # model_list = []
    # # model_list.append({'Model': model})
    #
    # # model_config['model_list'] = list(map(set_default_attr(model_config, model),
    # #                                        model_config['model_list']))
    #
    # preprocess_model_config(model_config['model_list'])
    # pprint.PrettyPrinter(indent=4).pprint(model_config['model_list'])
    model_config = {
                        'dataset': dataset,
                        'shuffle': True,
                        'train_size': dns_params['num_train_per_class'],
                        'validation_size': 0,
                        'validate': False,
                        'test_size': args.test_size,
                        'conv': 'gcn',
                        'max_degree': 2,
                        'learning_rate': args.lr,
                        'epochs': args.epochs,
                        'Model': model,
                        'Model_to_add_label': {'Model': 0},
                        'Model_to_predict': {'Model': 0},
                        'Model19': 'union',
                        'alpha': 1e-06,
                        'connection': connection,
                        'layer_size': layer_size,
                        'dropout': 0.5,
                        'weight_decay': args.decay,
                        'random_seed': args.seed,
                        'feature': 'bow',
                        'model_dir': './model/',
                        'name': None,
                        'threads': 8,
                        'train': True
                        }
    # return args, model_config['model_list']
    return args, model_config

def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        st_ = "{}_{}_".format(name, val)
        st += st_

    return st[:-1]


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals
