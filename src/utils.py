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


def load_data(dataset, num_train_per_class, num_test):
    return Planetoid(root='./data/{}'.format(dataset), name=dataset, split="random",
                     num_train_per_class=num_train_per_class,
                     num_val=500, num_test=num_test)


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


def DNS_parameters(dataset, model):
    if dataset == 'cora':
        label_rate = 4
        if model == 'GCN':
            seed, dc, lt_num, k, l, rootWeight, num_train_per_class = 1668587619, 2.5, 77, 15, 112, 2.69, 16
        elif model == 'lp':
            seed, dc, lt_num, k, l, rootWeight, num_train_per_class = 1668587619, 0.01, 10, 15, 112, 2, 16
        elif model == 'cotraining':
            seed, dc, lt_num, k, l, rootWeight, num_train_per_class = 1668590290, 0.01, 10, 15, 112, 2, 16
        elif model == 'selftraining':
            seed, dc, lt_num, k, l, rootWeight, num_train_per_class = 1668590290, 0.01, 10, 15, 112, 2, 16
        elif model == 'union':
            seed, dc, lt_num, k, l, rootWeight, num_train_per_class = 1668590290, 0.01, 10, 15, 112, 2, 16
        elif model == 'intersection':
            seed, dc, lt_num, k, l, rootWeight, num_train_per_class = 1668590290, 0.01, 10, 15, 112, 2, 16
    elif dataset == 'citeseer':
        label_rate = 2
        if model == 'GCN':
            seed, dc, lt_num, k, l, rootWeight, num_train_per_class = 4, 2.5, 1, 10, 66, 1.02, 11
        elif model == 'lp':
            seed, dc, lt_num, k, l, rootWeight, num_train_per_class = 1669639253, 0.92, 67, 11, 66, 1.99, 11
        elif model == 'cotraining':
            seed, dc, lt_num, k, l, rootWeight, num_train_per_class = 1669641433, 0.003, 86, 11, 66, 2.09, 11
        elif model == 'selftraining':
            seed, dc, lt_num, k, l, rootWeight, num_train_per_class = 1669643874, 1.67, 9, 7, 66, 1.11, 11
        elif model == 'union':
            seed, dc, lt_num, k, l, rootWeight, num_train_per_class = 1669696671, 0.01, 79, 11, 66, 1.19, 11
        elif model == 'intersection':
            seed, dc, lt_num, k, l, rootWeight, num_train_per_class = 1669696671, 1.67, 9, 7, 66, 1.19, 11
    elif dataset == 'pubmed':
        label_rate = 0.1
        if model == 'GCN':
            seed, dc, lt_num, k, l, rootWeight, num_train_per_class = 1670413855, 2.5, 10, 5, 21, 0.35, 7
        elif model == 'lp':
            seed, dc, lt_num, k, l, rootWeight, num_train_per_class = 1670413855, 2.5, 18, 5, 21, 0.36, 7
        elif model == 'cotraining':
            seed, dc, lt_num, k, l, rootWeight, num_train_per_class = 1670417397, 2.5, 18, 6, 21, 0.36, 7
        elif model == 'selftraining':
            seed, dc, lt_num, k, l, rootWeight, num_train_per_class = 1670417397, 2.5, 10, 5, 21, 0.39, 7
        elif model == 'union':
            seed, dc, lt_num, k, l, rootWeight, num_train_per_class = 1670482076, 2.5, 10, 5, 21, 0.39, 7
        elif model == 'intersection':
            seed, dc, lt_num, k, l, rootWeight, num_train_per_class = 1670482076, 2.4, 14, 5, 21, 0.39, 7

    dns_params = {'seed': seed, 'label_rate': label_rate, 'dc': dc, 'lt_num': lt_num, 'k': k, 'l': l,
                  'rootWeight': rootWeight, 'num_train_per_class': num_train_per_class}
    return dns_params


def model_parameters(dataset, model, args, dns_params):
    args = args
    model_config = None
    args.seed = dns_params['seed']
    random_seed = args.seed
    if dataset == 'cora':
        if model == 'GCN':
            args.layers = [16]
        elif model in ['lp', 'cotraining', 'selftraining', 'union', 'intersection']:
            connection = 'cc'
            layer_size = [16]
    if dataset == 'citeseer':
        if model == 'GCN':
            args.layers = [16, 16]
        elif model in ['lp', 'cotraining', 'selftraining', 'union', 'intersection']:
            connection = 'ccc'
            layer_size = [16, 16]
    if dataset == 'pubmed':
        if model == 'GCN':
            args.layers = [16, 16, 16]
        elif model in ['lp', 'cotraining', 'selftraining', 'union', 'intersection']:
            connection = 'cccc'
            layer_size = [16, 16, 16]
    if model in ['lp', 'cotraining', 'selftraining', 'union', 'intersection']:
        model_config = {'dataset': dataset,
                        'shuffle': True,
                        'train_size': dns_params['num_train_per_class'],
                        'validation_size': 0,
                        'validate': False,
                        'test_size': 1000,
                        'conv': 'gcn',
                        'max_degree': 2,
                        'learning_rate': 0.01,
                        'epochs': 200,
                        'Model': model,
                        'Model_to_add_label': {'Model': 0},
                        'Model_to_predict': {'Model': 0},
                        'Model19': 'union',
                        'alpha': 1e-06,
                        'connection': connection,
                        'layer_size': layer_size,
                        'dropout': 0.5,
                        'weight_decay': 0.0005,
                        'random_seed': 0,
                        'feature': 'bow',
                        'model_dir': './model/',
                        'name': None,
                        'threads': 32,
                        'train': True}

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
