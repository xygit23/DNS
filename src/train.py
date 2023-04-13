import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from src.gcn_torch import GCN
from src.utils import load_data, set_random_seeds, DNS_parameters, Determinate_Node_Selection, model_parameters
from DI.train import DI_train


def random_masking(data, num_train_per_class, num_classes, num_test):
    train_mask = torch.zeros(size=(data.num_nodes,), dtype=torch.bool)
    train_mask.fill_(False)
    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
        train_mask[idx] = True

    remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
    remaining = remaining[torch.randperm(remaining.size(0))]

    test_mask = torch.zeros(size=(data.num_nodes,), dtype=torch.bool)
    test_mask.fill_(False)
    test_mask[remaining[:num_test]] = True

    return train_mask, test_mask


def train(epochs, model, data, optimizer, train_mask, test_mask):
    s = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    _, pred = F.log_softmax(model(data), dim=1).max(dim=1)
    correct = pred[test_mask].eq(data.y[test_mask]).sum()
    acc = int(correct) / int(test_mask.sum())
    duration = time.time() - s

    return acc, duration


def GCN_train(args, select_idx, remain, num_train_per_class):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_random_seeds(args.seed)
    acc_mean = []
    acc_std = []
    duration_mean = []
    acc_DNS_mean = []
    acc_DNS_std = []
    duration_DNS_mean = []


    for r in range(args.repeating):
        acc = []
        duration = []
        acc_DNS = []
        duration_DNS = []
        for i in range(args.runs):
            dataset = load_data(args.dataset, num_train_per_class, args.test_size)
            data = dataset[0].to(device)
            model = GCN(dataset.num_node_features, args.layers, dataset.num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            if args.dataset in ['cora', 'citeseer', 'pubmed']:
                train_mask, test_mask = data.train_mask, data.test_mask
            else:
                train_mask, test_mask = random_masking(data, num_train_per_class, dataset.num_classes, args.test_size)
            test_acc, t = train(args.epochs, model, data, optimizer, train_mask, test_mask)
            acc.append(test_acc)
            duration.append(t)

            data = dataset[0]
            data.to(device)
            model2 = GCN(dataset.num_node_features, args.layers, dataset.num_classes).to(device)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01, weight_decay=5e-4)
            data_index = np.arange(data.y.shape[0])
            select_mask = torch.tensor(np.in1d(data_index, select_idx), dtype=torch.bool)
            np.random.shuffle(remain)
            test_idx = remain[:args.test_size]
            test_mask = torch.tensor(np.in1d(data_index, test_idx), dtype=torch.bool)

            test_acc, t = train(args.epochs, model2, data, optimizer2, select_mask, test_mask)
            acc_DNS.append(test_acc)
            duration_DNS.append(t)

        acc_mean.append(np.mean(acc))
        acc_std.append(np.std(acc))
        duration_mean.append(np.mean(duration))
        acc_DNS_mean.append(np.mean(acc_DNS))
        acc_DNS_std.append(np.std(acc_DNS))
        duration_DNS_mean.append(np.mean(duration_DNS))

    return acc_mean, acc_std, duration_mean, acc_DNS_mean, acc_DNS_std, duration_DNS_mean

def trainer(args, model):
    dns_params = DNS_parameters(args.dataset, model, args.label_rate)
    dataset = load_data(args.dataset, dns_params['num_train_per_class'], args.test_size)
    data = dataset[0]
    select_idx, remain = Determinate_Node_Selection(data, data.y.numpy(), dns_params['dc'], dns_params['lt_num'],
                                                    dns_params['k'], dns_params['l'], dns_params['rootWeight'])
    print('dns_index: ', select_idx.tolist())
    args, model_config = model_parameters(args.dataset, model, args, dns_params)
    if args.dataset == 'photo':
        acc_mean, acc_std, duration_mean = DI_train(args, model_config)
        acc_DNS_mean, acc_DNS_std, duration_DNS_mean = DI_train(args, model_config, DNS=True,
                                                                DNS_idx=select_idx)
    else:
        if model == 'GCN':
            acc_mean, acc_std, duration_mean, acc_DNS_mean, acc_DNS_std, duration_DNS_mean = GCN_train(args, select_idx,
                                                                                                       remain, dns_params['num_train_per_class'])
        elif model in ['lp', 'cotraining', 'selftraining', 'union', 'intersection']:
            acc_mean, acc_std, duration_mean = DI_train(args, model_config)
            # acc_mean, acc_std, duration_mean = 0, 0, 0
            acc_DNS_mean, acc_DNS_std, duration_DNS_mean = DI_train(args, model_config, DNS=True,
                                                                    DNS_idx=select_idx)

    return acc_mean, acc_std, duration_mean, acc_DNS_mean, acc_DNS_std, duration_DNS_mean