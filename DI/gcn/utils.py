from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as slinalg
import scipy.linalg as linalg
# from scipy.sparse.linalg.eigen.arpack import eigsh
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
import sys
from os import path
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import random
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
# import matplotlib.pyplot as plt
# import MyLeadingTreeGraph as lt
# from DeLaLA_select import DeLaLA_select

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

def get_triplet(y_train, train_mask, max_triplets):
#    print('y_train----',y_train.shape)        
    index_nonzero = y_train.nonzero()
#    for i in range(y_train.shape[1]):
#        label_count.append(index_nonzero[1][[index_nonzero[1]==i]].size)
    label_count = np.sum(y_train, axis=0)
    all_count = np.sum(label_count)
    
    index_nonzero = np.transpose(np.concatenate((index_nonzero[0][np.newaxis,:], index_nonzero[1]\
                                                 [np.newaxis, :]),axis=0)).tolist()
        
    index_nonzero = sorted(index_nonzero, key = lambda s: s[1])
    #print(index_nonzero)
    #print(label_count)
 
    def get_one_triplet(input_index, index_nonzero, label_count, all_count, max_triplets):
        triplet = []
        if label_count[input_index[1]]==0:
            return 0
        else:
 #           print('max_triplets', max_triplets)
  #          print(all_count)
   #         print(label_count[input_index[1]])
            n_triplets = min(max_triplets, int(all_count-label_count[input_index[1]]))
   #         print('----------')

            for j in range(int(label_count[input_index[1]])-1):
                positives = []
                negatives = []           
                for k, (value, label) in enumerate(index_nonzero):
                    #find a postive sample, and if only one sample then choose itself
                    if label == input_index[1] and (value != input_index[0] or label_count[input_index[1]]==1):
                        positives.append(index_nonzero[k])
                    if label != input_index[1]:
                        negatives.append(index_nonzero[k])
 #               print('positives' ,positives)
 #               print('negatives', negatives)
                negatives = random.sample(list(negatives), n_triplets)
                for value, label in negatives:
                    triplet.append([input_index[0], positives[j][0], value])
            return triplet
                
                                   
    triplet = []
    for i, j in enumerate(index_nonzero):
        triple = get_one_triplet(j, index_nonzero, label_count, all_count,max_triplets)
        
        if triple == 0:
            continue
        else:
            triplet.extend(triple)  
    np_triple = np.concatenate(np.array([triplet]), axis = 1)
    return np_triple

def load_data(dataset_str, train_size, validation_size, model_config, shuffle=True, DNS=False, DNS_idx=None):
    """Load data."""
    if dataset_str in ['USPS-Fea', 'CIFAR-Fea', 'Cifar_10000_fea', 'Cifar_R10000_fea', 'MNIST-Fea', 'MNIST-10000', 'MNIST-5000']:
        data = sio.loadmat(os.path.join(os.path.abspath('..'), 'data/{}.mat'.format(dataset_str)))
        l = data['labels'].flatten()
        labels = np.zeros([l.shape[0],np.max(data['labels'])+1])
        labels[np.arange(l.shape[0]), l.astype(np.int8)] = 1
        features = data['X']
        sample = features[0].copy()
        adj = data['G']
    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(os.path.join(os.path.abspath('.'), "data/{}/{}/raw/ind.{}.{}".format(dataset_str, dataset_str, dataset_str, names[i])), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        test_idx_reorder = parse_index_file(os.path.join(os.path.abspath('.'), "data/{}/{}/raw/ind.{}.test.index".format(dataset_str, dataset_str, dataset_str)))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        # features = sp.eye(features.shape[0]).tolil()
        # features = sp.lil_matrix(allx)

        labels = np.vstack((ally, ty))
        # labels = np.vstack(ally)

        if dataset_str.startswith('nell'):
            # Find relation nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(allx.shape[0], len(graph))
            isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - allx.shape[0], :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - allx.shape[0], :] = ty
            ty = ty_extended

            features = sp.vstack((allx, tx)).tolil()
            features[test_idx_reorder, :] = features[test_idx_range, :]
            labels = np.vstack((ally, ty))
            labels[test_idx_reorder, :] = labels[test_idx_range, :]

            idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)

            if not os.path.isfile("data/planetoid/{}.features.npz".format(dataset_str)):
                print("Creating feature vectors for relations - this might take a while...")
                features_extended = sp.hstack((features, sp.lil_matrix((features.shape[0], len(isolated_node_idx)))),
                                              dtype=np.int32).todense()
                features_extended[isolated_node_idx, features.shape[1]:] = np.eye(len(isolated_node_idx))
                features = sp.csr_matrix(features_extended, dtype=np.float32)
                print("Done!")
                save_sparse_csr("data/planetoid/{}.features".format(dataset_str), features)
            else:
                features = load_sparse_csr("data/planetoid/{}.features.npz".format(dataset_str))

            adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        features[test_idx_reorder, :] = features[test_idx_range, :]
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        features = preprocess_features(features, feature_type=model_config['feature'])

    global all_labels
    all_labels = labels.copy()

    # split the data set
    idx = np.arange(len(labels))
    no_class = labels.shape[1]  # number of class
    # validation_size = validation_size * len(idx) // 100
    # if not hasattr(train_size, '__getitem__'):
    train_size = [train_size for i in range(labels.shape[1])]
    if shuffle:
        np.random.shuffle(idx)
    idx_train = []
    count = [0 for i in range(no_class)]
    label_each_class = train_size
    next = 0
    for i in idx:
        if count == label_each_class:
            break
        next += 1
        for j in range(no_class):
            if labels[i, j] and count[j] < label_each_class[j]:
                idx_train.append(i)
                count[j] += 1

    if DNS is True:
    # DNS select
        idx_train = DNS_idx
        idx = np.arange(len(labels))
        idx = np.delete(idx, idx_train)
        idx_train = idx_train.tolist()
        idx = idx.tolist()
        if shuffle:
            np.random.shuffle(idx)

    test_size = model_config['test_size']
    if model_config['validate']:
        if test_size:
            assert next+validation_size<len(idx)
        idx_val = idx[next:next+validation_size]
        assert next+validation_size+test_size < len(idx)
        idx_test = idx[-test_size:] if test_size else idx[next+validation_size:]

    else:
        if test_size:
            assert next+test_size<len(idx)
        idx_val = idx[-test_size:] if test_size else idx[next:]
        idx_test = idx[-test_size:] if test_size else idx[next:]
    # else:
    #     labels_of_class = [0]
    #     while (np.prod(labels_of_class) == 0):
    #         np.random.shuffle(idx)
    #         idx_train = idx[0:int(len(idx) * train_size // 100)]
    #         labels_of_class = np.sum(labels[idx_train], axis=0)
    #     idx_val = idx[-500 - validation_size:-500]
    #     idx_test = idx[-500:]
    # print('labels of each class : ', np.sum(labels[idx_train], axis=0))
    # idx_val = idx[len(idx) * train_size // 100:len(idx) * (train_size // 2 + 50) // 100]
    # idx_test = idx[len(idx) * (train_size // 2 + 50) // 100:len(idx)]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    # else:
    #     idx_test = test_idx_range.tolist()
    #     idx_train = range(len(y))
    #     idx_val = range(len(y), len(y) + 500)
    #
    #     train_mask = sample_mask(idx_train, labels.shape[0])
    #     val_mask = sample_mask(idx_val, labels.shape[0])
    #     test_mask = sample_mask(idx_test, labels.shape[0])
    #
    #     y_train = np.zeros(labels.shape)
    #     y_val = np.zeros(labels.shape)
    #     y_test = np.zeros(labels.shape)
    #     y_train[train_mask, :] = labels[train_mask, :]
    #     y_val[val_mask, :] = labels[val_mask, :]
    #     y_test[test_mask, :] = labels[test_mask, :]

    size_of_each_class = np.sum(labels[idx_train], axis=0)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return tf.SparseTensorValue(coords, values, np.array(shape, dtype=np.int64))

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features, feature_type):
    if feature_type == 'bow':
        # """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        # normalize(features, norm='l1', axis=1, copy=False)
    elif feature_type == 'tfidf':
        transformer = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False)
        features = transformer.fit_transform(features)
    elif feature_type == 'none':
        features = sp.csr_matrix(sp.eye(features.shape[0]))
    else:
        raise ValueError('Invalid feature type: ' + str(feature_type))
    return features


def normalize_adj(adj, type='sym'):
    """Symmetrically normalize adjacency matrix."""
    if type == 'sym':
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        # d_inv_sqrt = np.power(rowsum, -0.5)
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # return adj*d_inv_sqrt*d_inv_sqrt.flatten()
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    elif type == 'rw':
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj_normalized = d_mat_inv.dot(adj)
        return adj_normalized


def preprocess_adj(adj, type='sym', loop=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if loop:
        adj = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj, type=type)  #
    return sparse_to_tuple(adj_normalized)


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    # largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    # scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], laplacian))

    return sparse_to_tuple(t_k)

def absorption_probability(W, alpha, stored_A=None, column=None):
    try:
        # raise Exception('DEBUG')
        A = np.load(stored_A + str(alpha) + '.npz')['arr_0']
        # print('load A from ' + stored_A + str(alpha) + '.npz')
        if column is not None:
            P = np.zeros(W.shape)
            P[:, column] = A[:, column]
            return P
        else:
            return A
    except:
        # W=sp.csr_matrix([[0,1],[1,0]])
        # alpha = 1
        n = W.shape[0]
        # print('Calculate absorption probability...')
        W = W.copy().astype(np.float32)
        D = W.sum(1).flat
        L = sp.diags(D, dtype=np.float32) - W
        L += alpha * sp.eye(W.shape[0], dtype=L.dtype)
        L = sp.csc_matrix(L)
        # print(np.linalg.det(L))

        if column is not None:
            A = np.zeros(W.shape)
            # start = time.time()
            A[:, column] = slinalg.spsolve(L, sp.csc_matrix(np.eye(L.shape[0], dtype='float32')[:, column])).toarray()
            # print(time.time()-start)
            return A
        else:
            # start = time.time()
            A = slinalg.inv(L).toarray()
            # print(time.time()-start)
            if stored_A:
                np.savez(stored_A + str(alpha) + '.npz', A)
            return A
            # fletcher_reeves

            # slinalg.solve(L, np.ones(L.shape[0]))
            # A_ = np.zeros(W.shape)
            # I = sp.eye(n)
            # Di = sp.diags(np.divide(1,np.array(D)+alpha))
            # for i in range(10):
            #     # A_=
            #     A_ = Di*(I+W.dot(A_))
            # print(time.time()-start)


def fletcher_reeves(A, B):
    # A=np.array(A)
    X = np.zeros(B.shape)
    r = np.array(B - A.dot(X))
    rsold = (r * r).sum(0)
    p = r
    for i in range(10):
        Ap = np.array(A.dot(p))
        pAp = (p * Ap).sum(0)
        alpha = rsold / pAp
        X += alpha * p
        r -= alpha * Ap
        rsnew = (r * r).sum(0)
        if True:
            pass
        p = r + rsnew / rsold * p
        rsold = rsnew
    return X


def cotraining(W, t, alpha, y_train, train_mask, stored_A=None):
    A = absorption_probability(W, alpha, stored_A, train_mask)
    y_train = y_train.copy()
    train_index = np.where(train_mask)[0]
    already_labeled = np.sum(y_train, axis=1)
    # if not isinstance(features, np.ndarray):
    #     features = features.toarray()
    # print("Additional Label:")
    if not hasattr(t, '__getitem__'):
        t = [t for _ in range(y_train.shape[1])]
    for i in range(y_train.shape[1]):
        y = y_train[:, i:i + 1]
        a = A.dot(y)
        a[already_labeled > 0] = 0
        # a[W.dot(y) > 0] = 0
        gate = (-np.sort(-a, axis=0))[t[i]]
        index = np.where(a.flat > gate)[0]

        # x1 = features[index, :].reshape((-1, 1, features.shape[1]))
        # x2 = features[y_train[:, i].astype(np.bool)].reshape((1, -1, features.shape[1]))
        # D = np.sum((x1 - x2) ** 2, axis=2) ** 0.5
        # D = np.mean(D, axis=1)
        # gate = 100000000 if t[i] >= D.shape[0] else np.sort(D, axis=0)[t[i]]
        # index = index[D<gate]
        train_index = np.hstack([train_index, index])
        y_train[index, i] = 1
        correct_label_count(index, i)
    # print()
    train_mask = sample_mask(train_index, y_train.shape[0])
    return y_train, train_mask


def selftraining(prediction, t, y_train, train_mask):
    new_gcn_index = np.argmax(prediction, axis=1)
    confidence = np.max(prediction, axis=1)
    sorted_index = np.argsort(-confidence)

    no_class = y_train.shape[1]  # number of class
    if hasattr(t, '__getitem__'):
        assert len(t) >= no_class
        index = []
        count = [0 for i in range(no_class)]
        for i in sorted_index:
            for j in range(no_class):
                if new_gcn_index[i] == j and count[j] < t[j] and not train_mask[i]:
                    index.append(i)
                    count[j] += 1
    else:
        index = sorted_index[:t]
    indicator = np.zeros(train_mask.shape, dtype=np.bool)
    indicator[index] = True
    indicator = np.logical_and(np.logical_not(train_mask), indicator)

    prediction = np.zeros(prediction.shape)
    prediction[np.arange(len(new_gcn_index)), new_gcn_index] = 1.0
    prediction[train_mask] = y_train[train_mask]

    correct_labels = np.sum(prediction[indicator] * all_labels[indicator], axis=0)
    count = np.sum(prediction[indicator], axis=0)
    # print('Additiona Label:')
    # for i, j in zip(correct_labels, count):
    #     print(int(i), '/', int(j), sep='', end='\t')
    # print()

    y_train = np.copy(y_train)
    train_mask = np.copy(train_mask)
    train_mask[indicator] = 1
    y_train[indicator] = prediction[indicator]

    # #add
    # temp = np.arange(train_mask.shape[0])
    # print('Sample selection by confidence:')
    # print(temp[train_mask][:84].tolist())
    # print('size:', temp[train_mask].shape)
    return y_train, train_mask


def lp(adj, alpha, y_train, train_mask, y_test, stored_A=None):
    P = absorption_probability(adj, alpha, stored_A=stored_A, column=train_mask)
    P = P[:, train_mask]

    # nearest clssifier
    predicted_labels = np.argmax(P, axis=1)
    # prediction = alpha*P
    prediction = np.zeros(P.shape)
    prediction[np.arange(P.shape[0]), predicted_labels] = 1

    y = np.sum(train_mask)
    label_per_sample = np.vstack([np.zeros(y), np.eye(y)])[np.add.accumulate(train_mask) * train_mask]
    sample2label = label_per_sample.T.dot(y_train)
    prediction = prediction.dot(sample2label)

    test_acc = np.sum(prediction * y_test) / np.sum(y_test)
    test_acc_of_class = np.sum(prediction * y_test, axis=0) / np.sum(y_test, axis=0)
    # print(test_acc, test_acc_of_class)
    return test_acc, test_acc_of_class, prediction


def union_intersection(prediction, t, y_train, train_mask, W, alpha, stored_A, union_or_intersection):
    no_class = y_train.shape[1]  # number of class

    # gcn index
    new_labels_gcn = np.argmax(prediction, axis=1)
    confidence = np.max(prediction, axis=1)
    sorted_index = np.argsort(-confidence)

    if not hasattr(t, '__getitem__'):
        t = [t for i in range(no_class)]

    assert len(t) >= no_class
    count = [0 for i in range(no_class)]
    index_gcn = [[] for i in range(no_class)]
    for i in sorted_index:
        j = new_labels_gcn[i]
        if count[j] < t[j] and not train_mask[i]:
            index_gcn[j].append(i)
            count[j] += 1

    # lp
    A = absorption_probability(W, alpha, stored_A, train_mask)
    train_index = np.where(train_mask)[0]
    already_labeled = np.sum(y_train, axis=1)
    index_lp = []
    for i in range(no_class):
        y = y_train[:, i:i + 1]
        a = np.sum(A[:, y.flat > 0], axis=1)
        a[already_labeled > 0] = 0
        # a[W.dot(y) > 0] = 0
        gate = (-np.sort(-a, axis=0))[t[i]]
        index = np.where(a.flat > gate)[0]
        index_lp.append(index)

    # print(list(map(len, index_gcn)))
    # print(list(map(len, index_lp)))

    y_train = y_train.copy()
    # print("Additional Label:")
    for i in range(no_class):
        assert union_or_intersection in ['union', 'intersection']
        if union_or_intersection == 'union':
            index = list(set(index_gcn[i]) | set(index_lp[i]))
        else:
            index = list(set(index_gcn[i]) & set(index_lp[i]))
        y_train[index, i] = 1
        train_mask[index] = True
        # print(np.sum(all_labels[index, i]), '/', len(index), sep='', end='\t')
    return y_train, train_mask


def ap_approximate(adj, features, alpha, k):
    adj = normalize(adj + sp.eye(adj.shape[0]), 'l1', axis=1) / (alpha + 1)
    # D = sp.diags(np.array(adj.sum(axis=1)).flatten())+alpha*sp.eye(adj.shape[0])
    # D = D.power(-1)
    # adj = D*adj
    # features = D*alpha*features
    if sp.issparse(features):
        features = features.toarray()
    new_feature = np.zeros(features.shape)
    for _ in range(k):
        new_feature = adj * new_feature + features
    new_feature *= alpha / (alpha + 1)
    return new_feature

all_labels = None


# dataset = None

def correct_label_count(indicator, i):
    count = np.sum(all_labels[:, i][indicator])
    if indicator.dtype == np.bool:
        total = np.where(indicator)[0].shape[0]
    elif indicator.dtype in [np.int, np.int8, np.int16, np.int32, np.int64]:
        total = indicator.shape[0]
    else:
        raise TypeError('indicator must be of data type np.bool or np.int')
    # print("     for class {}, {}/{} is correct".format(i, count, total))
    # print(count, '/', total, sep='', end='\t')


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def preprocess_model_config(model_config):
    if model_config['Model'] not in [17, 23]:
        model_config['connection'] = list(model_config['connection'])
        # judge if parameters are legal
        for c in model_config['connection']:
            if c not in ['c', 'd', 'r', 'f', 'C']:
                raise ValueError(
                    'connection string specified by --connection can only contain "c", "d", "r", "f", "C" but "{}" found'.format(
                        c))
        for i in model_config['layer_size']:
            if not isinstance(i, int):
                raise ValueError('layer_size should be a list of int, but found {}'.format(model_config['layer_size']))
            if i <= 0:
                raise ValueError('layer_size must be greater than 0, but found {}' % i)
        if not len(model_config['connection']) == len(model_config['layer_size']) + 1:
            raise ValueError('length of connection string should be equal to length of layer_size list plus 1')

    # Generate name
    if not model_config['name']:
        model_name = str(model_config['Model'])
        if model_config['Model'] != 'lp':
            model_name += '_' + model_config['connection'][0]
            for char, size in \
                    zip(model_config['connection'][1:], model_config['layer_size']):
                model_name += str(size) + char

            if model_config['conv'] == 'cheby':
                model_name += '_cheby' + str(model_config['max_degree'])
            elif model_config['conv'] == 'taubin':
                model_name += '_conv_taubin' + str(model_config['taubin_lambda']) \
                              + '_' + str(model_config['taubin_mu']) \
                              + '_' + str(model_config['taubin_repeat'])
            elif model_config['conv'] == 'test21':
                model_name += '_' + 'conv_test21' + '_' + str(model_config['alpha']) + '_' + str(model_config['beta'])
            elif model_config['conv'] == 'gcn_unnorm':
                model_name += '_' + 'gcn_unnorm'
            elif model_config['conv'] == 'gcn_noloop':
                model_name += '_' + 'gcn_noloop'
            if model_config['validate']:
                model_name += '_validate'

        if model_config['Model'] == 'cotraining':
            model_name += '_alpha_' + str(
                model_config['alpha'])
        # if model_config['Model'] == 'selftraining':
        #     Model_to_add_label = copy.deepcopy(model_config)
        #     if 'Model_to_add_label' in Model_to_add_label:
        #         del Model_to_add_label['Model_to_add_label']
        #     if 'Model_to_predict' in Model_to_add_label:
        #         del Model_to_add_label['Model_to_predict']
        #     Model_to_add_label.update({'Model': 'GCN'})
        #     model_config['Model_to_add_label'] = Model_to_add_label
        #     preprocess_model_config(model_config['Model_to_add_label'])
        #
        #     Model_to_predict = copy.deepcopy(model_config)
        #     if 'Model_to_add_label' in Model_to_predict:
        #         del Model_to_predict['Model_to_add_label']
        #     if 'Model_to_predict' in Model_to_predict:
        #         del Model_to_predict['Model_to_predict']
        #     Model_to_predict.update({'Model': 'GCN'})
        #     model_config['Model_to_predict'] = Model_to_predict
        #     preprocess_model_config(model_config['Model_to_predict'])
        #     model_name = 'Model' + str(model_config['Model']) \
        #                  + '_{' + model_config['Model_to_add_label']['name'] + '}' \
        #                  + '_{' + model_config['Model_to_predict']['name'] + '}'
        if model_config['Model'] in ['union', 'intersection','lp']:
            model_name += '_alpha_' + str(model_config['alpha'])

        if model_config['Model'] in ['union', 'intersection', 'selftraining']:
            Model_to_add_label = copy.deepcopy(model_config)
            if 'Model_to_add_label' in Model_to_add_label:
                del Model_to_add_label['Model_to_add_label']
            if 'Model_to_predict' in Model_to_add_label:
                del Model_to_add_label['Model_to_predict']
            Model_to_add_label.update({'Model': 'GCN'})
            model_config['Model_to_add_label'] = Model_to_add_label
            preprocess_model_config(model_config['Model_to_add_label'])

            Model_to_predict = copy.deepcopy(model_config)
            if 'Model_to_add_label' in Model_to_predict:
                del Model_to_predict['Model_to_add_label']
            if 'Model_to_predict' in Model_to_predict:
                del Model_to_predict['Model_to_predict']
            Model_to_predict.update({'Model': 'GCN'})
            model_config['Model_to_predict'] = Model_to_predict
            preprocess_model_config(model_config['Model_to_predict'])


        model_config['name'] = model_name

    return  model_config


if __name__ == '__main__':
    pass