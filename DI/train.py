from __future__ import division
from __future__ import print_function

import time
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from copy import deepcopy
from scipy import sparse
from DI.gcn.utils import construct_feed_dict, preprocess_adj, chebyshev_polynomials, \
    load_data, sparse_to_tuple, cotraining, selftraining, lp, union_intersection, preprocess_model_config
from DI.gcn.models import GCN_MLP

import warnings
warnings.filterwarnings('ignore')



def train(model_config, sess, seed, data_split = None, DNS=False, DNS_idx=None):
    # Print model_config
    very_begining = time.time()
    # print('',
    #       'name           : {}'.format(model_config['name']),
    #       'dataset        : {}'.format(model_config['dataset']),
    #       'train_size     : {}'.format(model_config['train_size']),
    #       'learning_rate  : {}'.format(model_config['learning_rate']),
    #       'feature        : {}'.format(model_config['feature']),
    #       sep='\n')

    if data_split:
        adj         = data_split['adj']
        features    = data_split['features']
        y_train     = data_split['y_train']
        y_val       = data_split['y_val']
        y_test      = data_split['y_test']
        train_mask  = data_split['train_mask']
        val_mask    = data_split['val_mask']
        test_mask   = data_split['test_mask']
    else:
        # Load data
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
            load_data(model_config['dataset'],train_size=model_config['train_size'],
                      validation_size=model_config['validation_size'],
                      model_config=model_config, shuffle=model_config['shuffle'], DNS=DNS, DNS_idx=DNS_idx)
        stored_A = model_config['dataset']
        # preprocess_features
        begin = time.time()
        # print(time.time()-begin,'s')
        data_split = {
            'adj' : adj,
            'features' : features,
            'y_train' : y_train,
            'y_val' : y_val,
            'y_test' : y_test,
            'train_mask' : train_mask,
            'val_mask' : val_mask,
            'test_mask' : test_mask,
        }
    laplacian = sparse.diags(adj.sum(1).flat, 0) - adj
    laplacian = laplacian.astype(np.float32).tocoo()
    eta = adj.shape[0]/(adj.sum()/adj.shape[0])**len(model_config['connection'])
    model_config['t'] = (y_train.sum(axis=0)*3*eta/y_train.sum()).astype(np.int64)
    # print('t=',model_config['t'])

    # origin_adj = adj
    if model_config['Model'] == 'GCN':
        pass
    elif model_config['Model'] == 'cotraining':
        y_train, train_mask = cotraining(adj, model_config['t'], model_config['alpha'],
                                         y_train, train_mask, stored_A = stored_A+'_A_I')
    elif model_config['Model'] == 'selftraining':
        with tf.Graph().as_default():
            with tf.Session(config=tf.ConfigProto(
                    intra_op_parallelism_threads=model_config['threads'])) as sub_sess:
                tf.set_random_seed(seed)
                test_acc, test_acc_of_class, prediction, t = train(model_config['Model_to_add_label'], sub_sess, seed, data_split=data_split)
        y_train, train_mask = selftraining(prediction, model_config['t'], y_train, train_mask)
        model_config = model_config['Model_to_predict']
        # print('',
        #       'name           : {}'.format(model_config['name']),
        #       'dataset        : {}'.format(model_config['dataset']),
        #       'train_size     : {}'.format(model_config['train_size']),
        #       'learning_rate  : {}'.format(model_config['learning_rate']),
        #       'feature        : {}'.format(model_config['feature']),
        #       sep='\n')
    elif model_config['Model'] == 'lp':
        stored_A = stored_A + '_A_I'
        test_acc, test_acc_of_class, prediction = lp(adj, model_config['alpha'], y_train, train_mask, y_test,
                                                     stored_A=stored_A)
        # print("Test set results: accuracy= {:.5f}".format(test_acc))
        # print("accuracy of each class=", test_acc_of_class)
        # print("Total time={}s".format(time.time()-very_begining))
        return test_acc, test_acc_of_class, prediction, time.time()-very_begining
    elif model_config['Model'] in ['union','intersection']:
        with tf.Graph().as_default():
            with tf.Session(config=tf.ConfigProto(
                    intra_op_parallelism_threads=model_config['threads'])) as sub_sess:
                tf.set_random_seed(seed)
                test_acc, test_acc_of_class, prediction, t = train(model_config['Model_to_add_label'], sub_sess, seed, data_split=data_split)
        stored_A = stored_A + '_A_I'
        # print(time.time()-very_begining)
        y_train, train_mask = union_intersection(prediction, model_config['t'], y_train, train_mask, adj, model_config['alpha'], stored_A, model_config['Model'])
        # print(time.time()-very_begining)
        model_config = model_config['Model_to_predict']
        # print('',
        #       'name           : {}'.format(model_config['name']),
        #       'dataset        : {}'.format(model_config['dataset']),
        #       'train_size     : {}'.format(model_config['train_size']),
        #       'learning_rate  : {}'.format(model_config['learning_rate']),
        #       'feature        : {}'.format(model_config['feature']),
        #       sep='\n')
    else:
        raise ValueError(
            '''model_config['Model'] must be in [0, 9, 16, 17, 19], but is {} now'''.format(model_config['Model']))

    # Some preprocessing
    if sparse.issparse(features):
        if model_config['connection'] == ['f' for i in range(len(model_config['connection']))]:
            train_features = sparse_to_tuple(features[train_mask])
            val_features = sparse_to_tuple(features[val_mask])
            test_features = sparse_to_tuple(features[test_mask])
        features = sparse_to_tuple(features)
    else:
        train_features = features[train_mask]
        val_features = features[val_mask]
        test_features = features[test_mask]

    if model_config['conv'] == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
    elif model_config['conv'] == 'gcn_unnorm':
        support = [sparse_to_tuple(adj.astype(np.float32))]
        num_supports = 1
    elif model_config['conv'] == 'gcn_noloop':
        support = [preprocess_adj(adj, loop=False)]
        num_supports = 1
    elif model_config['conv'] =='gcn_rw':
        support = [preprocess_adj(adj, type='rw')]
        num_supports = 1
    elif model_config['conv'] in ['cheby', 'chebytheta']:
        # origin_adj_support = chebyshev_polynomials(origin_adj, model_config['max_degree'])
        support = chebyshev_polynomials(adj, model_config['max_degree'])
        num_supports = 1 + model_config['max_degree']
    else:
        raise ValueError('Invalid argument for model_config["conv"]: ' + str(model_config['conv']))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32, name='support' + str(i)) for i in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, name='features') if isinstance(features, tf.SparseTensorValue) else tf.placeholder(tf.float32, shape=[None, features.shape[1]], name='features'),
        'labels': tf.placeholder(tf.int32, name='labels', shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32, name='labels_mask'),
        'dropout': tf.placeholder_with_default(0., name='dropout', shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32, name='num_features_nonzero'),
        # helper variable for sparse dropout
        'laplacian' : tf.SparseTensor(indices=np.vstack([laplacian.row, laplacian.col]).transpose()
                                      , values=laplacian.data, dense_shape=laplacian.shape),
    }

    # Create model
    model = GCN_MLP(model_config, placeholders, input_dim=features[2][1])

    # Random initialize
    sess.run(tf.global_variables_initializer())

    # Initialize FileWriter, saver & variables in graph
    train_writer = None
    valid_writer = None
    saver = None

    # Construct feed dictionary
    if model_config['connection'] == ['f' for i in range(len(model_config['connection']))]:
        train_feed_dict = construct_feed_dict(
            train_features, support,
            y_train[train_mask], np.ones(train_mask.sum(), dtype=np.bool), placeholders)
        train_feed_dict.update({placeholders['dropout']: model_config['dropout']})
        valid_feed_dict = construct_feed_dict(
            val_features, support,
            y_val[val_mask], np.ones(val_mask.sum(), dtype=np.bool), placeholders)
        test_feed_dict = construct_feed_dict(
            test_features, support,
            y_test[test_mask], np.ones(test_mask.sum(), dtype=np.bool), placeholders)
    else:
        train_feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        train_feed_dict.update({placeholders['dropout']: model_config['dropout']})
        valid_feed_dict = construct_feed_dict(features, support, y_val, val_mask, placeholders)
        test_feed_dict = construct_feed_dict(features, support, y_test, test_mask, placeholders)

    # tmp = sess.run([model.prediction, model.sample2label], feed_dict=test_feed_dict)

    # Some support variables
    valid_loss_list = []
    max_valid_acc = 0
    max_train_acc = 0
    t_test = time.time()
    test_cost, test_acc, test_acc_of_class, prediction = sess.run([model.loss, model.accuracy, model.accuracy_of_class, model.prediction], feed_dict=test_feed_dict)
    test_duration = time.time() - t_test
    timer = 0
    begin = time.time()

    # print(time.time() - very_begining)
    if model_config['train']:
        # Train model
        # print('training...')
        for step in range(model_config['epochs']):
            # Training step
            t = time.time()
            sess.run(model.opt_op, feed_dict=train_feed_dict)
            t = time.time()-t
            timer += t
            train_loss, train_acc, train_summary = sess.run([model.loss, model.accuracy, model.summary],
                                                            feed_dict=train_feed_dict)

            # If it's best performence so far, evalue on test set
            if model_config['validate']:
                valid_loss, valid_acc, valid_summary = sess.run(
                    [model.loss, model.accuracy, model.summary],
                    feed_dict=valid_feed_dict)
                valid_loss_list.append(valid_loss)
                if valid_acc > max_valid_acc:
                    max_valid_acc = valid_acc
                    t_test = time.time()
                    test_cost, test_acc, test_acc_of_class = sess.run(
                        [model.loss, model.accuracy, model.accuracy_of_class],
                        feed_dict=test_feed_dict)
                    test_duration = time.time() - t_test
                    prediction = sess.run(model.prediction,train_feed_dict)
                    # if args.verbose:
                    #     print('*', end='')
            else:
                if train_acc > max_train_acc:
                    max_train_acc = train_acc
                    t_test = time.time()
                    test_cost, test_acc, test_acc_of_class = sess.run(
                        [model.loss, model.accuracy, model.accuracy_of_class],
                        feed_dict=test_feed_dict)
                    test_duration = time.time() - t_test
                    prediction = sess.run(model.prediction,train_feed_dict)
                    # if args.verbose:
                    #     print('*', end='')

            # Print results
        #     if args.verbose:
        #         print("Epoch: {:04d}".format(step),
        #               "train_loss= {:.3f}".format(train_loss),
        #               "train_acc= {:.3f}".format(train_acc), end=' ')
        #         if model_config['validate']:
        #             print(
        #               "val_loss=", "{:.3f}".format(valid_loss),
        #               "val_acc= {:.3f}".format(valid_acc),end=' ')
        #         print("time=", "{:.5f}".format(t))
        # else:
        #     # print("Optimization Finished!")
        #     pass

        # Testing
        # print("Test set results:", "cost=", "{:.5f}".format(test_cost),
        #       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
        # print("accuracy of each class=", test_acc_of_class)

    # print("Total time={}s".format(time.time()-very_begining))
    return test_acc, test_acc_of_class, prediction, time.time()-very_begining


def DI_train(args, model_config, DNS=False, DNS_idx=None):
    # model_config = model_config['model_list']
    model_config = preprocess_model_config(model_config)
    acc_mean = []
    acc_std = []
    duration_mean = []
    np.random.seed(args.seed)

    for i in range(args.repeating):
        acc = []
        duration = []
        for j in range(args.runs):
            seed = np.random.random_integers(args.seed)
            # Initialize session
            with tf.Graph().as_default():
                tf.set_random_seed(seed)
                with tf.Session(config=tf.ConfigProto(
                        intra_op_parallelism_threads=model_config['threads'])) as sess:
                    test_acc, test_acc_of_class, prediction, t = train(model_config, sess, seed, DNS=DNS, DNS_idx=DNS_idx)
            acc.append(test_acc)
            duration.append(t)
        acc_mean.append(np.mean(acc))
        acc_std.append(np.std(acc))
        duration_mean.append(np.mean(duration))

    return acc_mean, acc_std, duration_mean







# if __name__ == '__main__':
#
#     acc = [[] for i in configuration['model_list']]
#     acc_of_class = [[] for i in configuration['model_list']]
#     duration = [[] for i in configuration['model_list']]
#     # Read configuration
#     r = 0
#     while r < (configuration['repeating']):
#         for model_config, i in zip(configuration['model_list'], range(len(configuration['model_list']))):
#             # Set random seed
#             seed = model_config['random_seed']
#             np.random.seed(seed)
#             model_config['random_seed'] = np.random.random_integers(1073741824)
#
#             # Initialize session
#             with tf.Graph().as_default():
#                 tf.set_random_seed(seed)
#                 with tf.Session(config=tf.ConfigProto(
#                         intra_op_parallelism_threads=model_config['threads'])) as sess:
#                     test_acc, test_acc_of_class, prediction, t = train(model_config, sess, seed)
#                     acc[i].append(test_acc)
#                     acc_of_class[i].append(test_acc_of_class)
#                     duration[i].append(t)
#         print('repeated ', r + 1, 'rounds')
#         r += 1
#
#     acc_means = np.mean(acc, axis=1)
#     acc_stds = np.std(acc, axis=1)
#     acc_of_class_means = np.mean(acc_of_class, axis=1)
#     duration = np.mean(duration, axis=1)
#     # print mean, standard deviation, and model name
#
#     print("REPEAT\t{}".format(configuration['repeating']))
#     print("{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}".format('DATASET', 'train_size', 'valid_size', 'RESULTS', 'STD', 'TRAIN_TIME', 'NAME'))
#     for model_config, acc_mean, acc_std, t in zip(configuration['model_list'], acc_means, acc_stds, duration):
#         print("{:<8}\t{:<8}\t{:<8}\t{:<8.6f}\t{:<8.6f}\t{:<8.2f}\t{:<8}".format(model_config['dataset'],
#                                                                           str(model_config['train_size']) + ' per class',
#                                                                           str(model_config['validation_size']),
#                                                                           acc_mean,
#                                                                           acc_std,
#                                                                           t,
#                                                                           model_config['name']))
#     print('acc in {} runs.'.format(len(acc[0])))
#
#     for model_config, acc_of_class_mean in zip(configuration['model_list'], acc_of_class_means):
#         print('[',end='')
#         for acc_of_class in acc_of_class_mean:
#             print('{:0<5.3}'.format(acc_of_class),end=', ')
#         print(']',end='')
#         print('\t{:<8}'.format(model_config['name']))
