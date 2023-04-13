from __future__ import division
import time
from copy import deepcopy
from os import cpu_count
from DI.gcn.utils import preprocess_model_config
import argparse
import pprint
# import math

# configuration ={
#     # repeating times
#     'repeating'             : 10,
#
#     # The default model configuration
#     'default':{
#         'dataset'           : 'citeseer',     # 'Dataset string. (cora | citeseer | pubmed)'
#         'shuffle'           : True,
#         'train_size'        : 45,         # if train_size is a number, then use TRAIN_SIZE labels per class.
#         # 'train_size'        : [2234, 2048, 1292, 1821, 2623, 646, 23, 2046, 887, 910, 1251, 1448, 1779, 2318],
#         # 'train_size'        : [20 for i in range(10)], # if train_size is a list of numbers, then it specifies training labels for each class.
#         'validation_size'   : 0,           # 'Use VALIDATION_SIZE data to train model'
#         'validate'          : False,        # Whether use validation set
#         'test_size'         : 1000,         # If None, all rest are test set
#         'conv'              : 'gcn',        # 'conv type. (gcn | cheby | chebytheta | gcn_rw | taubin | test21 | gcn_unnorm | gcn_noloop)'
#         'max_degree'        : 2,            # 'Maximum Chebyshev polynomial degree.'
#         'learning_rate'     : 0.01,         # 'Initial learning rate.'
#         'epochs'            : 200,          # 'Number of epochs to train.'
#
#         # config the absorption probability
#         'Model'             : 0,
#         'Model_to_add_label': { 'Model' :0 }, # for model 16
#         'Model_to_predict'  : { 'Model' :0 }, # for model 16
#         'Model19'           : 'union',        # 'union' | 'intersection'
#         'alpha'             : 1e-6,
#
#         'connection'        : 'cc',
#         # A string contains only char "c" or "f" or "d".
#         # "c" stands for convolution.
#         # "f" stands for fully connected.
#         # "d" stands for dense net.
#         # See layer_size for details.
#
#         'layer_size'        : [16],
#         # A list or any sequential object. Describe the size of each layer.
#         # e.g. "--connection ccd --layer_size [7,8]"
#         #     This combination describe a network as follow:
#         #     input_layer --convolution-> 7 nodes --convolution-> 8 nodes --dense-> output_layer
#         #     (or say: input_layer -c-> 7 -c-> 8 -d-> output_layer)
#
#         'dropout'           : 0.5,          # 'Dropout rate (1 - keep probability).'
#         'weight_decay'      : 5e-4,         # 'Weight for L2 loss on embedding matrix.'
#
#         'random_seed'       : 0,     #'Random seed.'
#         'feature'           : 'bow',        # 'bow' | 'tfidf' | 'none'.
#
#         'model_dir'         : './model/',
#         'name'              : None,           # 'name of the model. Serve as an ID of model.'
#         # if logdir or name are empty string,
#         # logdir or name will be auto generated according to the
#         # structure of the network
#
#         'threads'           : 2*cpu_count(),  #'Number of threads'
#         'train'             : True,
#     },
#
#     # The list of model to be train.
#     # Only configurations that's different with default are specified here
#     'model_list':
#     [
#         {
#             'Model'     : 'GCN',
#         },
#         # {
#         #     'Model'     : 'cotraining',
#         # },
#         # {
#         #     'Model'     : 'selftraining',
#         # },
#         # {
#         #     'Model'     : 'lp',
#         # },
#         # {
#         #     'Model'     : 'union',
#         # },
#         # {
#         #     'Model'     : 'intersection',
#         # },
#     ]
# }

# Parse args
# parser = argparse.ArgumentParser(description=(
#     "This is used to train and test Graph Convolution Network for node classification problem.\n"
#     "Most configuration are specified in config.py, please read it and modify it as you want."))
# parser.add_argument("-v", "--verbose", action="store_true")
# parser.add_argument("--dataset", type=str)
# parser.add_argument("--train_size", type=str)
# parser.add_argument("--repeating", type=int)
#
# args = parser.parse_args()
# print(args)
# if args.dataset is not None:
#     configuration['default']['dataset'] = args.dataset
# if args.train_size is not None:
#     configuration['default']['train_size'] = eval(args.train_size)
# if args.repeating is not None:
#     configuration['repeating']=args.repeating
# pprint.PrettyPrinter(indent=4).pprint(configuration)
# exit()

def set_default_attr(configuration, model):
    model_config = deepcopy(configuration['default'])
    model_config.update(model)
    return model_config

# configuration['model_list'] = list(map(set_default_attr,
#     configuration['model_list']))
#
# for model_config in configuration['model_list']:
#     preprocess_model_config(model_config)
