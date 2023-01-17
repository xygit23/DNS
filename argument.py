import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="citeseer", help="cora, citeseer, pubmed")
    parser.add_argument("--models", nargs='+', default=['GCN', 'lp', 'cotraining', 'selftraining', 'union', 'intersection'], help="'GCN', 'lp', 'cotraining', 'selftraining', 'union', 'intersection'")

    parser.add_argument("--repeating", type=int, default=10, help="The number of repeat times.")
    parser.add_argument("--runs", type=int, default=10, help="The number of runs.")
    parser.add_argument("--test_size", type=int, default=1000)

    parser.add_argument("--epochs", type=int, default=200, help="The number of epochs.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate. Default is 0.01.")
    parser.add_argument("--decay", type=float, default=5e-4, help="L2 Regularization for weight. Default is 5e-4.")

    # The following parameters is optional since we provided the recommend parameters.
    # parser.add_argument("--train_size", type=int, default=18, help="The number of labeled samples in per class")
    # parser.add_argument("--layers", nargs='+', default=[16], help="The number of hidden units of each layer of the GCN.")
    # parser.add_argument("--label_rate", type=int, default=1, help="The rate of labeled samples(%) for recommend parameters.")
    # parser.add_argument("--dc", type=float, default=2.5, help="Bandwidth parameter for GOLF.")
    # parser.add_argument("--lt_num", type=int, default=79, help="The number of Leading Tree.")
    # parser.add_argument("--k", type=int, default=11, help="The number of samples selected from per class.")
    # # parser.add_argument("--l", type=int, default=112, help="The number of labeled samples.")
    # parser.add_argument("--rootWeight", type=float, default=2.67, help="A parameter to adjust the weight between typcialness and divergence of labeled samples.")
    # parser.add_argument("--seed", type=int, default=int(time.time()))
    
    return parser.parse_known_args()[0]
