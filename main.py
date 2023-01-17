from argument import parse_args
from src.utils import load_data, Determinate_Node_Selection, config2string, DNS_parameters, model_parameters
from src.train import GCN_train
from DI.train import DI_train



def main():
    args = parse_args()

    for model in args.models:
        dns_params = DNS_parameters(args.dataset, model)
        dataset = load_data(args.dataset, dns_params['num_train_per_class'], args.test_size)
        data = dataset[0]
        select_idx, remain = Determinate_Node_Selection(data, data.y.numpy(), dns_params['dc'], dns_params['lt_num'],
                                                        dns_params['k'], dns_params['l'], dns_params['rootWeight'])
        args, model_config = model_parameters(args.dataset, model, args, dns_params)
        config = config2string(args)
        print("Config: ", config)
        if model == 'GCN':
            acc_mean, acc_std, duration_mean, acc_DNS_mean, acc_DNS_std, duration_DNS_mean = GCN_train(args, select_idx, remain, dns_params['num_train_per_class'])
        elif model in ['lp', 'cotraining', 'selftraining', 'union', 'intersection']:
            acc_mean, acc_std, duration_mean = DI_train(args.repeating, args.seed, model_config)
            acc_DNS_mean, acc_DNS_std, duration_DNS_mean = DI_train(args.repeating, args.seed, model_config, DNS=True, DNS_idx=select_idx)
        print("test in {} runs.".format(args.runs))
        print("{:<8}\t{:<8}\t{:<8}\t{:<8}\t\t\t{:<8}\t\t\t{:<8}\t{:<8}".format('DATASET', 'MODEL', 'label_rate', 'ACC',
                                                                        'STD', 'TRAIN_TIME', 'NAME'))
        print("{:<8}\t{:<8}\t{:<8}\t{:<8.4f}\t\t\t{:<8.4f}\t\t\t{:<8.2f}\t{:<8}".format(args.dataset,
                                                                                 model,
                                                                                 dns_params['label_rate'],
                                                                                 acc_mean,
                                                                                 acc_std,
                                                                                 duration_mean,
                                                                                 'GCN'))
        print("{:<8}\t{:<8}\t{:<8}\t{:<8.4f}(+{:<4.3f})\t{:<8.4f}({:<4.3f})\t{:<8.2f}\t{:<8}".format(args.dataset,
                                                                                              model,
                                                                                              dns_params['label_rate'],
                                                                                              acc_DNS_mean,
                                                                                              acc_DNS_mean - acc_mean,
                                                                                              acc_DNS_std,
                                                                                              acc_DNS_std - acc_std,
                                                                                              duration_DNS_mean,
                                                                                              'GCN+DNS'))


if __name__ == "__main__":
    main()