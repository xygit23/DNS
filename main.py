import numpy as np

from argument import parse_args
from src.utils import config2string, DNS_parameters
from src.train import trainer




def main():
    args = parse_args()

    for model in args.models:
        config = config2string(args)
        print("Config: ", config)

        acc_mean, acc_std, duration_mean, acc_DNS_mean, acc_DNS_std, duration_DNS_mean = trainer(args, model)

        dns_params = DNS_parameters(args.dataset, model, args.label_rate)
        print("test in {} runs.".format(args.runs))
        print("{:<8}\t{:<16}\t\t\t\t{:<8}\t\t\t{:<8}\t\t\t{:<8}\t\t\t{:<8}".format('DATASET', 'MODEL', 'label_rate', 'ACC',
                                                                        'STD', 'TRAIN_TIME'))
        print("{:<8}\t{:<16}\t\t\t\t{:<8}\t\t\t{:<8.4f}\t\t\t{:<8.4f}\t\t\t{:<8.2f}".format(args.dataset,
                                                                                 model,
                                                                                 str(dns_params['label_rate']) + '%',
                                                                                 np.mean(acc_mean),
                                                                                 np.mean(acc_std),
                                                                                 np.mean(duration_mean),
                                                                                 ))
        print("{:<8}\t{:<16}\t\t\t\t{:<8}\t\t\t{:<8.4f}(+{:<4.3f})\t{:<8.4f}({:<4.3f})\t{:<8.2f}".format(args.dataset,
                                                                                              model + '+DNS',
                                                                                              str(dns_params['label_rate']) + '%',
                                                                                              np.mean(acc_DNS_mean),
                                                                                              np.mean(acc_DNS_mean) - np.mean(acc_mean),
                                                                                              np.mean(acc_DNS_std),
                                                                                              np.mean(acc_DNS_std) - np.mean(acc_std),
                                                                                              np.mean(duration_DNS_mean),
                                                                                              ))


if __name__ == "__main__":
    main()
