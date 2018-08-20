import os
import argparse
import warnings
import sys


def get_args():
    parser = argparse.ArgumentParser(description='Active Learning in MOSM')

    # gp model 
    parser.add_argument('--max_iterations', default=200, type=int, help='number of training iterations')
    parser.add_argument('--n_components', default=10, type=int, help='number of spectral mixture components')
    
    parser.add_argument('--data_file', default=None, help='pickle file to load data from')
    parser.add_argument('--num_samples', default=40, type=int, help='maximum number of samples')
    parser.add_argument('--alpha', default=0, help='sensing cost factor')
    parser.add_argument('--beta', default=0, help='distance travelled factor')
    parser.add_argument('--heterotopic', action='store_true', help='only one measurement allowed at a location')
    parser.add_argument('--norm_factor', default=1, type=float, help='divide all observations by this factor! (normalize observations)')

    parser.add_argument('--utility', default='mutual_information', help='one from {mutual_information, entropy}')
    parser.add_argument('--num_pretrain_samples', default=20, type=int, help='number of samples in pilot survey for model initialization')
    parser.add_argument('--render', action='store_true')
    
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--id', default=1, type=int, help='unique id of every instance')
    parser.add_argument('--save_dir', default='results', help='save directory')
    parser.add_argument('--eval_only', action='store_true', help='will not save anything in this setting')
    parser.add_argument('--logs_wb', default='results.xls')

    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, str(args.id))
    if not args.eval_only:
        if os.path.exists(args.save_dir):
            warnings.warn('SAVE DIRECTORY ALREADY EXISTS!')
            ch = input('Press c to continue and s to stop: ')
            if ch == 's':
                sys.exit(0)
            elif ch == 'c':
                os.rename(args.save_dir, args.save_dir+'_old')
            elif ch != 'c':
                raise NotImplementedError 

        os.makedirs(args.save_dir)               
    return args