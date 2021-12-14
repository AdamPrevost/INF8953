import argparse
import os
from mdp import MDP
from algorithms import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, choices=['h_PI', 'hm_PI', 'h_lambda_PI', 'NC_hm_PI', 'NC_h_lambda_PI'], default='h_PI')
    parser.add_argument("--mdp_size", type=int, default=15)
    parser.add_argument("--h", type=int, default=5)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--lda", type=float, default=1.)
    parser.add_argument('--save_folder', type=str, default=None)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if args.save_folder is not None:
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        elif not os.path.isdir(args.save_folder):
            raise IOError('Provided save folder is not a proper directory.')

    mdp = MDP(N=args.mdp_size)
    if args.algorithm == 'h_PI':
        policy, V = eval(args.algorithm)(mdp, h=args.h)
    elif args.algorithm in ['h_lambda_PI', 'NC_h_lambda_PI']:
        policy, V = eval(args.algorithm)(mdp, h=args.h, lda=args.lda)
    else:
        policy, V = eval(args.algorithm)(mdp, h=args.h, m=args.m)

    save_file = None if args.save_folder is None else os.path.join(args.save_folder, 'rewards.jpg')
    mdp.displayer.display_rewards(title="MDP's rewards", save_file=save_file)

    save_file = None if args.save_folder is None else os.path.join(args.save_folder, 'state_values.jpg')
    mdp.displayer.display_heatmap(V, save_file=save_file)

    save_file = None if args.save_folder is None else os.path.join(args.save_folder, 'best_actions.jpg')
    mdp.displayer.display_actions(mdp.utils.greedy_policy_to_actions(policy), title="Best actions", save_file=save_file)

    print(f"Number of calls to 'simulator': {mdp.n_calls}")
