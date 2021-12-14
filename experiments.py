import argparse
import json
import matplotlib
import matplotlib.pyplot as plt
import os

import numpy as np
from tqdm.auto import tqdm
from mdp import MDP
from algorithms import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo_type", type=str, choices=['lambda', 'm'], default='lambda')
    parser.add_argument("--mdp_size", type=int, default=10)
    parser.add_argument("--h_values", help='Range of values for h (start, stop, n_step)', type=int, nargs=3, default=[1,20,20])
    parser.add_argument("--m_values", help='Range of values for m (start, stop, n_step)', type=int, nargs=3, default=[1,21,21])
    parser.add_argument("--lambda_values", help='Range of values for lambda (start, stop, n_step)', type=float, nargs=3, default=[0., 1., 21])
    parser.add_argument("--max_calls", type=int, default=4e6)
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--error_range", type=float, default=.3)
    parser.add_argument('--save_folder', type=str, default=None)

    args = parser.parse_args()

    if args.algo_type == 'lambda':
        args.algorithms = ['NC_h_lambda_PI', 'h_lambda_PI']
    else:
        args.algorithms = ['NC_hm_PI', 'hm_PI']

    args.h_values = np.linspace(args.h_values[0], args.h_values[1], args.h_values[2]).astype(int).tolist()
    args.m_values = np.linspace(args.m_values[0], args.m_values[1], args.m_values[2]).astype(int).tolist()
    args.lambda_values = np.linspace(args.lambda_values[0], args.lambda_values[1], int(args.lambda_values[2])).tolist()

    return args


def plot_heatmaps(values, title, fname, args):
    fig, axes = plt.subplots(1, len(args.algorithms), figsize=(16, 8))

    vmin = np.min(values)
    vmax = np.max(values)

    for i in range(len(args.algorithms)):
        im = axes[i].imshow(values[i, :, :], norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
        axes[i].set_title(f'{args.algorithms[i]}')
        axes[i].set_xlabel(f'{args.algo_type}')
        axes[i].set_ylabel('h')

        # axes[i].set_yscale('log')
        axes[i].set_yticks(range(len(args.h_values)))
        axes[i].set_yticklabels([f"{h:d}" for h in np.flip(args.h_values)])

        parameter_values = args.lambda_values if args.algo_type == 'lambda' else args.m_values
        axes[i].set_xticks(range(0, len(parameter_values), 5))
        if args.algo_type == 'lambda':
            axes[i].set_xticklabels([f"{parameter_values[i]:.2f}" for i in range(0, len(parameter_values), 5)])
        else:
            axes[i].set_xticklabels([f"{parameter_values[i]:d}" for i in range(0, len(parameter_values), 5)])

    fig.colorbar(im, ax=axes.ravel().tolist())
    fig.suptitle(title)

    if args.save_folder is not None:
        plt.savefig(os.path.join(args.save_folder, fname))
    else:
        plt.show()


def plot_queries_curve(values, title, fname, args):
    fig, axes = plt.subplots(1, len(args.algorithms), sharey=True, figsize=(16, 8))

    std_err = np.std(values, axis=-1)
    parameter_values = args.lambda_values if args.algo_type == 'lambda' else args.m_values

    for i in range(len(args.algorithms)):
        for j in range(min(5, len(args.h_values))):
            axes[i].errorbar(parameter_values, np.mean(values[i, -(j + 1), :, :], axis=1), std_err[i, -(j + 1), :],
                             label=f"h={args.h_values[j]}")
        axes[i].set_title(f'{args.algorithms[i]}')
        axes[i].set_xlabel(f'{args.algo_type}')
        axes[i].set_ylabel('Total queries')
        axes[i].legend()

    fig.suptitle(title)
    if args.save_folder is not None:
        plt.savefig(os.path.join(args.save_folder, fname))
    else:
        plt.show()


def convergence_time(args):
    print("Computing noise-less convergence time...")
    calls = np.zeros((len(args.algorithms), len(args.h_values), len(args.lambda_values), args.n_runs))

    for i, h in tqdm(enumerate(args.h_values), total=len(args.h_values), desc="Testing different h values", position=0):
        parameter_values = args.lambda_values if args.algo_type == 'lambda' else args.m_values
        for j, p in tqdm(enumerate(parameter_values), total=len(parameter_values), leave=False, desc=f"Testing different {args.algo_type} values", position=1):
            for r in range(args.n_runs):
                mdp = MDP(N=args.mdp_size)
                for k, algo in enumerate(args.algorithms):
                    if args.algo_type == 'lambda':
                        eval(algo)(mdp, h=h, lda=p)
                    else:
                        eval(algo)(mdp, h=h, m=p)
                    calls[k, -(i+1), j, r] += mdp.n_calls
                    mdp.reset()

    mean_calls = np.mean(calls, axis=-1)
    plot_heatmaps(mean_calls, 'Time of convergence for both algorithms (in number of calls)', 'convergence_time.jpg', args)
    plot_queries_curve(calls, 'Convergence time curve for both algorithms', 'convergence_curve', args)


def noisy_performance(args):
    print("Computing noisy performance...")

    errors = np.zeros((len(args.algorithms), len(args.h_values), len(args.lambda_values)))

    for i, h in tqdm(enumerate(args.h_values), total=len(args.h_values), desc="Testing different h values", position=0):
        parameter_values = args.lambda_values if args.algo_type == 'lambda' else args.m_values
        for j, p in tqdm(enumerate(parameter_values), total=len(parameter_values), leave=False,
                         desc=f"Testing different {args.algo_type} values", position=1):
            for _ in range(args.n_runs):
                mdp = MDP(N=args.mdp_size)
                _, v_star = h_PI(mdp, h=h)
                for k, algo in enumerate(args.algorithms):
                    if args.algo_type == 'lambda':
                        _, V = eval(algo)(mdp, h=h, lda=p, max_calls=args.max_calls)
                    else:
                        _, V = eval(algo)(mdp, h=h, m=p, max_calls=args.max_calls)
                    errors[k, -(i + 1), j] += np.max(np.abs(V - v_star)) / args.n_runs
                    mdp.reset()
    plot_heatmaps(errors, 'Distance to optimum value for both algorithms', 'distance_optimum.jpg', args)


if __name__ == '__main__':
    args = parse_args()

    if args.save_folder is not None:
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        elif not os.path.isdir(args.save_folder):
            raise IOError('Provided save folder is not a proper directory.')
        elif len(os.listdir(args.save_folder)):
            raise IOError('Provided save folder is not empty.')

        with open(os.path.join(args.save_folder, 'args.json'), 'w') as f:
            json.dump(vars(args), f)

    convergence_time(args)
    noisy_performance(args)
