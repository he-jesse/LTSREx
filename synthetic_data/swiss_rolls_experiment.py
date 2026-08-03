import sys
sys.path.append('../')

import numpy as np
import warnings

from lego import lego
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.neighbors import kneighbors_graph
from local_ts_regression import lpca, run_ts_regression, run_ts_regression_voting, run_ts_regression_smoothed

from two_swiss_rolls import two_swiss_rolls

def run_experiment(seed, noise, k, smoothing=False):
    np.random.seed(seed)
    d = 2
    X, F = two_swiss_rolls(n_samples=2500, offset = 50.0, noise=noise, outliers=0, seed=seed)
    F_names = ['s1', 's2', 't1', 't2']
    regressor = MultiTaskElasticNet(alpha=0.5, l1_ratio=0.5, fit_intercept=False)

    neighbor_graph = kneighbors_graph(X, k, include_self=True)

    trivial_basis = np.stack([np.eye(X.shape[1])] * X.shape[0], axis=0)
    lpca_basis = lpca(X, d, neighbor_graph)
    wlpca_basis = lpca(X, d, neighbor_graph, r=1.0)

    ts_method = 'lego'
    n_eig_for_grad = 40
    n_eig = 100
    lego_estimate = lego.lego(
        X,
        opts={'emb_dim': d, 'k_nn': k, 'n_eig_for_grad': n_eig_for_grad}, 
        gl_opts={'n_eig': n_eig}
    )
    lego_basis = lego_estimate['tang_basis']

    tsbases = [trivial_basis, lpca_basis, wlpca_basis, lego_basis]
    ts_methods = ['trivial', 'lpca', 'wlpca', 'lego']
    ts_correct = {method: 0 for method in ts_methods}

    subsample_size = X.shape[0] // 10
    subsample = np.random.choice(X.shape[0], size=subsample_size, replace=False)
    for index in subsample:
        for method, tsbasis in zip(ts_methods, tsbases):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if smoothing:
                    reg_coefs, _ = run_ts_regression_smoothed(X, F, index, neighbor_graph, tsbasis, regressor=regressor, d=2, gamma=1.0, k=1)
                else:
                    reg_coefs, _ = run_ts_regression(X, F, index, neighbor_graph, tsbasis, d=d, regressor=regressor)
            strengths = np.linalg.norm(reg_coefs, axis=0)
            support = np.where(strengths > 0)[0]
            if index < X.shape[0] // 2 and set(support) == set([0, 2]):
                ts_correct[method] += 1
            elif index >= X.shape[0] // 2 and set(support) == set([1, 3]):
                ts_correct[method] += 1
    
    np.savez(f'results/swiss_rolls_results_seed={seed}_noise={noise}_k={k}_smoothing={smoothing}.npz',
             trivial_correct=ts_correct['trivial'] / subsample_size,
             lpca_correct=ts_correct['lpca'] / subsample_size,
             wlpca_correct=ts_correct['wlpca'] / subsample_size,
             lego_correct=ts_correct['lego'] / subsample_size)
    
def main():
    seeds = [0, 1, 2, 3, 4]
    noise_levels = [0.0, 0.5, 1.0, 1.5, 2.0]
    ks = [8, 16, 32]
    for seed in seeds:
        for noise in noise_levels:
            for k in ks:
                print(f'Running experiment with seed={seed}, noise={noise}, k={k}, smoothing=False')
                run_experiment(seed, noise, k, smoothing=False)
                # print(f'Running experiment with seed={seed}, noise={noise}, k={k}, voting=True')
                # run_experiment(seed, noise, k, voting=True)
                print(f'Running experiment with seed={seed}, noise={noise}, k={k}, smoothing=True')
                run_experiment(seed, noise, k, smoothing=True)

if __name__ == "__main__":
    main()