import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression, MultiTaskLasso
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

def k_hop_subgraph(neighbor_graph, index, k=1):
    """
    Return the indices and induced subgraph (sparse matrix) of nodes within k hops
    of `index` in the neighbor_graph. The returned nodes include `index`.
    """
    if k < 1:
        nodes = np.array([index], dtype=int)
        return nodes, neighbor_graph[nodes][:, nodes]

    visited = set([int(index)])
    frontier = {int(index)}
    for _ in range(k):
        next_frontier = set()
        for node in frontier:
            nbrs = neighbor_graph[node].nonzero()[1]
            for nb in nbrs:
                nb = int(nb)
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.add(nb)
        frontier = next_frontier
        if not frontier:
            break

    nodes = np.array(sorted(visited), dtype=int)
    subgraph = neighbor_graph[nodes][:, nodes]
    return nodes, subgraph

def lpca(X, d, neighbor_graph, r=None):
    local_pcs = []
    if r is not None:
        K = RBF(r)(X)
    for index in range(X.shape[0]):
        nbr_ind = neighbor_graph[index].nonzero()[1]
        if r is not None:
            K_local = K[index].reshape(-1,1)
            w_mean = np.sum(K_local * X, axis = 0) / K_local.sum()
            Z = np.sqrt(K_local) * (X - w_mean)
            tsbasis = randomized_svd(Z.T @ Z, n_components = d)[2]
            local_pcs.append(tsbasis)
        else:
            pca = PCA(n_components=d).fit(X[nbr_ind])
            local_pcs.append(pca.components_)
    return np.stack(local_pcs)

def connection_laplacian(X, tsbasis, nodes, graph, r = None):
    n = nodes.shape[0]
    d = tsbasis.shape[1]
    S = np.zeros((n * d, n * d))
    D = np.zeros((n * d, n * d))
    for i in range(n):
        nbrs = graph[i].nonzero()[1]
        for j in nbrs:
            C_ij = tsbasis[i] @ tsbasis[j].T
            U, _, Vh = np.linalg.svd(C_ij, full_matrices=False)
            O_ij = U @ Vh
            if r is None:
                w_ij = 1.0
            else:
                w_ij = np.exp(-np.linalg.norm(X[nodes[i]] - X[nodes[j]])**2 / (r**2))
            S[i*d:(i+1)*d, j*d:(j+1)*d] = w_ij * O_ij
            D[i*d:(i+1)*d, i*d:(i+1)*d] += w_ij * np.eye(d)
    # L = D - S
    L = np.eye(n * d) - np.linalg.inv(D) @ S
    return L

def multitask_path(X, y, d, regressor, alpha_0 = 0.0, alpha_1 = 5.0, max_iter=500):
    alpha = (alpha_0 + alpha_1) / 2 
    regressor.set_params(alpha=alpha)
    reg = regressor.fit(X, y)
    support = np.count_nonzero(reg.coef_)
    i = 0
    while support != d * y.shape[1] and i < max_iter:
        if support < d * y.shape[1]:
            alpha_1 = alpha
        else:
            alpha_0 = alpha
        alpha = (alpha_0 + alpha_1) / 2 
        regressor.set_params(alpha=alpha)
        reg = regressor.fit(X, y)
        support = np.count_nonzero(reg.coef_)
        i += 1
    return reg.coef_

def run_ts_regression(X, F, index, neighbor_graph, tsbasis, d=None,
                      regressor = MultiTaskLasso(alpha=1.0, fit_intercept=False),
                      normalizer=StandardScaler(with_mean=False)):
    neighbors = sparse.find(neighbor_graph[index])[1]
    proj = (X[neighbors] - X[index]) @ tsbasis[index].T
    normalized_F = F[neighbors] - F[index]
    normalized_F = normalizer.fit_transform(normalized_F)
    regressor.fit(normalized_F, proj)
    if d is not None and hasattr(regressor, "alpha"):
        coef = multitask_path(normalized_F, proj, d, regressor)
    else:
        coef = regressor.coef_
    return coef, proj

def run_ts_regression_all_neighbors(X, F, index, neighbor_graph, tsbasis,
                                    d=None, k=None, neighbors=None,
                                   regressor = MultiTaskLasso(alpha=1.0, fit_intercept=False),
                                   normalizer=StandardScaler(with_mean=False)):
    if neighbors is None and k is None:
        neighbors = sparse.find(neighbor_graph[index])[1]
    elif neighbors is None and k is not None:
        neighbors, _ = k_hop_subgraph(neighbor_graph, index, k=k)
    else:
        neighbors = np.array(neighbors, dtype=int)
    
    coefs = {}
    projs = {}
    supp_counts = np.zeros(F.shape[1])
    for neighbor in neighbors:
        # compute regression using the neighbor as the local center
        nbr_coef, nbr_proj = run_ts_regression(X, F, int(neighbor), neighbor_graph, tsbasis,
                                               d=d, regressor=regressor, normalizer=normalizer)
        coefs[int(neighbor)] = nbr_coef
        projs[int(neighbor)] = nbr_proj
        supp_counts += np.count_nonzero(nbr_coef, axis=0)

    return coefs, projs

def run_ts_regression_voting(X, F, index, neighbor_graph, tsbasis, d=None,
                             k=None, neighbors=None, min_count=0.25,
                             regressor = MultiTaskLasso(alpha=1.0, fit_intercept=False),
                             normalizer=StandardScaler(with_mean=False)):
    
    coefs, projs = run_ts_regression_all_neighbors(X, F, index, neighbor_graph, tsbasis,
                                                   d=d, k=k, neighbors=neighbors,
                                                   regressor=regressor, normalizer=normalizer)
    supp_counts = np.zeros(F.shape[1])
    for nbr, coef in coefs.items():
        supp_counts += np.count_nonzero(coef, axis=0)
        
    if d is not None:
        top_by_count = np.argsort(supp_counts)[-d:][::-1]
    else:
        if min_count < 1.0:
            min_count = int(min_count * len(coefs))
        top_by_count = np.unique(np.where(supp_counts >= min_count)[0])
    F_filtered = F[:,top_by_count]
    filtered_coef, proj = run_ts_regression(X, F_filtered, index, neighbor_graph, tsbasis, d=d,
                                            regressor=regressor, normalizer=normalizer)
    final_coef = np.zeros((filtered_coef.shape[0], F.shape[1]))
    final_coef[:, top_by_count] = filtered_coef
    return final_coef, proj

def run_ts_regression_smoothed(X, F, index, neighbor_graph, tsbasis,
                               d=None, k=None, neighbors=None, gamma=1.0, r=None,
                               regressor = MultiTaskLasso(alpha=1.0, fit_intercept=False),
                               normalizer=StandardScaler(with_mean=False)):
    
    coefs, projs = run_ts_regression_all_neighbors(X, F, index, neighbor_graph, tsbasis,
                                                   d=d, k=k, neighbors=neighbors,
                                                   regressor=regressor, normalizer=normalizer)

    local_nodes, local_graph = k_hop_subgraph(neighbor_graph, index, k=k)
    if r is None:
        r = np.median(np.linalg.norm(X[local_nodes] - X[index], axis=1))
    L = connection_laplacian(X, tsbasis, local_nodes, local_graph, r=r)
    denoiser = np.linalg.inv(np.eye(L.shape[0]) + gamma * L)
    denoised_coefs = denoiser @ np.vstack([coefs[int(nbr)] for nbr in local_nodes])
    denoised_coefs = denoised_coefs.reshape((local_nodes.shape[0], tsbasis.shape[1], -1))
    final_coef = denoised_coefs[local_nodes == index][0]

    if d is not None:
        strengths = np.linalg.norm(final_coef, axis=0)
        zero_indices = np.argsort(strengths)[:-d]
        final_coef[:, zero_indices] = 0.0
    return final_coef, projs[index]

def run_ts_regression_neighborhood(X, F, index, neighbor_graph, tsbasis,
                                  k=1, d=None, alpha = 1.0,
                                  normalizer=StandardScaler(with_mean=False)):

    if d is None:
        d = tsbasis.shape[2]
    k_hop_neighbors, _ = k_hop_subgraph(neighbor_graph, index, k=k)

    nbr_proj = {}
    nbr_F = {}
    for nbr_idx in k_hop_neighbors:
        neighbors = sparse.find(neighbor_graph[nbr_idx])[1]
        proj = (X[neighbors] - X[nbr_idx]) @ tsbasis[nbr_idx].T
        normalized_F = F[neighbors] - F[nbr_idx]
        normalized_F = normalizer.fit_transform(normalized_F)
        nbr_proj[nbr_idx] = proj
        nbr_F[nbr_idx] = normalized_F

    # Precompute quantities for a vectorized objective
    K = len(k_hop_neighbors)
    p = F.shape[1]
    A = np.zeros((K, p, p), dtype=float)   # F_i^T F_i
    B = np.zeros((K, p, d), dtype=float)   # F_i^T P_i
    proj_norm_sq = np.zeros(K, dtype=float)  # ||P_i||_F^2
    # map neighbor index to position in stacked arrays
    for i, nbr_idx in enumerate(k_hop_neighbors):
        Fi = nbr_F[nbr_idx]
        Pi = nbr_proj[nbr_idx]
        A[i] = Fi.T @ Fi
        B[i] = Fi.T @ Pi
        proj_norm_sq[i] = np.sum(Pi * Pi)

    lam = alpha / np.sqrt(d * K)

    def _objective(W):
        W = W.reshape(K, p, d)
        # constant term: 0.5 * sum ||P_i||_F^2
        const = 0.5 * proj_norm_sq.sum()
        # linear term: - sum trace(W_i^T B_i) = - sum(W * B)
        lin = np.einsum('kpd,kpd->', W, B)
        # quadratic term: 0.5 * sum trace(W_i^T A_i W_i)
        AW = np.einsum('kpq,kqd->kpd', A, W)
        quad = 0.5 * np.einsum('kpd,kpd->', W, AW)
        obj = const - lin + quad
        # group-lasso penalty over feature groups: sum_j ||W[:, j, :]||_F
        group_norms = np.linalg.norm(W, axis=(0, 2))
        obj += lam * group_norms.sum()
        return float(obj)
    
    local_coef, _ = run_ts_regression(X, F, index, neighbor_graph, tsbasis, d=d, regressor=LinearRegression(fit_intercept=False), normalizer=normalizer)
    W0 = np.repeat(local_coef[np.newaxis, :, :], K, axis=0).flatten()
    result = minimize(_objective, W0, method='BFGS', options={'maxiter' : 1000})
    return result