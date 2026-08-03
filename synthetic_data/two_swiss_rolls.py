import numpy as np
from scipy.spatial.transform import Rotation as R

def noisy_swiss_roll(n_samples=1000, noise=0.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    t = 2 * np.pi * (1 + 2 * np.random.rand(n_samples))
    y = 10 * np.random.rand(n_samples)

    x = t * np.cos(t)
    z = t * np.sin(t)
    X = np.vstack((x, y, z)).T

    if noise > 0.0:
        temp = X.copy()
        temp[:, 1] = 0
        temp = temp / np.linalg.norm(temp, axis=1)[:, None]
        X = X + noise * np.random.uniform(-1,1,(X.shape[0],1)) * temp

    return X, t, y

def two_swiss_rolls(n_samples=1000, noise=0.0, offset = 20.0, outliers = 0, seed=None):
    if seed is not None:
        np.random.seed(seed)
   
    X1, t1, s1 = noisy_swiss_roll(n_samples=n_samples, noise=noise)
    X2, t2, s2 = noisy_swiss_roll(n_samples=n_samples, noise=noise)

    r1 = R.random().as_matrix()
    r2 = R.random().as_matrix()
    X1 = X1 @ r1
    X2 = X2 @ r2
    X2[:, 0] += offset

    X = np.concatenate([X1, X2], axis=0)
    s1 = np.concatenate([s1, np.random.rand(n_samples)*s1.max()])
    s2 = np.concatenate([np.random.rand(n_samples)*s2.max(), s2])
    t1 = np.concatenate([t1, np.random.rand(n_samples)*t1.max()])
    t2 = np.concatenate([np.random.rand(n_samples)*t2.max(), t2])
    F = np.stack([s1, s2, t1, t2], axis=1)

    if outliers > 0:
        outlier_X = np.random.uniform(low=[X[:,0].min(), X[:,1].min(), X[:,2].min()], high=[X[:,0].max(), X[:,1].max(), X[:,2].max()], size=(outliers, X.shape[1]))
        outlier_F = np.random.uniform(low=[F[:,0].min(), F[:,1].min(), F[:,2].min(), F[:,3].min()], high=[F[:,0].max(), F[:,1].max(), F[:,2].max(), F[:,3].max()], size=(outliers, F.shape[1]))
        X = np.concatenate([X, outlier_X], axis=0)
        F = np.concatenate([F, outlier_F], axis=0)
    
    return X, F