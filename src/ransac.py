import numpy as np
from utils import make_homogeneous
from skimage.transform import FundamentalMatrixTransform


def fundamental_fit(X, Y):
    regressor = FundamentalMatrixTransform()
    regressor.estimate(X, Y)
    F = regressor.params
    return F

def calculate_residuals(F, X, Y):
    X = np.column_stack([X, np.ones(X.shape[0])])
    Y = np.column_stack([Y, np.ones(Y.shape[0])])

    F_src = F @ X.T
    Ft_dst = F.T @ Y.T
    dst_F_src = np.sum(Y * F_src.T, axis=1)

    return np.abs(dst_F_src) / np.sqrt(F_src[0] ** 2 + F_src[1] ** 2 + Ft_dst[0] ** 2 + Ft_dst[1] ** 2)

def ransac(pairs, min_samples, residual_threshold, max_trials):
    i = 0
    max_model = None
    max_inliers = None
    mask_inliers = None

    while i < max_trials:
        
        random_n = np.random.randint(pairs.shape[0], size=min_samples)
        
        x = []
        y = []
        
        for n in random_n:
            x.append(pairs[n][0])
            y.append(pairs[n][1])

        x = np.array(x)
        y = np.array(y)
    
        F = fundamental_fit(x, y)
        residuals = np.abs(calculate_residuals(F, pairs[:, 0], pairs[:, 1]))

        mask = residuals < residual_threshold
        n_inliers = np.count_nonzero(mask)

        if max_inliers is None:
            max_inliers = n_inliers
            max_model = F
            mask_inliers = mask

        if n_inliers > max_inliers:
            max_inliers = n_inliers
            max_model = F
            mask_inliers = mask
            
        i += 1
        
    return max_model, mask_inliers
