import numpy as np
from skimage.transform import FundamentalMatrixTransform


def ransac(pairs, min_samples, residual_threshold, max_trials):
    i = 0
    max_model = None
    max_inliers = None
    mask_inliers = None

    while i < max_trials:
        regressor = FundamentalMatrixTransform()
        random_n = np.random.randint(pairs.shape[0], size=min_samples)
        
        x = []
        y = []
        
        for n in random_n:
            x.append(pairs[n][0])
            y.append(pairs[n][1])

        x = np.array(x)
        y = np.array(y)
        
        regressor.estimate(x, y)
        residuals = np.abs(regressor.residuals(pairs[:, 0], pairs[:, 1]))
        mask = residuals < residual_threshold
        n_inliers = np.count_nonzero(mask)

        if max_inliers is None:
            max_inliers = n_inliers
            max_model = regressor
            mask_inliers = mask

        if n_inliers > max_inliers:
            max_inliers = n_inliers
            max_model = regressor
            mask_inliers = mask
            
        i += 1
        
    return max_model, mask_inliers