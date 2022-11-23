import numpy as np
from utils import make_homogeneous
from skimage.transform import FundamentalMatrixTransform


class FundamentalMatrixModel:
    def __init__(self):
        regressor = FundamentalMatrixTransform()
        self.regressor = regressor
        self.params = None

    def fit(self, X, Y):
        self.regressor.estimate(X, Y)
        self.params = self.regressor.params

    def calculate_residuals(self, X, Y):
        return self.regressor.residuals(X, Y)


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

        regressor = FundamentalMatrixModel()
        regressor.fit(x, y)
        F = regressor.params
        residuals = np.abs(regressor.calculate_residuals(pairs[:, 0], pairs[:, 1]))

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
