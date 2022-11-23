import numpy as np
from utils import make_homogeneous, center_and_normalize_points


class FundamentalMatrixModel:
    def __init__(self):
        self.params = None

    def fit(self, X, Y):
        X_matrix, X = center_and_normalize_points(X)
        Y_matrix, Y = center_and_normalize_points(Y)
        m = X.shape[0]
        A = np.zeros((m, 9))

        for i in range(m):
            x = X[i, 0]
            xp = Y[i, 0]
            y = X[i, 1]
            yp = Y[i, 1]
            A[i] = [xp*x, xp*y, xp, yp*x, yp*y, yp, x, y, 1]

        U, S, Vt = np.linalg.svd(A)
        F = Vt[-1, :].reshape(3, 3)

        U, S, Vt = np.linalg.svd(F)
        S[2] = 0
        F = U @ np.diag(S) @ Vt    
        self.params = Y_matrix.T @ F @ X_matrix

    def calculate_residuals(self, X, Y):
        # Compute the Sampson distance.
        # src_homogeneous = np.column_stack([X, np.ones(X.shape[0])])
        src_homogeneous = make_homogeneous(X)
        dst_homogeneous = make_homogeneous(Y)

        F_src = self.params @ src_homogeneous.T
        Ft_dst = self.params.T @ dst_homogeneous.T

        dst_F_src = np.sum(dst_homogeneous * F_src.T, axis=1)

        return np.abs(dst_F_src) / np.sqrt(F_src[0] ** 2 + F_src[1] ** 2
                                        + Ft_dst[0] ** 2 + Ft_dst[1] ** 2)


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
