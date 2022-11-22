import numpy as np
from utils import make_homogeneous
from skimage.transform import FundamentalMatrixTransform


def fundamental_fit(X, Y):
    regressor = FundamentalMatrixTransform()
    regressor.estimate(X, Y)
    F = regressor.params
    return F
#     M = X.shape[0]
#     A = np.ones((M, 9))
#     n, d = X.shape
#     centroid = np.mean(X, axis=0)

#     centered = X - centroid
#     rms = np.sqrt(np.sum(centered ** 2) / n)

#     # if all the points are the same, the transformation matrix cannot be
#     # created. We return an equivalent matrix with np.nans as sentinel values.
#     # This obviates the need for try/except blocks in functions calling this
#     # one, and those are only needed when actual 0 is reached, rather than some
#     # small value; ie, we don't need to worry about numerical stability here,
#     # only actual 0.
#     if rms == 0:
#         return np.full((d + 1, d + 1), np.nan), np.full_like(X, np.nan)

#     norm_factor = np.sqrt(d) / rms

#     part_matrix = norm_factor * np.concatenate(
#             (np.eye(d), -centroid[:, np.newaxis]), axis=1
#             )
#     matrix = np.concatenate(
#             (part_matrix, [[0,] * d + [1]]), axis=0
#             )

#     points_h = np.row_stack([X.T, np.ones(n)])

#     new_points_h = (matrix @ points_h).T

#     new_points = new_points_h[:, :d]
#     new_points /= new_points_h[:, d:]

#     A[:, :2] = X
#     A[:, :3] *= X[:, 0, np.newaxis]
#     A[:, 3:5] = X
#     A[:, 3:6] *= Y[:, 1, np.newaxis]
#     A[:, 6:8] = X

#     # Solve for the nullspace of the constraint matrix.
#     _, _, V = np.linalg.svd(A)
#     F_normalized = V[-1, :].reshape(3, 3)

#     # Enforcing the internal constraint that two singular values must be
#     # non-zero and one must be zero.
#     U, S, V = np.linalg.svd(F_normalized)
#     S[2] = 0
#     F = U @ np.diag(S) @ V

#     params = Y.T @ F @ X

#     # _, _, V = np.linalg.svd(A)
#     # F = V[-1, :].reshape(3, 3)

#     # # Enforcing the internal constraint that two singular values must be
#     # # non-zero and one must be zero.
#     # U, S, V = np.linalg.svd(F)
#     # S[0] = S[1] = (S[0] + S[1]) / 2.0
#     # S[2] = 0
#     # params = U @ np.diag(S) @ V
#     return params

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