import numpy as np


def fundamentalToEssential(F):
    U, _, V = np.linalg.svd(F)
    S = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    return U @ S @ V

# Extract pose from the essential matrix E
def extract_pose(E, K, x1, x2):
    U, _, V = np.linalg.svd(E)
    
    # There are four possible poses from the essential matrix decomposition.
    # to resolve these ambiguities. The determinate of U and V if less than zero needs to have the sign flipped. (6.1.4) page 47.
    if np.linalg.det(U) < 0:
        U = -U

    if np.linalg.det(V) < 0:
        V = -V

    # Solve for rotation R with the formula from Hartley and Zisserman's "Multiple View Geometry in computer vision"
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R = U @ W @ V

    # From Hartley and Zisserman extract the normalized translation vector t
    t = U[:, 2]

    # Cheap hack from geohot to resolve ambiguities
    if np.sum(R.diagonal()) < 0:
        R = U @ W.T @ V

    if t[2] < 0:
        t = -t

    # # We can use the technique from "Geometry, Constraints and Computation of the Trifocal Tensor" by Ressl
    # M = skew_symmetric(R.T @ t)
    # x1 = make_homogeneous(x1)
    # x2 = make_homogeneous(x2)
    # X1 = M @ np.linalg.inv(K) @ x1
    # X2 = M @ R.T @ np.linalg.inv(K) @ x2

    # if X1[2] * X2[2] < 0:
    #     R = U @ W.T @ V
    #     M = skew_symmetric(R.T @ t)
    #     X1 = M @ np.linalg.inv(K) @ x1

    # if X1[2] < 0:
    #     t = -t
    # t = -t

    return R, t

# Triangulate from Hartley and Zisserman
def triangulate(pose1, pose2, pts1, pts2, R, t):
    out = np.zeros((pts1.shape[0], 4))

    r1 = R[0]
    r2 = R[1]
    r3 = R[2]
    
    for i, points in enumerate(zip(pts1, pts2)):
        X, Xp = points
        p3 = pose1[2]
        p3p = pose2[2]
        p1 = pose1[0]
        p2 = pose1[1]
        p1p = pose2[0]
        p2p = pose2[1]
        
        x, y =  X
        xp, yp = Xp
        A = np.array([
            [x * p3 - p1],
            [y * p3 - p2],
            [xp * p3p - p1p],
            [yp * p3p - p2p],
        ]).reshape(4, 4)
        _, _, Vt = np.linalg.svd(A)
        out[i] = Vt[3]

    return out
