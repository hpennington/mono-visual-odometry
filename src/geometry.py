import numpy as np


def normalize(T, x):
    x = make_homogeneous(x)
    result = np.dot(T, x.T).T[:, :2]
    return result

def make_homogeneous(x):
    return np.column_stack([x, np.ones(x.shape[0])])

def skew_symmetric(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

def create_normalization_matrix(H, W):
    T = np.eye(3)
    sx = 1 / (W // 2)
    sy = 1 / (H // 2)
    tx = sx * (W // 2)
    ty = sy * (H // 2)
    T[0, 2] = -tx 
    T[1, 2] = -ty
    T[0, 0] = sx
    T[1, 1] = sy
    return T

def fundamental_to_essential(F):
    U, _, V = np.linalg.svd(F)
    S = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    return U @ S @ V

# Extract pose from the essential matrix E
def extract_pose(E, x1, x2):
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

    return R, t

def pack_Rt(R, t):
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t.T
    return pose

def integrate_pose(pose_abs, R, t):
    pose = pack_Rt(R, t)
    return np.dot(pose_abs, pose)

def calculate_projection(R, t, last_proj):
    pose = pack_Rt(R, t)
    RT = np.linalg.inv(pose)
    return np.dot(RT, last_proj)

# Triangulate from Hartley and Zisserman
def triangulate(colors, pose1, pose2, pts1, pts2, R, t):
    out = np.zeros((pts1.shape[0], 4))
    out_colors = np.zeros((pts1.shape[0], 3))
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
        out_colors[i] = colors[i]

    return out, out_colors

class FundamentalMatrixModel:
    def __init__(self):
        self.params = None

    def fit(self, X, Y):
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
        self.params = F

    def calculate_residuals(self, x, y):
        X = make_homogeneous(x)
        Y = make_homogeneous(y)

        F = self.params
        Fx = F @ X.T
        Fty = F.T @ Y.T

        numerator = np.abs(np.sqrt(np.sum(Y.T * Fx, axis=0)**2))
        denominater = np.sqrt(Fx[0]**2 + Fx[1]**2 + Fty[0]**2 + Fty[1]**2)
        result = (numerator / denominater).T
        
        return result
    