import io
import numpy as np
import PIL.Image
from IPython import display


def normalize(Kinv, pts):
    pts = np.concatenate([pts, np.array([1.0])])
    return (Kinv @ pts).T[:2]

def display_mat(M, fmt='jpeg'):
    f = io.BytesIO()
    PIL.Image.fromarray(M).save(f, fmt)
    display.display(display.Image(data=f.getvalue()))

def make_homogeneous(x):
    return np.column_stack([x, np.ones(x.shape[0])])

def skew_symmetric(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
