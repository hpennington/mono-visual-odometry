import io
import numpy as np
import PIL.Image
from IPython import display


def display_mat(M, fmt='jpeg'):
    f = io.BytesIO()
    PIL.Image.fromarray(M).save(f, fmt)
    display.display(display.Image(data=f.getvalue()))