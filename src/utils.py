import io
import cv2
import numpy as np
import PIL.Image
from IPython import display


def display_mat(M, fmt='jpeg'):
    f = io.BytesIO()
    PIL.Image.fromarray(M).save(f, fmt)
    display.display(display.Image(data=f.getvalue()))

def draw_points(im, feature_pairs, multiplier_x, multiplier_y):
    for pt1, pt2 in feature_pairs:
        u1,v1 = int(round(pt1[0]) * multiplier_x), int(round(pt1[1]) * multiplier_y)
        u2,v2 = int(round(pt2[0]) * multiplier_x), int(round(pt2[1]) * multiplier_y)
        cv2.circle(im, (u1, v1), color=(0, 255, 0), radius=3)
        cv2.line(im, (u1, v1), (u2, v2), color=(255, 0, 0))
        cv2.circle(im, (u2, v2), color=(0, 0, 255), radius=3)