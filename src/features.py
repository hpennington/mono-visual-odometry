import cv2
import numpy as np

## Create the ORB feature detector
orb = cv2.ORB_create()

## Create the feature matcher
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

def compute_orb_descriptors(im, keypoints):
    return orb.compute(im, keypoints)

def bruteforce_match(descriptors1, descriptors2):
    matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)
    matches = np.asarray(matches)
    return matches

def extract_features(im, max_corners, quality, kernel_size, keypoint_size):
    corners = cv2.goodFeaturesToTrack(
        im,
        max_corners,
        quality,
        kernel_size
    )

    key_points = [cv2.KeyPoint(x=x[0][0], y=x[0][1], size=keypoint_size) for x in corners]
    return key_points