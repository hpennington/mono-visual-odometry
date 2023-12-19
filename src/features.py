import cv2
import numpy as np


## Create the ORB feature detector
orb = cv2.ORB_create()

## Create the feature matcher
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

def bruteforce_match(descriptors1, descriptors2):
    matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)
    matches = np.asarray(matches)
    return matches

def extract_features(im, max_corners, quality, kernel_size, min_distance):
    corners = cv2.goodFeaturesToTrack(
        im,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_distance,
        blockSize=kernel_size
    )

    return corners