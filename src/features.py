import cv2


def extract_features(im, max_corners, quality, kernel_size, keypoint_size):
    corners = cv2.goodFeaturesToTrack(
        im,
        max_corners,
        quality,
        kernel_size
    )

    key_points = [cv2.KeyPoint(x=x[0][0], y=x[0][1], size=keypoint_size) for x in corners]
    return key_points