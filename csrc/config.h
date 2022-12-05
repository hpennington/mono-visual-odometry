#ifndef CONFIG_H
#define CONFIG_H
#include <string>

// Script parameters
const bool KITTI = true;
const std::string KITTI_DATA_DIRECTORY = "/Users/haydenpennington/dev/data/kitti/";
const std::string DATA_INPUT = "../../videos/test_countryroad.mp4";
const int im_w = 640;
const int im_h = 480;
const int im_size[2] = {im_w, im_h};

// ORB Detector parameters
const int max_corners = 1500;
const int kernel_size = 6;
const float min_distance = 9;
const float quality = 0.001;
const int keypoint_size = 6;

// Ransac parameters
const int ransac_minsamples = 8;
const int ransac_max_trials = 150;
const float ransac_residual_threshold = 0.001;

// Pose extratction translation scaling
const float tscale = 1.0;

// Point cloud clustering
const int n_points = 2;
const float dbscan_eps = tscale * 0.5;

#endif
