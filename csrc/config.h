#ifndef CONFIG_H
#define CONFIG_H
#include <array>
#include <string>

// Script parameters
const bool KITTI = false;
const std::string KITTI_DATA_DIRECTORY = "/Users/haydenpennington/dev/data/kitti/";
const int im_w = 1920;
const int im_h = 1080;
const int im_size[2] = {im_w, im_h};

// Camera calibration. Intrinsics are specified at the calibration image size
// and are scaled to im_w x im_h before VO runs.
const int camera_calibration_w = im_w;
const int camera_calibration_h = im_h;
const double camera_fx = im_w;
const double camera_fy = im_w;
const double camera_cx = im_w / 2.0;
const double camera_cy = im_h / 2.0;
const bool use_distortion_correction = true;
const std::array<double, 5> camera_distortion_coeffs = {
    0.0, // k1
    0.0, // k2
    0.0, // p1
    0.0, // p2
    0.0  // k3
};

// ORB Detector parameters
const int max_corners = 10000;
const int kernel_size = 12;
const float min_distance = 3;
const float quality = 0.01;
const int keypoint_size = 27;

// Ransac parameters
const int ransac_minsamples = 8;
const int ransac_max_trials = 250;
const float ransac_residual_threshold = 0.05;

// Pose extratction translation scaling
const float tscale = 1.0;
const double pose_scale = 1.0;

// Keyframe selection — minimum median feature displacement (pixels) before triangulating
const float kf_min_displacement = 2.0f;
const double max_triangulated_depth = 100.0;
const double max_reprojection_error_px = 3.0;

// Local bundle adjustment
const bool use_bundle_adjustment = true;
const int ba_min_points = 20;
const int ba_max_iterations = 20;

// Semi-dense mapping. Sparse ORB still estimates pose; LK tracks add map density.
const bool use_dense_mapping = true;
const int dense_max_points = 20000;
const double dense_quality = 0.001;
const double dense_min_distance = 4.0;
const double dense_lk_max_error = 20.0;
const double dense_min_displacement = 1.0;

// Point cloud clustering
const int n_points = 2;
const float dbscan_eps = tscale * 0.5;

#endif
