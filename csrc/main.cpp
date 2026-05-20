#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/matx.hpp>
#include <open3d/Open3D.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <atomic>
#include <numeric>
#include <random>

#include "./config.h"

static std::atomic<bool> g_shutdown{false};
static void handle_sigint(int) { g_shutdown = true; }

auto orb = cv::ORB::create();
auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);

struct FeaturesResult
{
    cv::Mat corners;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

struct RansacResult
{
    Eigen::Matrix<bool, Eigen::Dynamic, 1> mask;
    Eigen::MatrixXf model;
};

struct MatchesResult
{
   std::vector<std::vector<std::vector<float>>> pairs;
   std::vector<std::vector<std::vector<float>>> norm_pairs;
   Eigen::MatrixXf F = Eigen::MatrixXf::Zero(3, 3);
   bool has_F = false;
};

Eigen::MatrixXf make_homogeneous(Eigen::MatrixXf in)
{
    Eigen::MatrixXf out = Eigen::MatrixXf::Ones(in.rows(), in.cols() + 1);
    for (int i = 0; i < in.rows(); i += 1)
    {
        out(i, 0) = in(i, 0);
        out(i, 1) = in(i, 1);
        out(i, 2) = 1.0;
    }
    return out;
}

Eigen::MatrixXf create_normalization_matrix(int h, int w)
{
    Eigen::MatrixXf T = Eigen::MatrixXf::Identity(3, 3);
    float sx = 1.0 / (float)(w / 2);
    float sy = 1.0 / (float)(h / 2);
    float tx = sx * (float)(w / 2);
    float ty = sy * (float)(h / 2);
    T(0, 2) = -tx;
    T(1, 2) = -ty;
    T(0, 0) = sx;
    T(1, 1) = sy;
    return T;
}

// Enforce essential matrix constraint via SVD (singular values -> [1,1,0])
Eigen::MatrixXf fundamental_to_essential(Eigen::MatrixXf F)
{
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXf S = Eigen::MatrixXf::Zero(3, 3);
    S(0, 0) = 1.0f;
    S(1, 1) = 1.0f;
    return svd.matrixU() * S * svd.matrixV().transpose();
}

// Port of geometry.py extract_pose — geohot diagonal/t[2] hack to resolve the 4-way E ambiguity
std::pair<Eigen::MatrixXf, Eigen::VectorXf> extract_pose(Eigen::MatrixXf E)
{
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXf U = svd.matrixU();
    Eigen::MatrixXf V = svd.matrixV();

    if (U.determinant() < 0) U = -U;
    if (V.determinant() < 0) V = -V;

    Eigen::MatrixXf W = Eigen::MatrixXf::Zero(3, 3);
    W(0, 1) = -1; W(1, 0) = 1; W(2, 2) = 1;

    Eigen::MatrixXf R = U * W * V.transpose();
    Eigen::VectorXf t = U.col(2);

    if (R.diagonal().sum() < 0)
        R = U * W.transpose() * V.transpose();

    if (t(2) < 0)
        t = -t;

    return {R, t};
}

Eigen::Matrix4f pack_Rt(Eigen::Matrix3f R, Eigen::Vector3f t)
{
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    pose.block<3, 3>(0, 0) = R;
    pose.block<3, 1>(0, 3) = t;
    return pose;
}

Eigen::MatrixXf normalize(Eigen::MatrixXf T, Eigen::MatrixXf x)
{
    auto X = make_homogeneous(x.transpose());
    auto result = (T * X.transpose()).transpose()(Eigen::indexing::all, Eigen::seq(0, 2));
    return result;
}

class FundamentalMatrixTransform
{
public:
    Eigen::Matrix<float, 3, 3> params;

    void fit(Eigen::Matrix<float, Eigen::Dynamic, 2> X, Eigen::Matrix<float, Eigen::Dynamic, 2> Y)
    {
        int m = X.rows();
        Eigen::MatrixXf A = Eigen::MatrixXf::Zero(m, 9);

        for (int i = 0; i < m; i += 1)
        {
            float x = X(i, 0);
            float y = X(i, 1);
            float xp = Y(i, 0);
            float yp = Y(i, 1);

            A(i, 0) = xp*x;
            A(i, 1) = xp*y;
            A(i, 2) = xp;
            A(i, 3) = yp*x;
            A(i, 4) = yp*y;
            A(i, 5) = yp;
            A(i, 6) = x;
            A(i, 7) = y;
            A(i, 8) = 1;
        }

        Eigen::JacobiSVD<Eigen::MatrixXf> svd;
        svd.compute(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXf f = svd.matrixV().col(svd.matrixV().cols() - 1);
        Eigen::MatrixXf F(3, 3);
        F << f(0), f(1), f(2),
             f(3), f(4), f(5),
             f(6), f(7), f(8);
        
        Eigen::JacobiSVD<Eigen::MatrixXf> svd2;
        svd2.compute(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXf S = svd2.singularValues();
        S(2) = 0;

        Eigen::MatrixXf d = Eigen::MatrixXf::Identity(3, 3);
        d(0, 0) = S(0);
        d(1, 1) = S(1);
        d(2, 2) = S(2);
        Eigen::MatrixXf F_prime = svd2.matrixU() * d * svd2.matrixV().transpose();
        this->params = F_prime;
    }
        
    Eigen::MatrixXf calculate_residuals(Eigen::Matrix<float, Eigen::Dynamic, 2> X, Eigen::Matrix<float, Eigen::Dynamic, 2> Y) 
    {
        Eigen::MatrixXf X_prime = make_homogeneous(X);
        Eigen::MatrixXf Y_prime = make_homogeneous(Y);
        Eigen::MatrixXf F = this->params;
        Eigen::MatrixXf Fx = F * X_prime.transpose();
        Eigen::MatrixXf Fty = F.transpose() * Y_prime.transpose();
    
        auto numerator = (Y_prime.transpose().cwiseProduct(Fx)).colwise().sum().array().abs();
        auto denominator = (Fx.row(0).array().square() + Fx.row(1).array().square() + Fty.row(0).array().square() + Fty.row(1).array().square()).sqrt();

        return (numerator / denominator).transpose();
    }
};

RansacResult ransac(FundamentalMatrixTransform model, Eigen::Matrix<float, Eigen::Dynamic, 2> kps1, Eigen::Matrix<float, Eigen::Dynamic, 2> kps2)
{
    RansacResult result;
    int max_inliers = -1;
    std::vector<int> indices(kps1.rows());
    std::iota(indices.begin(), indices.end(), 0);
    static std::mt19937 rng(std::random_device{}());

    for (int i = 0; i < ransac_max_trials; i += 1)
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        Eigen::Matrix<float, Eigen::Dynamic, 2> kps1_sub(ransac_minsamples, 2);
        Eigen::Matrix<float, Eigen::Dynamic, 2> kps2_sub(ransac_minsamples, 2);
        for (int j = 0; j < ransac_minsamples; ++j) {
            kps1_sub.row(j) = kps1.row(indices[j]);
            kps2_sub.row(j) = kps2.row(indices[j]);
        }
        model.fit(kps1_sub, kps2_sub);
        Eigen::MatrixXf residuals = model.calculate_residuals(kps1, kps2);
        Eigen::Matrix<bool, Eigen::Dynamic, 1> mask = residuals.array() <= ransac_residual_threshold;
        int n_inliers = mask.count();

        if (n_inliers > max_inliers)
        {
            max_inliers = n_inliers;
            result.mask = mask;
            result.model = model.params;
        }
    }

    return result;
}

cv::Mat transform_image(cv::Mat in, int n_rows, int n_columns)
{
    cv::Mat out = cv::Mat(n_rows, n_columns, CV_8UC1);
    cv::resize(in, out, cv::Size(n_columns, n_rows), cv::INTER_LINEAR);
    cv::cvtColor(out, out, cv::COLOR_BGR2GRAY, 1);
    return out;
}

Eigen::Vector3d sample_rgb_color(const cv::Mat& frame, const cv::Point2f& point)
{
    int x = std::clamp((int)std::lround(point.x), 0, frame.cols - 1);
    int y = std::clamp((int)std::lround(point.y), 0, frame.rows - 1);
    cv::Vec3b bgr = frame.at<cv::Vec3b>(y, x);
    return Eigen::Vector3d(bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0);
}

int minimum(int a, int b)
{
    return a > b ? b : a;
}

FeaturesResult extract_features(cv::Mat frame, const int max_corners, double quality, double min_distance)
{
    FeaturesResult result;
    cv::Mat corners;
    cv::goodFeaturesToTrack(frame, corners, max_corners, quality, min_distance);
    result.corners = corners;

    if (corners.empty()) {
        return result;
    }

    std::vector<cv::KeyPoint> keypoints;
    // Convert corners to keypoints
    for (int i = 0; i < minimum(max_corners, corners.size().height); i += 1) {
        auto keypoint = cv::KeyPoint();
        keypoint.pt.x = corners.at<float>(i, 0);
        keypoint.pt.y = corners.at<float>(i, 1);
        keypoint.size = keypoint_size;
        keypoints.push_back(keypoint);
    }

    cv::Mat descriptors;
    orb->compute(frame, keypoints, descriptors);

    result.keypoints = keypoints;
    result.descriptors = descriptors;

    return result;
}

MatchesResult match_frames(cv::Mat corners1, cv::Mat corners2, std::vector<cv::KeyPoint> kps1, std::vector<cv::KeyPoint> kps2, cv::Mat descriptors1, cv::Mat descriptors2, Eigen::MatrixXf T)
{
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<std::vector<std::vector<float>>> pairs;

    if (descriptors1.empty() || descriptors2.empty()) {
        return MatchesResult{};
    }
    
    matcher->knnMatch(descriptors1, descriptors2, matches, 2);

    std::vector<std::vector<std::vector<float>>> lowes_matches;
    
    for (int j = 0; j < matches.size(); j += 1) 
    {   
        if (matches[j].size() < 2) {
            continue;
        }

        auto m = matches[j][0];
        auto n = matches[j][1];
        
        if (m.distance < n.distance * 0.75) {

            auto pt1 = kps1[m.queryIdx].pt;
            auto pt2 = kps2[m.trainIdx].pt;
            float pt1x = pt1.x;
            float pt1y = pt1.y;
            float pt2x = pt2.x;
            float pt2y = pt2.y;
            
            std::vector<std::vector<float>> pair = {{pt1x, pt1y}, {pt2x, pt2y}};
            Eigen::MatrixXf pt1_matrix = Eigen::MatrixXf::Zero(2, 1);
            Eigen::MatrixXf pt2_matrix = Eigen::MatrixXf::Zero(2, 1);
            pt1_matrix(0) = pair[0][0];
            pt1_matrix(1) = pair[0][1];
            pt2_matrix(0) = pair[1][0];
            pt2_matrix(1) = pair[1][1];
            auto norm_pt1_matrix = normalize(T, pt1_matrix);
            auto norm_pt2_matrix = normalize(T, pt2_matrix);
            std::vector<float> norm_pt1 = {norm_pt1_matrix(0), norm_pt1_matrix(1)};
            std::vector<float> norm_pt2 = {norm_pt2_matrix(0), norm_pt2_matrix(1)};
            std::vector<std::vector<float>> norm_pair = {norm_pt1, norm_pt2};
            pairs.push_back(pair);
            lowes_matches.push_back(norm_pair);
        }
    }
    
    Eigen::MatrixXf left_pt = Eigen::MatrixXf(lowes_matches.size(), 2);
    Eigen::MatrixXf right_pt = Eigen::MatrixXf(lowes_matches.size(), 2);

    for (int j = 0; j < lowes_matches.size(); j += 1)
    {
        float pt1x = lowes_matches[j][0][0];
        float pt1y = lowes_matches[j][0][1];
        float pt2x = lowes_matches[j][1][0];
        float pt2y = lowes_matches[j][1][1];
        
        left_pt(j, 0) = pt1x;
        left_pt(j, 1) = pt1y;

        right_pt(j, 0) = pt2x;
        right_pt(j, 1) = pt2y;
    }

    if (left_pt.rows() > ransac_minsamples && right_pt.rows() > ransac_minsamples)
    {
        auto model = FundamentalMatrixTransform();
        auto result = ransac(model, left_pt, right_pt);
        auto mask = result.mask;
        auto F = result.model;

        std::vector<std::vector<std::vector<float>>> sub;
        std::vector<std::vector<std::vector<float>>> sub_norm;
        for (size_t i = 0; i < mask.size(); ++i){
            if (mask[i]) sub.push_back(pairs[i]);
            if (mask[i]) sub_norm.push_back(lowes_matches[i]);
        }

        MatchesResult res;
        res.pairs = sub;
        res.norm_pairs = sub_norm;
        res.F = T.transpose() * F * T;
        res.has_F = true;

        return res;

    }

    MatchesResult result2;
    result2.pairs = pairs;
    result2.norm_pairs = lowes_matches;

    return result2;
}

void draw_points(cv::Mat frame, std::vector<std::vector<std::vector<float>>> pairs, float mul_x, float mul_y) 
{   
    for (int i = 0; i < pairs.size(); i += 1)
    {
        float u1 = (mul_x * pairs[i][0][0]);
        float v1 = (mul_y * pairs[i][0][1]);
        float u2 = (mul_x * pairs[i][1][0]);
        float v2 = (mul_y * pairs[i][1][1]);

        cv::Point center = cv::Point(u1, v1);
        cv::Scalar line_color(0, 255, 0);
        cv::circle(frame, center, 3.0, line_color, 1.0);

        cv::Point center2 = cv::Point(u2, v2);
        cv::Scalar line_color2(0, 0, 255);
        cv::circle(frame, center2, 3.0, line_color2, 1.0);

        // Draw line connecting points
        cv::Scalar line_color3(255, 0, 0);
        cv::line(frame, center, center2, line_color3);
    }
}

float median_displacement(const std::vector<std::vector<std::vector<float>>>& pairs)
{
    std::vector<float> dists;
    dists.reserve(pairs.size());
    for (const auto& pair : pairs) {
        float dx = pair[1][0] - pair[0][0];
        float dy = pair[1][1] - pair[0][1];
        dists.push_back(std::sqrt(dx*dx + dy*dy));
    }
    if (dists.empty()) return 0.0f;
    std::sort(dists.begin(), dists.end());
    return dists[dists.size() / 2];
}

int main(int argc, char *argv[])
{
    std::signal(SIGINT, handle_sigint);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video path or camera index>" << std::endl;
        return 1;
    }

    auto DATA_INPUT = argv[1];
    auto cap = cv::VideoCapture(DATA_INPUT);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap.set(cv::CAP_PROP_FPS, 20); // set fps before set fourcc
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cv::Mat cv2_original;
    std::vector<cv::KeyPoint> last_keypoints;
    cv::Mat last_descriptors;
    cv::Mat last_corners;
    auto T = create_normalization_matrix(im_h, im_w);

    cv::Mat K = (cv::Mat_<double>(3, 3) << camera_fx,       0.0, camera_cx,
                                                   0.0, camera_fy, camera_cy,
                                                   0.0,       0.0,       1.0);

    auto point_cloud = std::make_shared<open3d::geometry::PointCloud>();

    Eigen::Matrix4f pose_abs = Eigen::Matrix4f::Identity();

    while (cap.isOpened() && !g_shutdown) {

        auto t0 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        bool frameGrabbed = cap.read(cv2_original);

        if (frameGrabbed == false) {
            break;
        }

        // Convert to B&W then resize; also keep a resized color frame for sampling
        cv::Mat cv2_frame = transform_image(cv2_original, im_h, im_w);
        cv::Mat cv2_color_frame;
        cv::resize(cv2_original, cv2_color_frame, cv::Size(im_w, im_h), cv::INTER_LINEAR);
        float mul_x = (float)cv2_original.cols / (float)cv2_frame.cols;
        float mul_y = (float)cv2_original.rows / (float)cv2_frame.rows;

        FeaturesResult results = extract_features(cv2_frame, max_corners, quality, min_distance);
        cv::Mat corners = results.corners;
        std::vector<cv::KeyPoint> keypoints = results.keypoints;

        cv::Mat descriptors = results.descriptors;

        if (last_descriptors.rows > 0 && last_keypoints.size() > 0 && last_corners.rows > 0)
        {
            MatchesResult result = match_frames(corners, last_corners, keypoints, last_keypoints, descriptors, last_descriptors, T);
            auto pairs = result.pairs;

            draw_points(cv2_original, pairs, mul_x, mul_y);
            cv::imshow("Frame", cv2_original);

            if (result.has_F && pairs.size() >= 5) {
                std::vector<cv::Point2f> pts_prev, pts_curr;
                for (const auto& pair : pairs) {
                    pts_curr.push_back(cv::Point2f(pair[0][0], pair[0][1]));
                    pts_prev.push_back(cv::Point2f(pair[1][0], pair[1][1]));
                }

                cv::Mat pose_mask;
                cv::Mat E_cv = cv::findEssentialMat(pts_prev, pts_curr, K, cv::RANSAC, 0.999, 1.0, pose_mask);
                if (E_cv.empty() || E_cv.rows != 3 || E_cv.cols != 3) {
                    continue;
                }

                cv::Mat R_cv, t_cv;
                int pose_inliers = cv::recoverPose(E_cv, pts_prev, pts_curr, K, R_cv, t_cv, pose_mask);
                if (pose_inliers < ransac_minsamples) {
                    continue;
                }

                Eigen::Matrix3f R_rel;
                Eigen::Vector3f t_rel;
                cv::cv2eigen(R_cv, R_rel);
                cv::cv2eigen(t_cv, t_rel);

                Eigen::Matrix4f pose_prev = pose_abs;
                pose_abs = pack_Rt(R_rel, t_rel * (float)pose_scale) * pose_abs;

                if (median_displacement(pairs) < kf_min_displacement) {
                    continue;
                }

                cv::Mat Rt_prev = cv::Mat::eye(3, 4, CV_64F);
                cv::Mat Rt_curr;
                cv::hconcat(R_cv, t_cv * pose_scale, Rt_curr);
                cv::Mat P1 = K * Rt_prev;
                cv::Mat P2 = K * Rt_curr;

                cv::Mat pts1_mat, pts2_mat;
                cv::Mat(pts_prev).reshape(1).convertTo(pts1_mat, CV_64F);
                cv::Mat(pts_curr).reshape(1).convertTo(pts2_mat, CV_64F);
                pts1_mat = pts1_mat.t();
                pts2_mat = pts2_mat.t();

                cv::Mat points4D;
                cv::triangulatePoints(P1, P2, pts1_mat, pts2_mat, points4D);
                Eigen::Matrix4f cam_prev_to_world = pose_prev.inverse();

                for (int i = 0; i < points4D.cols; i++) {
                    if (!pose_mask.empty() && pose_mask.at<uchar>(i) == 0) {
                        continue;
                    }

                    double w = points4D.at<double>(3, i);
                    if (std::abs(w) > 1e-7) {
                        double x = points4D.at<double>(0, i) / w;
                        double y = points4D.at<double>(1, i) / w;
                        double z = points4D.at<double>(2, i) / w;
                        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                            continue;
                        }

                        Eigen::Vector3f p_prev((float)x, (float)y, (float)z);
                        Eigen::Vector3f p_curr = R_rel * p_prev + t_rel * (float)pose_scale;
                        if (p_prev.z() <= 0.0f || p_curr.z() <= 0.0f || p_prev.z() > max_triangulated_depth) {
                            continue;
                        }

                        Eigen::Vector4f p_world_h = cam_prev_to_world * Eigen::Vector4f(p_prev.x(), p_prev.y(), p_prev.z(), 1.0f);
                        point_cloud->points_.push_back(Eigen::Vector3d(p_world_h.x(), p_world_h.y(), p_world_h.z()));

                        point_cloud->colors_.push_back(sample_rgb_color(cv2_color_frame, pts_curr[i]));
                    }
                }
            }
        } else {
            cv::imshow("Frame", cv2_original);
        }

        int keyCode = cv::waitKey(1);
        if (keyCode == 113 || g_shutdown) {
            break;
        }

        last_keypoints = keypoints;
        last_descriptors = descriptors;
        last_corners = corners;

        auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::cout << (1.0f / (t1 - t0) * 1000) << std::endl;
    }

    cap.release();
    cv::destroyAllWindows();

    open3d::visualization::DrawGeometries({point_cloud}, "Point Cloud");

    return 0;
}
