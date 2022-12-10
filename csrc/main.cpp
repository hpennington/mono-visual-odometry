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

#include "./config.h"

auto orb = cv::ORB::create();
auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);

struct FeatureResults
{
    cv::Mat corners;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

struct RansacTupleResult
{
    Eigen::Matrix<bool, Eigen::Dynamic, 1> mask;
    Eigen::MatrixXf model;
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

Eigen::MatrixXf make_homogeneous_single(Eigen::MatrixXf in)
{
    Eigen::MatrixXf out = Eigen::MatrixXf::Ones(1, in.rows() + 1);
    for (int i = 0; i < in.rows(); i += 1)
    {
        out(0, 0) = in(0);
        out(0, 1) = in(1);
        out(0, 2) = 1.0;
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

Eigen::MatrixXf normalize(Eigen::MatrixXf T, Eigen::MatrixXf x)
{
    auto X = make_homogeneous_single(x);
    auto result = (T * X.transpose()).transpose()(Eigen::all, Eigen::seq(0, 2));
    // std::cout << result << std::endl;
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
        Eigen::MatrixXf F = svd.matrixV().transpose()(Eigen::last-1, Eigen::seq(0, Eigen::last)).reshaped(3, 3);        
        
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
    
        auto numerator = ((Y_prime.transpose().cwiseProduct(Fx)).colwise().sum() + (Y_prime.transpose().cwiseProduct(Fx)).colwise().sum()).array().sqrt().abs();
        auto denominator = (pow(Fx(0), 2) + pow(Fx(1), 2) + pow(Fty(0), 2) + pow(Fty(1), 2));

        return (numerator / denominator).transpose();
    }
};

RansacTupleResult ransac(FundamentalMatrixTransform model, Eigen::Matrix<float, Eigen::Dynamic, 2> kps1, Eigen::Matrix<float, Eigen::Dynamic, 2> kps2)
{
    RansacTupleResult result;
    int max_inliers = -1;

    for (int i = 0; i < ransac_max_trials; i += 1)
    {
        int rand_value = rand() % (kps1.rows() - ransac_minsamples);
        auto kps1_sub = kps1(Eigen::seq(rand_value, rand_value + ransac_minsamples), Eigen::all);
        auto kps2_sub = kps2(Eigen::seq(rand_value, rand_value + ransac_minsamples), Eigen::all);
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

int minimum(int a, int b)
{
    return a > b ? b : a;
}

FeatureResults extract_features(cv::Mat frame, const int max_corners, double quality, double min_distance)
{
    cv::Mat corners;
    cv::goodFeaturesToTrack(frame, corners, max_corners, quality, min_distance);
    
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

    FeatureResults result;
    result.corners = corners;
    result.keypoints = keypoints;
    result.descriptors = descriptors;

    return result;
}

struct MatchReturn
{
   std::vector<std::vector<std::vector<float>>> pairs;
   std::vector<std::vector<std::vector<float>>> norm_pairs; 
};

MatchReturn match_frames(cv::Mat corners1, cv::Mat corners2, std::vector<cv::KeyPoint> kps1, std::vector<cv::KeyPoint> kps2, cv::Mat descriptors1, cv::Mat descriptors2, Eigen::MatrixXf T)
{
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<std::vector<std::vector<float>>> pairs;
    
    matcher->knnMatch(descriptors1, descriptors2, matches, 2);

    std::vector<std::vector<std::vector<float>>> lowes_matches;
    
    for (int j = 0; j < matches.size(); j += 1) 
    {   
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
            // std::vector<std::vector<float>> pair_norm = {{pt1x, pt1y}, {pt2x, pt2y}};
            // norm_pairs.push_back(pair_norm)
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

    if (left_pt.rows() >= ransac_minsamples && right_pt.rows() >= ransac_minsamples)
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

        MatchReturn res;
        res.pairs = sub;
        res.norm_pairs = sub_norm;

        return res;

    }

    MatchReturn result2;
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

int main(int argc, char *argv[]) 
{
    auto cap = cv::VideoCapture(DATA_INPUT);
    cv::Mat cv2_original;
    std::vector<cv::KeyPoint> last_keypoints;
    cv::Mat last_descriptors;
    cv::Mat last_corners;
    auto T = create_normalization_matrix(im_h, im_w);

    while (cap.isOpened()) {
        bool frameGrabbed = cap.read(cv2_original);
        
        if (frameGrabbed == false) {
            break;
        }

        // Convert to B&W then resize
        cv::Mat cv2_frame = transform_image(cv2_original, im_h, im_w);
        float mul_x = (float)cv2_original.cols / (float)cv2_frame.cols;
        float mul_y = (float)cv2_original.rows / (float)cv2_frame.rows;

        FeatureResults results = extract_features(cv2_frame, max_corners, quality, min_distance);
        cv::Mat corners = results.corners;
        std::vector<cv::KeyPoint> keypoints = results.keypoints;

        cv::Mat descriptors = results.descriptors;

        if (last_descriptors.rows > 0 && last_keypoints.size() > 0 && last_corners.rows > 0) 
        {
            MatchReturn result = match_frames(corners, last_corners, keypoints, last_keypoints, descriptors, last_descriptors, T);
            auto pairs = result.pairs;
            
            auto norm_pairs = result.norm_pairs;
            
            draw_points(cv2_original, pairs, mul_x, mul_y);
            cv::imshow("Frame", cv2_original);

            int keyCode = cv::waitKey(1);
            if (keyCode == 113) {
                break;
            }
        }

        last_keypoints = keypoints;
        last_descriptors = descriptors;
        last_corners = corners;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
