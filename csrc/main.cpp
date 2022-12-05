#include <iostream>
#include <Eigen/Dense>
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

struct FeatureResults
{
    cv::Mat corners;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

cv::Mat transform_image(cv::Mat in, int n_rows, int n_columns)
{
    cv::Mat out = cv::Mat(n_rows, n_columns, CV_8UC1);
    cv::resize(in, out, cv::Size(n_columns, n_rows), cv::INTER_LINEAR);
    cv::cvtColor(out, out, cv::COLOR_BGR2GRAY, 1);
    return out;
}

FeatureResults extract_features(cv::Mat frame, const int max_corners, double quality, double min_distance)
{
    cv::Mat corners;
    cv::goodFeaturesToTrack(frame, corners, max_corners, quality, min_distance);
    
    std::vector<cv::KeyPoint> keypoints;
    // Convert corners to keypoints
    for (int i = 0; i < max_corners; i += 1) {
        auto keypoint = cv::KeyPoint();
        keypoint.pt.x = corners.ptr(i)[0];
        keypoint.pt.y = corners.ptr(i)[1];
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

void draw_points(cv::Mat frame, cv::Mat features, float mul_x, float mul_y) 
{   
    for (int i = 0; i < features.rows; i += 1)
    {
        float u = (mul_x * features.ptr<float>(i)[0]);
        float v = (mul_y * features.ptr<float>(i)[1]);
        cv::Point center = cv::Point(u, v);
        cv::Scalar line_color(0, 255, 0);
        cv::circle(frame, center, 3.0, line_color, 1.0);
    }
}

int main(int argc, char *argv[]) 
{
    auto cap = cv::VideoCapture(DATA_INPUT);
    cv::Mat cv2_original;
    
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

        draw_points(cv2_original, corners, mul_x, mul_y);

        // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> frame;
        
        // std::cout << cv2_frame.channels() << std::endl;
        // cv::cv2eigen(cv2_frame, frame);

        // Eigen::IOFormat fmt(4, 0, ", ", "\n", "", "");

        // std::cout << frame.format(fmt) << std::endl;

        cv::imshow("Frame", cv2_original);

        int keyCode = cv::waitKey(1);
        if (keyCode == 113) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
