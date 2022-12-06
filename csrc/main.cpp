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
auto bf_matcher = cv::BFMatcher::create(cv::NORM_HAMMING);

std::vector<cv::KeyPoint> last_keypoints;
cv::Mat last_descriptors;
cv::Mat last_corners;

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

std::vector<std::vector<std::vector<float>>> match_frames(std::vector<cv::KeyPoint> kps1, std::vector<cv::KeyPoint> kps2, cv::Mat descriptors1, cv::Mat descriptors2)
{
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<std::vector<std::vector<float>>> pairs;
    bf_matcher->knnMatch(descriptors1, descriptors1, matches, 2);

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
            pairs.push_back(pair);
        }
    }

    return pairs;
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

        if (last_descriptors.rows > 0 && last_keypoints.size() > 0 && last_corners.rows > 0) 
        {
            auto pairs = match_frames(keypoints, last_keypoints, descriptors, last_descriptors);
            
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
