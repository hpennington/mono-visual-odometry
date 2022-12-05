#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "./config.h"

cv::Mat transform_image(cv::Mat in, int n_rows, int n_columns)
{
    cv::Mat out = cv::Mat(n_rows, n_columns, CV_8UC1);
    cv::resize(in, out, cv::Size(n_columns, n_rows), cv::INTER_LINEAR);
    cv::cvtColor(in, out, cv::COLOR_BGR2GRAY, 1);
    return out;
}

cv::Mat extract_features(cv::Mat frame, int max_corners, double quality, double min_distance)
{
    cv::Mat corners;
    cv::goodFeaturesToTrack(frame, corners, max_corners, quality, min_distance);
    return corners;
}

void draw_points(cv::Mat frame, cv::Mat features) 
{   
    for (int i = 0; i < features.rows; i += 1)
    {
        int u = features.ptr<float>(i)[0];
        int v = features.ptr<float>(i)[1];
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

        // Convert to B&W then resize and convert to Eigen matrix
        cv::Mat cv2_frame = transform_image(cv2_original, im_h, im_w);

        cv::Mat corners = extract_features(cv2_frame, max_corners, quality, min_distance);

        draw_points(cv2_original, corners);

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
