#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>

#include "./config.h"

int main(int argc, char *argv[]) 
{
    auto cap = cv::VideoCapture(DATA_INPUT);
    cv::Mat frame;

    while (cap.isOpened()) {
        bool frameGrabbed = cap.read(frame);
        
        if (frameGrabbed == false) {
            break;
        }

        cv::imshow("Frame", frame);

        int keyCode = cv::waitKey(1);
        if (keyCode == 113) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
