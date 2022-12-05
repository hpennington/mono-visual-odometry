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
        cap.read(frame);

        cv::imshow("Frame", frame);
        int keyCode = cv::waitKey(1);

        if (keyCode == 113) {
            break;
        }
    }

    return 0;
}
