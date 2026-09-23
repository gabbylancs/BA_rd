#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class PipeUnwrapper {
public:
    PipeUnwrapper(double radius, double fx, double fy, double cx, double cy);

    // out_w: circumfrence pixels, out_h: depth pixels
    // z_range: how many mm of pipe wall we want to pull from this single frame
    cv::Mat unwrap(const cv::Mat& src, double pitch, double yaw,
        double x_off, double y_off,
        int out_w, int out_h, double z_start, double z_end);

private:
    double R, fx, fy, cx, cy;
};
