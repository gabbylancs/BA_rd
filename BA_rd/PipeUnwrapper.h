#ifndef PIPE_UNWRAPPER_H
#define PIPE_UNWRAPPER_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class PipeUnwrapper {
public:
    PipeUnwrapper(double r, double _fx, double _fy, double _cx, double _cy);

    /**
     * @param z_start How far in front of the camera the slice begins (mm)
     * @param view_length The "thickness" of the pipe ring to unroll (mm)
     */
    cv::Mat unwrap(const cv::Mat& src, double x_off, double y_off,
        double p_rad, double y_rad,
        double z_start = 50.0, double view_length = 400.0,
        int out_w = 1200, int out_h = 400);

private:
    double radius;
    double fx, fy, cx, cy;
};

#endif
