#include "PipeUnwrapper.h"

PipeUnwrapper::PipeUnwrapper(double r, double _fx, double _fy, double _cx, double _cy)
    : radius(r), fx(_fx), fy(_fy), cx(_cx), cy(_cy) {
}

cv::Mat PipeUnwrapper::unwrap(const cv::Mat& src, double x_off, double y_off,
    double p_rad, double y_rad,
    double z_start, double view_length,
    int out_w, int out_h) {
    if (src.empty()) return cv::Mat();

    cv::Mat map_x(out_h, out_w, CV_32FC1);
    cv::Mat map_y(out_h, out_w, CV_32FC1);

    // Rotation matrices
    Eigen::Matrix3d R_yaw, R_pitch, R_pipe;
    R_pitch << 1, 0, 0,
        0, std::cos(-p_rad), -std::sin(-p_rad),
        0, std::sin(-p_rad), std::cos(-p_rad);
    R_yaw << std::cos(y_rad), 0, std::sin(y_rad),
        0, 1, 0,
        -std::sin(y_rad), 0, std::cos(y_rad);
    R_pipe = R_yaw * R_pitch;

    for (int v = 0; v < out_h; ++v) {
        // Map v directly to the slice depth
        double z_local = z_start + ((double)v / out_h) * view_length;

        for (int u = 0; u < out_w; ++u) {
            // Map u to circumference
            double theta = (2.0 * CV_PI* u) / out_w;

            // 3D Point on the pipe wall
            Eigen::Vector3d pt_pipe(radius * std::cos(theta), radius * std::sin(theta), z_local);

            // Transform based on optimized robot pose (Sag + Tilt)
            Eigen::Vector3d pt_cam = R_pipe * pt_pipe + Eigen::Vector3d(x_off, -y_off, 0);

            // Project to image
            if (pt_cam.z() > 1.0) {
                map_x.at<float>(v, u) = (float)(pt_cam.x() * fx / pt_cam.z() + cx);
                map_y.at<float>(v, u) = (float)(pt_cam.y() * fy / pt_cam.z() + cy);
            }
            else {
                map_x.at<float>(v, u) = -1.0f;
                map_y.at<float>(v, u) = -1.0f;
            }
        }
    }

    cv::Mat dst;
    cv::remap(src, dst, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return dst;
}
