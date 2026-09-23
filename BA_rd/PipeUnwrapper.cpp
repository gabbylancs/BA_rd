#include "PipeUnwrapper.h"

PipeUnwrapper::PipeUnwrapper(double radius, double _fx, double _fy, double _cx, double _cy)
    : R(radius), fx(_fx), fy(_fy), cx(_cx), cy(_cy){
}

cv::Mat PipeUnwrapper::unwrap(const cv::Mat& src, double pitch, double yaw,
    double x_off, double y_off,
    int out_w, int out_h, double z_start, double z_end) 
{

    cv::Mat mapX(out_h, out_w, CV_32FC1);
    cv::Mat mapY(out_h, out_w, CV_32FC1);

    // 1. Setup Rotation (Camera's tilt relative to Pipe Axis)
    Eigen::Matrix3d R_pitch, R_yaw, R_cam, R_pipe;

    // To move points FROM Pipe Space TO Camera Space, 
    // we use the negative angles (the inverse rotation)
    double p = pitch;
    double y = yaw;

    R_pitch << 1, 0, 0,
        0, cos(p), -sin(p),
        0, sin(p), cos(p);

    R_yaw << cos(y), 0, sin(y),
        0, 1, 0,
        -sin(y), 0, cos(y);

    // Combined rotation
    R_cam = R_pitch * R_yaw;
    R_pipe = R_yaw * R_pitch;

    for (int v = 0; v < out_h; ++v) 
    {
        // Linearly space the depth (z) from start to end
        double z_pipe = z_start + (double)v / (out_h - 1) * (z_end - z_start);

        for (int u = 0; u < out_w; ++u) 
        {
            // Angle around the pipe circumference
            double theta = (2.0 * CV_PI * u) / (out_w - 1);

            // 2. Point on the Pipe Wall in "Pipe Coordinates"
            // We assume the pipe center is (0,0) and the camera is offset by x_off, y_off
            Eigen::Vector3d pt_pipe(R * cos(theta), R * sin(theta), z_pipe);

            // 3. Transform point to Camera-Local space
            // Shift by the camera offset first, then rotate into the camera's tilted frame
            // pt_cam is where that wall point "hits" the camera's coordinate system
            Eigen::Vector3d pt_cam = R_pipe * pt_pipe + Eigen::Vector3d(x_off, y_off, 0);


            // 4. Project to Image Pixels
            if (pt_cam.z() > 0.1) {
                mapX.at<float>(v, u) = (float)(pt_cam.x() * fx / pt_cam.z() + cx);
                mapY.at<float>(v, u) = (float)(pt_cam.y() * fy / pt_cam.z() + cy);
            }
            else {
                mapX.at<float>(v, u) = -1.0f;
                mapY.at<float>(v, u) = -1.0f;
            }
        }
    }

    cv::Mat dst;
    cv::remap(src, dst, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return dst;
}
