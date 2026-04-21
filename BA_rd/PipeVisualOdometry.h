#ifndef PIPE_VISUAL_ODOMETRY_H
#define PIPE_VISUAL_ODOMETRY_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct CylindricalFeature
{
    double theta, z;
    int c1_u, c1_v, c2_u, c2_v;

    CylindricalFeature(double _t, double _z, int _c1u, int _c1v, int _c2u, int _c2v)
        : theta(_t), z(_z), c1_u(_c1u), c1_v(_c1v), c2_u(_c2u), c2_v(_c2v) {
    }
};

class PipeVisualOdometry
{
public:
    // Constructor: Sets up the camera and pipe environment
    PipeVisualOdometry(double r, double _fx, double _fy, double _cx, double _cy);

    // Processes two images to find movement and features
    bool processFrames(cv::Mat& img1, cv::Mat& img2);

    // Exports the results to a text file for PhD analysis/Seeding
    void saveResults(const std::string& filename);

    // Visualizes the matches and the calculated epipole
    void display(cv::Mat img1, cv::Mat img2);

    // Allows other classes to access the detected features
    std::vector<CylindricalFeature> getInitialFeatures() const
    {
        return initialFeatures;
    }

    // Getters
    double getEstimatedDistance() const { return estimatedDistance; }
    double getRoll() const { return roll; }
    double getPitch() const { return pitch; }
    double getYaw() const { return yaw; }

private:
    // Parameters
    double radius;
    double fx, fy, cx, cy;
    cv::Mat K;

    // Internal state
    cv::Mat R, t;
    std::vector<cv::KeyPoint> kp1, kp2;
    std::vector<cv::DMatch> ransacMatches;
    std::vector<CylindricalFeature> initialFeatures;
    double estimatedDistance, roll, pitch, yaw;

    // Helper for image cleanup
    cv::Mat preprocess(const cv::Mat& img);
};

#endif