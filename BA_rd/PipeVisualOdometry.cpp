#include "PipeVisualOdometry.h"
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <cmath>



// Constructor to initialise camera calibration parameters and geometric pipe scale
PipeVisualOdometry::PipeVisualOdometry(double r, double _fx, double _fy, double _cx, double _cy)
    : radius(r), fx(_fx), fy(_fy), cx(_cx), cy(_cy), estimatedDistance(0.0) 
{

    // Construct the standard 3x3 camera intrinsic matrix
    K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

}



// Image enhancement pipeline to prepare raw frames for feature detection
cv::Mat PipeVisualOdometry::preprocess(const cv::Mat& img)
{
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Apply Contrast Limited Adaptive Histogram Equalization to normalise lighting inside the pipe
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(gray, gray);

    // Smooth image to reduce high-frequency noise and compression artifacts
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);

    return gray;
}



// Main execution loop for tracking camera movement between two sequential frames
bool PipeVisualOdometry::processFrames(cv::Mat& img1, cv::Mat& img2)
{
    if (img1.empty() || img2.empty()) return false;

    // Convert both frames to grayscale and boost local contrast
    cv::Mat gray1 = preprocess(img1);
    cv::Mat gray2 = preprocess(img2);

    // Compute keypoints and descriptors using ORB features
    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    cv::Mat desc1, desc2;
    orb->detectAndCompute(gray1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(gray2, cv::noArray(), kp2, desc2);

    // Find matching features using Hamming distance (suitable for binary ORB descriptors)
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);

    // Separate matching descriptor metadata into actual 2D coordinate lists
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : matches)
    {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }

    // Run a preliminary RANSAC filter via the fundamental matrix to drop wild mismatches
    std::vector<uchar> inliersMask;
    cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3.0, 0.99, inliersMask);

    // Collect all image points that passed the fundamental matrix criteria
    ransacMatches.clear();
    for (size_t i = 0; i < inliersMask.size(); i++)
    {
        if (inliersMask[i]) ransacMatches.push_back(matches[i]);
    }

    // Sort valid matches so the absolute best/closest feature pairs sit at the top
    std::sort(ransacMatches.begin(), ransacMatches.end(), [](const cv::DMatch& a, const cv::DMatch& b)
        {return a.distance < b.distance; });

    std::cout << "Total Matches: " << ransacMatches.size() << ", RANSAC Inliers: " << ransacMatches.size() << std::endl;

    // Cap the feature count at 600 to keep downstream geometry math fast
    if (ransacMatches.size() > 600) ransacMatches.resize(600);

    // Compute the Essential matrix using calibrated coordinates and extract relative camera pose
    cv::Mat inliersMask2;
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0, inliersMask2);
    cv::recoverPose(E, pts1, pts2, K, R, t, inliersMask);

    // Extract Euler angles (Pitch, Yaw, Roll) from the calculated 3x3 rotation matrix
    if (!R.empty())
    {
        const double rad2deg = 180.0 / 3.14159265358979323846;

        // Rotation around the X-axis
        pitch = std::atan2(-R.at<double>(2, 1), R.at<double>(2, 2)) * rad2deg;

        // Rotation around the Y-axis
        double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
        bool singular = sy < 1e-6;

        if (!singular)
        {
            yaw = std::atan2(R.at<double>(2, 0), sy) * rad2deg;
            roll = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0)) * rad2deg;
        }
        else
        {
            // Fallback parameters to prevent mathematical breakdown at gimbal lock gimbal boundaries
            yaw = std::atan2(R.at<double>(2, 0), sy) * rad2deg;
            roll = 0;
        }
    }

    initialFeatures.clear();


    // Use the known cylinder radius of the pipe to project and find the absolute physical scale
    if (!t.empty() && !ransacMatches.empty())
    {
        std::vector<cv::Point2f> inliers1, inliers2;
        std::vector<cv::Point3f> pts3D_frame1;

        for (size_t i = 0; i < inliersMask2.rows; i++)
        {
            if (inliersMask2.at<uchar>(i))
            {

                // Keep the point only if it belongs to our top-tier sorted match group
                bool isTopFeature = std::any_of(ransacMatches.begin(), ransacMatches.end(), [&](const cv::DMatch& m)
                {
                    return m.queryIdx == matches[i].queryIdx && m.trainIdx == matches[i].trainIdx;
                });

                if (!isTopFeature) continue;

                // Back-project the 2D pixel coordinate onto a 3D cylindrical pipe model surface
                cv::Point2f p = pts1[i];
                cv::Point2f p2 = pts2[i];
                double rx = (p.x - cx) / fx, ry = (p.y - cy) / fy;
                double s = radius / std::sqrt(rx * rx + ry * ry);

                pts3D_frame1.push_back(cv::Point3f(s * rx, s * ry, s * 1.0));
                inliers1.push_back(p);
                inliers2.push_back(p2);
                initialFeatures.emplace_back(std::atan2(ry, rx), s, p.x, p.y, p2.x, p2.y);
            }
        }

        // Loose constraint safety fallback: if filtering left us with no points, re-evaluate all available inliers
        if (inliers1.empty() || inliers1.size() < 2)
        {
            for (size_t i = 0; i < inliersMask2.rows; i++) {
                if (inliersMask2.at<uchar>(i))
                {
                    bool isTopFeature = std::any_of(ransacMatches.begin(), ransacMatches.end(),
                        [&](const cv::DMatch& m) {
                            return m.queryIdx == matches[i].queryIdx && m.trainIdx == matches[i].trainIdx;
                        });

                    cv::Point2f p = pts1[i];
                    cv::Point2f p2 = pts2[i];
                    double rx = (p.x - cx) / fx, ry = (p.y - cy) / fy;
                    double s = radius / std::sqrt(rx * rx + ry * ry);

                    pts3D_frame1.push_back(cv::Point3f(s * rx, s * ry, s * 1.0));
                    inliers1.push_back(p);
                    inliers2.push_back(p2);
                    initialFeatures.emplace_back(std::atan2(ry, rx), s, p.x, p.y, p2.x, p2.y);
                }
            }

        }


        // IGNORE DONT USE --------------------------------------------------------------
        cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);
        cv::Mat Rt2; cv::hconcat(R, t, Rt2);
        cv::Mat P2 = K * Rt2;
        cv::Mat pts4D;

        // Find 3D coordinate estimates by calculating the intersection of the two viewpoints
        cv::triangulatePoints(P1, P2, inliers1, inliers2, pts4D);

        // Compute the scaling ratio by comparing our known cylinder depth against the unscaled triangulated depth
        double scaleSum = 0; int count = 0;
        for (int i = 0; i < pts4D.cols; i++) 
        {
            float w = pts4D.at<float>(3, i);
            float z_tri = pts4D.at<float>(2, i) / w;
            if (z_tri > 0) 
            {
                scaleSum += (pts3D_frame1[i].z / z_tri);
                count++;
            }
        }

        // Apply the average scaling factor to convert the translation vector into real world units
        if (count > 0) estimatedDistance = cv::norm(t * (scaleSum / count));
    }
    return true;

}



// Export tracked travel changes and active coordinate metrics to a CSV structured text file
void PipeVisualOdometry::saveResults(const std::string& filename) 
{

    std::ofstream outFile(filename);
    outFile << "delta_z_mm: " << estimatedDistance << "\ncount: " << initialFeatures.size() << "\n";

    for (size_t i = 0; i < initialFeatures.size(); ++i) {
        auto& f = initialFeatures[i];
        outFile << i << "," << f.theta << "," << f.z << "," << f.c1_u << "," << f.c1_v << "\n";
    }

    outFile.close();

}



// Debug function to display live point matches tracked side-by-side across both images
void PipeVisualOdometry::display(cv::Mat img1, cv::Mat img2) 
{
    cv::Mat imgMatches;
    cv::hconcat(img1, img2, imgMatches);

    for (const auto& m : ransacMatches) {
        cv::Point2f pt1 = kp1[m.queryIdx].pt;
        cv::Point2f pt2 = kp2[m.trainIdx].pt + cv::Point2f((float)img1.cols, 0.0f);
        cv::line(imgMatches, pt1, pt2, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("Pipe VO", imgMatches);
    cv::waitKey(0);

}
