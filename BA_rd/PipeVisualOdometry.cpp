#include "PipeVisualOdometry.h"
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <cmath>

PipeVisualOdometry::PipeVisualOdometry(double r, double _fx, double _fy, double _cx, double _cy)
    : radius(r), fx(_fx), fy(_fy), cx(_cx), cy(_cy), estimatedDistance(0.0) {
    K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
}

cv::Mat PipeVisualOdometry::preprocess(const cv::Mat& img) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(gray, gray);
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
    return gray;
}

bool PipeVisualOdometry::processFrames(cv::Mat& img1, cv::Mat& img2) {
    if (img1.empty() || img2.empty()) return false;

    cv::Mat gray1 = preprocess(img1);
    cv::Mat gray2 = preprocess(img2);

    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    cv::Mat desc1, desc2;
    orb->detectAndCompute(gray1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(gray2, cv::noArray(), kp2, desc2);

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);

    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : matches)
    {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }

    std::vector<uchar> inliersMask;
    cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3.0, 0.99, inliersMask);

    ransacMatches.clear();
    for (size_t i = 0; i < inliersMask.size(); i++) {
        if (inliersMask[i]) ransacMatches.push_back(matches[i]);
    }

    std::sort(ransacMatches.begin(), ransacMatches.end(), [](const cv::DMatch& a, const cv::DMatch& b)
        {return a.distance < b.distance; });

    std::cout << "Total Matches: " << ransacMatches.size() << ", RANSAC Inliers: " << ransacMatches.size() << std::endl;

    if (ransacMatches.size() > 600) ransacMatches.resize(600);

    cv::Mat inliersMask2;
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0, inliersMask2);
    cv::recoverPose(E, pts1, pts2, K, R, t, inliersMask);

    if (!R.empty())
    {
        const double rad2deg = 180.0 / 3.14159265358979323846;

        // Pitch (X-axis rotation)
        pitch = std::atan2(-R.at<double>(2, 1), R.at<double>(2, 2)) * rad2deg;

        // Yaw (Y-axis rotation)
        // We use the element at (2,0) and the projection of the other elements
        double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
        bool singular = sy < 1e-6; // Check for gimbal lock

        if (!singular) {
            yaw = std::atan2(R.at<double>(2, 0), sy) * rad2deg;
            roll = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0)) * rad2deg;
        }
        else {
            yaw = std::atan2(R.at<double>(2, 0), sy) * rad2deg;
            roll = 0;
        }
    }

    initialFeatures.clear();
    if (!t.empty() && !ransacMatches.empty()) {
        std::vector<cv::Point2f> inliers1, inliers2;
        std::vector<cv::Point3f> pts3D_frame1;

        for (size_t i = 0; i < inliersMask2.rows; i++) {
            if (inliersMask2.at<uchar>(i))
            {
                // 2. CHECK: Is this specific match 'i' part of our top 100 ransacMatches ?
                    // We compare the queryIdx/trainIdx to ensure it's the exact same match.
                bool isTopFeature = std::any_of(ransacMatches.begin(), ransacMatches.end(),
                    [&](const cv::DMatch& m) {
                        return m.queryIdx == matches[i].queryIdx && m.trainIdx == matches[i].trainIdx;
                    });

                //if (!isTopFeature) continue; // Skip if it's an inlier but not in our "Top 100"

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

        cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);
        cv::Mat Rt2; cv::hconcat(R, t, Rt2);
        cv::Mat P2 = K * Rt2;
        cv::Mat pts4D;
        cv::triangulatePoints(P1, P2, inliers1, inliers2, pts4D);

        double scaleSum = 0; int count = 0;
        for (int i = 0; i < pts4D.cols; i++) {
            float w = pts4D.at<float>(3, i);
            float z_tri = pts4D.at<float>(2, i) / w;
            if (z_tri > 0) {
                scaleSum += (pts3D_frame1[i].z / z_tri);
                count++;
            }
        }
        if (count > 0) estimatedDistance = cv::norm(t * (scaleSum / count));
    }
    return true;
}

void PipeVisualOdometry::saveResults(const std::string& filename) {
    std::ofstream outFile(filename);
    outFile << "delta_z_mm: " << estimatedDistance << "\ncount: " << initialFeatures.size() << "\n";
    for (size_t i = 0; i < initialFeatures.size(); ++i) {
        auto& f = initialFeatures[i];
        outFile << i << "," << f.theta << "," << f.z << "," << f.c1_u << "," << f.c1_v << "\n";
    }
    outFile.close();
}

void PipeVisualOdometry::display(cv::Mat img1, cv::Mat img2) {
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