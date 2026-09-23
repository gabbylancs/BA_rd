#include "PipeVisualOdometry.h"
#include "PipeBundleAdjuster.h"
#include "PipeUnwrapper.h"
#include "PipeMeshGenerator.h"
#include <iostream>
#include <fstream>
#include <map>
#include <filesystem>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;


int main()
{

    // ------------------------------------------------------------------------
    // CONSTANTS/DECLARATIONS
    // ------------------------------------------------------------------------
    double radius = 67.42 / 2;
    double fx = 2608.721575, fy = 2584.771277, cx = 1493.821507, cy = 1191.874317;
    double robot_y_offset = -8;
    const double deg2rad = 3.14159265358979323846 / 180.0;
    const double rad2deg = 180.0 / 3.14159265358979323846;
    std::string folder_path = "ds0//";
    std::string folder_look_up = "stitches_ds0.txt";
    double totalDistance = 0.0;
    std::vector<std::string> image_files;

    // ------------------------------------------------------------------------
    // INITIALIZATION
    // ------------------------------------------------------------------------
    PipeVisualOdometry pvo(radius, fx, fy, cx, cy);
    PipeBundleAdjuster ba(fx, fy, cx, cy);
    PipeUnwrapper unwrapper(radius, fx, fy, cx, cy); //dont use
    PipeMeshGenerator meshGen(radius); //dont use


    // ------------------------------------------------------------------------
    // FILENAME TO REAL DISTANCE VALS
    // ------------------------------------------------------------------------
    std::map<int, double> distance_map;
    std::ifstream lookup_file(folder_look_up);
    int file_id_key;
    double dist_val;

    if (lookup_file.is_open())
    {
        while (lookup_file >> file_id_key >> dist_val)
        {
            distance_map[file_id_key] = dist_val;
        }
        lookup_file.close();
    }
    else
    {
        std::cerr << "Error: Could not open " << folder_look_up << std::endl;
        return -1;
    }


    for (const auto& entry : fs::directory_iterator(folder_path))
    {
        if (entry.path().extension() == ".JPG")
        {
            std::string stem = entry.path().stem().string();

            if (stem.length() >= 2 && stem.substr(stem.length() - 2) == "_7")
            {
                image_files.push_back(entry.path().string());
            }
        }
    }

    auto get_clean_id = [](const std::string& path_str) 
    {
        std::string stem = fs::path(path_str).stem().string();
        if (stem.length() > 2)
        {
            stem = stem.substr(0, stem.length() - 2);
        }
        return std::stoi(stem);
    };

    //c sort images in numerical order 
    std::sort(image_files.begin(), image_files.end(), [&](const std::string& a, const std::string& b)
    {
        return get_clean_id(a) < get_clean_id(b);
    });


    // Load first images
    cv::Mat img_ref = cv::imread(image_files[0]);
    int last_file_id = get_clean_id(image_files[0]);
    double last_encoder_val = 0.0;

    if (distance_map.count(last_file_id))
    {
        last_encoder_val = distance_map[last_file_id];
    }
    else
    {
        std::cout << "Warning: No distance found for image " << last_file_id << std::endl;
    }


    // ============================================================================
    // FRAME TRACKING
    // ============================================================================
    for (size_t i = 1; i < image_files.size(); ++i)
    {
        cv::Mat img_next = cv::imread(image_files[i]);
        if (img_next.empty())
        {
            continue;
        }

        if (pvo.processFrames(img_ref, img_next))
        {
            auto features = pvo.getInitialFeatures();

            if (features.size() > 0)
            {
                int current_file_id = get_clean_id(image_files[i]);
                double current_encoder_val = 0.0;

                if (distance_map.count(current_file_id))
                {
                    current_encoder_val = distance_map[current_file_id];
                }
                else
                {
                    std::cout << "Warning: No distance found for image " << current_file_id << std::endl;
                    continue;
                }

                double dz_guess = pvo.getEstimatedDistance(); // dont use

                if (current_encoder_val - last_encoder_val > 0)
                {
                    double encoder_val = current_encoder_val - last_encoder_val;
                    std::cout << "\nProcessing " << image_files[i] << " | Encoder Distance: " << encoder_val << " mm\n";
                    last_encoder_val = current_encoder_val;

                    // SOLVE -------------------
                    ba.solve(features, 30, 0.1, 0.12, 0.13, 0.11, robot_y_offset, 40, 0.5);
                    Eigen::Vector3d final_t = ba.getOptimizedTranslation();
                    Eigen::Vector3d final_rpy = ba.getOptimizedRPY();
                    double x_off = ba.getOptimizedXOffset();
                    double y_off_opt = ba.getOptimizedYOffset();


                    // --------------------------------------------------------
                    // PRINT RESULTS
                    // --------------------------------------------------------
                    std::cout << std::fixed << std::setprecision(3);
                    std::cout << "Match: " << current_file_id << "\n"
                        << " | Z: " << final_t.z() << " mm\n"
                        << " | Shared X Offset: " << x_off << " mm\n"
                        << " | Shared Y Offset: " << y_off_opt << " mm\n"
                        << " | Shared Pitch: " << final_rpy.y() * rad2deg << " deg\n"
                        << " | Shared Yaw: " << final_rpy.z() * rad2deg << " deg\n"
                        << " | Final Roll: " << final_rpy.x() * rad2deg << " deg" << std::endl;

                    double x_off_ = ba.getOptimizedXOffset();
                    double y_off_opt_ = ba.getOptimizedYOffset();
                    double p_rad_ = -final_rpy.y();
                    double y_rad_ = final_rpy.z();


                    // --------------------------------------------------------
                    // CENTRALISATION ANALYSIS --------------------------------------------------
                    // --------------------------------------------------------
                    img_ref = img_next.clone();


                    cv::Mat img_mesh = img_next.clone();
                    double pipe_len = 1000.0;
                    int num_rings = 10;
                    int num_points = 32;

                    double p_rad = -final_rpy.y();
                    double y_rad = final_rpy.z();

                    Eigen::Matrix3d R_yaw, R_pitch, R_pipe;
                    R_pitch << 1, 0, 0, 0, cos(p_rad), -sin(p_rad), 0, sin(p_rad), cos(p_rad);
                    R_yaw << cos(y_rad), 0, sin(y_rad), 0, 1, 0, -sin(y_rad), 0, cos(y_rad);
                    R_pipe = R_yaw * R_pitch;

                    for (int r = 0; r <= num_rings; ++r)
                    {
                        double z_local = (pipe_len / num_rings) * r;
                        std::vector<cv::Point2f> current_ring_pts;

                        for (int p = 0; p <= num_points; ++p)
                        {
                            double theta = (2.0 * CV_PI / num_points) * p;


                            // 1. DEFINE POINT IN PIPE-LOCAL SPACE (CYLINDRICAL AXIS)
                            Eigen::Vector3d pt_pipe(radius * cos(theta), radius * sin(theta), z_local);


                            // 2. TRANSFORM TO CAMERA SPACE: P_cam = R_pipe * P_pipe + T_offset
                            Eigen::Vector3d pt_cam = R_pipe * pt_pipe + Eigen::Vector3d(x_off, -y_off_opt, 0);


                            // 3. PERSPECTIVE PROJECTION INTEGRATION VIA PINHOLE MODEL
                            if (pt_cam.z() > 1.0)
                            {
                                double u = (pt_cam.x() * fx) / pt_cam.z() + img_next.cols / 2;
                                double v = (pt_cam.y() * fy) / pt_cam.z() + img_next.rows / 2;
                                current_ring_pts.push_back(cv::Point2f(u, v));
                            }

                        }

                        for (size_t k = 1; k < current_ring_pts.size(); ++k)
                        {
                            cv::line(img_mesh, current_ring_pts[k - 1], current_ring_pts[k], cv::Scalar(0, 255, 0), 2);
                        }

                    }

                    cv::Mat img_small;
                    double mesh_scale = 800.0 / img_mesh.cols;
                    cv::resize(img_mesh, img_small, cv::Size(), mesh_scale, mesh_scale);
                    //cv::imshow("Pipe Mesh Projection", img_small);
                    //cv::waitKey(2);

                    std::string output_dir = "centralisation_ds1";

                    if (!fs::exists(output_dir))
                    {
                        fs::create_directory(output_dir);
                    }


                    // --------------------------------------------------------
                    // EXPORT TRACKING PATH IDENTIFIER ALIAS GENERATION
                    // --------------------------------------------------------
                    std::string save_name = std::to_string(current_file_id) + "__" + std::to_string(features.size()) + ".jpg";
                    std::string full_save_path = output_dir + "/" + save_name;
                    std::string full_save_path2 = output_dir + "/unwrap" + save_name;



                    //cv::imwrite(full_save_path, img_mesh);
                    std::cout << "Saved: " << full_save_path << std::endl;



                    double circumference = 2.0 * CV_PI * radius;
                    double ppm = 1200.0 / circumference;


                    double z_range = 200.0 - 120.0;
                    int scaled_h = static_cast<int>(z_range * ppm);



                    cv::Mat unwrappedSlice = unwrapper.unwrap(img_next, 0, 0, 0, 0, 1200, scaled_h, 120, 200);
                    //cv::imshow("Unwrapped Pipe Slice", unwrappedSlice);
                    //cv::imwrite(full_save_path2, unwrappedSlice);


                    cv::waitKey(100);

                }

            }

        }

    }

    return 0;

}
