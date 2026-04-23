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
    // ... Calibration and offsets ...
    double radius = 67.42/2;
    double fx = 2608.721575, fy = 2584.771277, cx = 1493.821507, cy = 1191.874317;
    double robot_y_offset = -7;

    const double deg2rad = 3.14159265358979323846 / 180.0;
    const double rad2deg = 180.0 / 3.14159265358979323846;

    PipeVisualOdometry pvo(radius, fx, fy, cx, cy);
    PipeBundleAdjuster ba(fx, fy, cx, cy);

    std::string folder_path = "ds0//";
    std::string folder_look_up = "stitches_ds0.txt";
    PipeUnwrapper unwrapper(radius, fx, fy, cx, cy);
    PipeMeshGenerator meshGen(radius);

    // 1. Load the Lookup Map (Filename -> Distance)
    std::map<int, double> distance_map;
    std::ifstream lookup_file(folder_look_up);
    int file_id_key;
    double dist_val;

    if (lookup_file.is_open()) {
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

	double totalDistance = 0.0;

    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(folder_path))
    {
        // 1. Check for extension
        if (entry.path().extension() == ".JPG")
        {
            std::string stem = entry.path().stem().string();

            // 2. Filter: Only save if the filename ends in "_7"
            if (stem.length() >= 2 && stem.substr(stem.length() - 2) == "_8")// || stem.substr(stem.length() - 2) == "_8")
            {
                // Push the full, original path so imread still works
                image_files.push_back(entry.path().string());
            }
        }
    }


    // Helper lambda to get the "Clean ID" (removes last 2 chars from filename)
    auto get_clean_id = [](const std::string& path_str) {
        std::string stem = fs::path(path_str).stem().string();
        if (stem.length() > 2) {
            stem = stem.substr(0, stem.length() - 2); // Strip last 2 characters
        }
        return std::stoi(stem);
        };


    /* Numerical sort(e.g., 6710.jpg, 7296.jpg...)
    std::sort(image_files.begin(), image_files.end(), [](const std::string& a, const std::string& b) 
        {
        return std::stoi(fs::path(a).stem().string()) < std::stoi(fs::path(b).stem().string());
        });*/

        // Numerical sort using the clean ID
    std::sort(image_files.begin(), image_files.end(), [&](const std::string& a, const std::string& b) {
        return get_clean_id(a) < get_clean_id(b);
        });



    // 3. Process Frames
    cv::Mat img_ref = cv::imread(image_files[0]);
    int last_file_id = get_clean_id(image_files[0]); // Use clean ID for lookup
    double last_encoder_val = 0.0;
    if (distance_map.count(last_file_id))
    {
        last_encoder_val = distance_map[last_file_id];
    }
    else
    {
        std::cout << "Warning: No distance found for image " << last_file_id << std::endl;
    }

    for (size_t i = 1; i < image_files.size(); ++i)
    {
        cv::Mat img_next = cv::imread(image_files[i]);
        if (img_next.empty()) continue;

        if (pvo.processFrames(img_ref, img_next))
        {
            auto features = pvo.getInitialFeatures();
            if (features.size() > 0)
            {
                // Extract number from current filename
                int current_file_id = get_clean_id(image_files[i]);

                // Lookup distance from our map
                double current_encoder_val = 0.0;
                if (distance_map.count(current_file_id)) 
                {
                    current_encoder_val = distance_map[current_file_id];
                }
                else 
                {
                    std::cout << "Warning: No distance found for image " << current_file_id << std::endl;
                    continue; // Skip if no ground truth distance exists
                }

                double dz_guess = pvo.getEstimatedDistance();

				if (current_encoder_val - last_encoder_val > 2) //only perform BA if there's a significant movement (e.g., >2mm) to avoid noise
                {
                    double encoder_val = current_encoder_val - last_encoder_val;
					std::cout << "\nProcessing " << image_files[i] << " | Encoder Distance: " << encoder_val << " mm\n";
                    last_encoder_val = current_encoder_val;
                    // Step 4: Perform BA with the actual distance from the file
					ba.solve(features, dz_guess, 0.1, 0.12, 0.13, 0.11, robot_y_offset, 39.44, 100); //non zero values for r1,r2,p,y to prevent algo getting stuck

                    // Step 5: Extract Results
                    Eigen::Vector3d final_t = ba.getOptimizedTranslation();
                    Eigen::Vector3d final_rpy = ba.getOptimizedRPY(); // [roll2, shared_pitch, shared_yaw]
                    double x_off = ba.getOptimizedXOffset();
                    double y_off_opt = ba.getOptimizedYOffset();

                    std::cout << std::fixed << std::setprecision(3);
                    std::cout << "Match: " << current_file_id << "\n"
                        << " | Z: " << final_t.z() << " mm\n"
                        << " | Shared X Offset: " << x_off << " mm\n"
                        << " | Shared Y Offset: " << y_off_opt << " mm\n"  // Added to log
                        << " | Shared Pitch: " << final_rpy.y() * rad2deg << " deg\n"
                        << " | Shared Yaw: " << final_rpy.z() * rad2deg << " deg\n"
                        << " | Final Roll: " << final_rpy.x() * rad2deg << " deg" << std::endl;

                    double x_off_ = ba.getOptimizedXOffset();
                    double y_off_opt_ = ba.getOptimizedYOffset();
                    double p_rad_ = -final_rpy.y(); // Use the same orientation logic as your mesh
                    double y_rad_ = final_rpy.z();

                    img_ref = img_next.clone();

                    // --- Render 3D Pipe Mesh ---
                    cv::Mat img_mesh = img_next.clone();
                    double pipe_len = 1000.0; // Length of the mesh in mm
                    int num_rings = 10;       // Number of circular segments
                    int num_points = 32;      // Points per circle

                    // Convert RPY from Bundle Adjuster (ba) to radians
                    /*The Y-Axis (Yaw) Flip: Since your camera's Y-axis points down (OpenCV convention), 
                    it is already "pre-flipped" compared to standard Cartesian math. 
                    This internal inversion means the solver's "Left/Right" output already matches 
                    the mesh's movement without you doing anything.
                    The X-Axis (Pitch) Flip: Your X-axis (horizontal) is "normal," so it doesn't have that built-in flip. 
                    To account for the Observer vs. 
                    Object perspective—where the pipe must tilt Down if the camera tilts Up—you have to manually add that negative sign.*/

                    double p_rad = -final_rpy.y(); // Shared Pitch
                    double y_rad = final_rpy.z(); // Shared Yaw

                    // Rotation matrix for pipe orientation relative to camera
                    Eigen::Matrix3d R_yaw, R_pitch, R_pipe;
                    R_pitch << 1, 0, 0, 0, cos(p_rad), -sin(p_rad), 0, sin(p_rad), cos(p_rad);
                    R_yaw << cos(y_rad), 0, sin(y_rad), 0, 1, 0, -sin(y_rad), 0, cos(y_rad);
                    R_pipe = R_yaw * R_pitch;

                    for (int r = 0; r <= num_rings; ++r) {
                        double z_local = (pipe_len / num_rings) * r;
                        std::vector<cv::Point2f> current_ring_pts;

                        for (int p = 0; p <= num_points; ++p) {
                            double theta = (2.0 * CV_PI / num_points) * p;

                            // 1. Define point in pipe-local space (Circle around its own axis)
                            Eigen::Vector3d pt_pipe(radius * cos(theta), radius * sin(theta), z_local);

                            //x_off = 0;
                            // 2. Transform to Camera Space: P_cam = R_pipe * P_pipe + T_offset
                            Eigen::Vector3d pt_cam = R_pipe * pt_pipe + Eigen::Vector3d(x_off, -y_off_opt, 0);

                            // 3. Project to pixels
                            
                            if (pt_cam.z() > 1.0) { // Clip points behind camera
                                double u = (pt_cam.x() * fx) / pt_cam.z() + img_next.cols / 2;
                                double v = (pt_cam.y() * fy) / pt_cam.z() + img_next.rows / 2;
                                current_ring_pts.push_back(cv::Point2f(u, v));
                            }
                        }

                        // Draw the ring
                        for (size_t k = 1; k < current_ring_pts.size(); ++k) {
                            cv::line(img_mesh, current_ring_pts[k - 1], current_ring_pts[k], cv::Scalar(0, 255, 0), 2);
                        }
                    }

                    // Resize and show
                    cv::Mat img_small;
                    double mesh_scale = 800.0 / img_mesh.cols;
                    cv::resize(img_mesh, img_small, cv::Size(), mesh_scale, mesh_scale);
                    cv::imshow("Pipe Mesh Projection", img_small);
                    cv::waitKey(1);


                    // --- Save Results to Folder ---
                    std::string output_dir = "ds0_7_res3";
                    if (!fs::exists(output_dir)) {
                        fs::create_directory(output_dir);
                    }

                    // Construct filename: file_id + feature_vector_size (e.g., 6710_152.jpg)
                    std::string save_name = std::to_string(current_file_id) + "_" +
                        std::to_string(features.size()) + ".jpg";
                    std::string full_save_path = output_dir + "/" + save_name;

                    // Save the image with the mesh overlay
                    cv::imwrite(full_save_path, img_mesh);

                    std::cout << "Saved: " << full_save_path << std::endl;

                    // In your frame loop:
                    cv::Mat pipe_slice = unwrapper.unwrap(img_next, x_off, y_off_opt, p_rad, y_rad, 150.0, 50.0);

                    if (!pipe_slice.empty()) {
                        // 1. Convert to LAB color space to isolate brightness
                        cv::Mat lab_img;
                        cv::cvtColor(pipe_slice, lab_img,cv::COLOR_BGR2Lab);

                        // 2. Extract the L (Lightness) channel
                        std::vector<cv::Mat> channels;
                        cv::split(lab_img, channels);

                        // 3. Apply CLAHE to the Lightness channel only
                        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
                        clahe->apply(channels[0], channels[0]);

                        // 4. Merge back and convert to BGR
                        cv::Mat enhanced_slice;
                        cv::merge(channels, lab_img);
                        cv::cvtColor(lab_img, enhanced_slice, cv::COLOR_Lab2BGR);

                        // Display the results
                        cv::imshow("Focused & Enhanced Pipe Slice", enhanced_slice);
                    }

                    totalDistance += 40;

                    // 2. In the loop, after you get your slice:
                    // Assuming you know the global Z from your encoder
                    meshGen.addSlice(pipe_slice, totalDistance, 40.0);


                    cv::waitKey(1); // Ensure windows update


                }
            }
        }
    }
    return 0;
}


