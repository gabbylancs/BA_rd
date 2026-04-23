#include "PipeMeshGenerator.h"

void PipeMeshGenerator::addSlice(const cv::Mat& slice, double global_z_start, double slice_length) {
    int rows = slice.rows;
    int cols = slice.cols;

    // 1. Write Vertices (the physical points)
    for (int v = 0; v < rows; ++v) {
        double z = global_z_start + ((double)v / rows) * slice_length;
        for (int u = 0; u < cols; ++u) {
            double theta = (2.0 * CV_PI * u) / cols;
            double x = radius * std::cos(theta);
            double y = radius * std::sin(theta);

            // Format: v x y z
            outfile << "v " << x << " " << y << " " << z << "\n";

            // Format: vc r g b (Vertex Colors - optional, supported by MeshLab)
            cv::Vec3b color = slice.at<cv::Vec3b>(v, u);
            outfile << "vc " << color[2] / 255.0 << " " << color[1] / 255.0 << " " << color[0] / 255.0 << "\n";
        }
    }

    // 2. Write Faces (Connecting the points into a solid surface)
    // Only do this if you want a solid mesh rather than just points
    for (int v = 0; v < rows - 1; ++v) {
        for (int u = 0; u < cols; ++u) {
            int next_u = (u + 1) % cols;
            int current = vertex_count + v * cols + u + 1;
            int right = vertex_count + v * cols + next_u + 1;
            int down = vertex_count + (v + 1) * cols + u + 1;

            outfile << "f " << current << " " << right << " " << down << "\n";
        }
    }
    vertex_count += rows * cols;
}
