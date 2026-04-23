#ifndef PIPE_MESH_GENERATOR_H
#define PIPE_MESH_GENERATOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>

class PipeMeshGenerator {
public:
    // We only need the radius since the slice is already "rectified"
    PipeMeshGenerator(double r) : radius(r), vertex_count(0) {
        outfile.open("reconstructed_pipe.obj");
    }

    ~PipeMeshGenerator() { if (outfile.is_open()) outfile.close(); }

    void addSlice(const cv::Mat& slice, double global_z_start, double slice_length);

private:
    double radius;
    std::ofstream outfile;
    int vertex_count;
};

#endif
