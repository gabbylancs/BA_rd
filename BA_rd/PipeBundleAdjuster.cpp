#include "PipeBundleAdjuster.h"

PipeBundleAdjuster::PipeBundleAdjuster(double _fx, double _fy, double _cx, double _cy)
    : fx(_fx), fy(_fy), cx(_cx), cy(_cy) {
    optimized_t.setZero();
    optimized_rpy.setZero();
}

void PipeBundleAdjuster::solve(const std::vector<CylindricalFeature>& features,
    double initial_dz, double r1, double r2, double p, double y,
    double y_off, double enc_dist, double alpha) {
    int n_pts = (int)features.size();
    Eigen::VectorXd x(6 + n_pts * 2);
    x.setZero();

    // Seed shared camera parameters
    x[0] = 0.1;        // shared_tx
    x[1] = initial_dz; // tz2
    x[2] = p;          // shared_pitch
    x[3] = y;          // shared_yaw
    x[4] = r1;         // roll1
    x[5] = r2;         // roll2

    for (int i = 0; i < n_pts; ++i) {
        x[6 + i * 2] = features[i].theta;
        x[7 + i * 2] = features[i].z;
    }

    BAFunctor functor(features, fx, fy, cx, cy, 67.42/2, y_off, enc_dist, alpha, p, y);
    Eigen::NumericalDiff<BAFunctor> numDiff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<BAFunctor>> lm(numDiff);

    lm.parameters.maxfev = 200;
    lm.parameters.factor = 0.1;       // Smaller values = smaller initial steps
    lm.parameters.ftol = 1e-14;      // Tighten function tolerance
    lm.parameters.xtol = 1e-14;      // Tighten parameter tolerance
    lm.minimize(x);

    // Optimized results (returning frame 2's pose)
    optimized_t = Eigen::Vector3d(x[0], y_off, x[1]);
    optimized_rpy = Eigen::Vector3d(x[5], x[2], x[3]); // [roll2, shared_p, shared_y]
}
