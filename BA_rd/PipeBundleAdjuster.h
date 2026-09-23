#ifndef PIPE_BUNDLE_ADJUSTER_H
#define PIPE_BUNDLE_ADJUSTER_H

#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <vector>
#include "PipeVisualOdometry.h"

struct BAFunctor 
{
    typedef double Scalar;
    enum { InputsAtCompileTime = Eigen::Dynamic, ValuesAtCompileTime = Eigen::Dynamic };
    typedef Eigen::VectorXd InputType;
    typedef Eigen::VectorXd ValueType; 
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> JacobianType;

    const std::vector<CylindricalFeature>& observations;
    double fx, fy, cx, cy, r, known_y_off, encoder_distance, alpha;

    // Store initial VO values as "anchors"
    double init_pitch, init_yaw;

    BAFunctor(const std::vector<CylindricalFeature>& obs, double _fx, double _fy, double _cx, double _cy, double _r, double _known_y_off, double enc_dist, double _alpha, double _p, double _y)
        : observations(obs), fx(_fx), fy(_fy), cx(_cx), cy(_cy), r(_r), known_y_off(_known_y_off), encoder_distance(enc_dist), alpha(_alpha), init_pitch(_p), init_yaw(_y) {
    }

    int operator()(const InputType& params, ValueType& fvec) const 
    {
        const double M_PI = 3.14159265359;
        const double deg2rad = M_PI / 180.0;

        double s_tx = params[0];
        double tz2 = params[1];
        // Only wrap angles if they are massive; fmod can sometimes jitter gradients
        double s_p = params[2];
        double s_y = params[3];
        double opt_y_off = params[6];

        const double delta = 1.5;
        const double delta2 = delta * delta;

        for (int f = 0; f < 2; ++f) 
        {
            double current_roll = params[4 + f];

            Eigen::AngleAxisd rA(current_roll, Eigen::Vector3d::UnitZ());
            Eigen::AngleAxisd pA(s_p, Eigen::Vector3d::UnitY());
            Eigen::AngleAxisd yA(s_y, Eigen::Vector3d::UnitX());
            Eigen::Matrix3d R = (rA * pA * yA).toRotationMatrix();

            Eigen::Vector3d T(s_tx, opt_y_off, (f == 0) ? 0.0 : tz2); //if first frame tz = 0 

            for (int i = 0; i < (int)observations.size(); ++i) //loop through features
            {
                double th = params[7 + i * 2];              // Parameterised coordinates
                double zw = params[8 + i * 2];
                Eigen::Vector3d P_w(r * std::cos(th), r * std::sin(th), zw); // Convert to World Frame

                Eigen::Vector3d P_c = R.transpose() * (P_w - T);             // Transform to Cam Frame
                double z_c = std::max(P_c.z(), 0.1);                         // Check if behind cam/OOB

                double res_u = (fx * P_c.x() / z_c + cx) - (f == 0 ? observations[i].c1_u : observations[i].c2_u); //Transform into image frame
                double res_v = (fy * P_c.y() / z_c + cy) - (f == 0 ? observations[i].c1_v : observations[i].c2_v);

                double r2 = res_u * res_u + res_v * res_v; //calc RMS
                double weight = 1.0 / (1.0 + r2 / delta2);
                double sqrt_w = std::sqrt(weight);

                int idx = i * 4 + f * 2;   //begin the error vector
                fvec[idx + 0] = res_u * sqrt_w;
                fvec[idx + 1] = res_v * sqrt_w;
            }
        }

        // --- Constraints & Anchors ---
        int base_idx = (int)observations.size() * 4;
        double sq_alpha = std::sqrt(alpha);

        // 1. Encoder Constraint
        fvec[base_idx + 0] = sq_alpha* 0.007 * (tz2 - encoder_distance);

        // 2. Limit Penalties (Using Pow 4 to ensure they are always positive/unsigned)
        double limit_x = 67.42 / 2.0;
        double limit_rads = 35.0 * deg2rad;

        fvec[base_idx + 1] = sq_alpha * std::pow(s_tx / limit_x, 4);
        fvec[base_idx + 2] = sq_alpha * std::pow(s_p / limit_rads, 4);
        fvec[base_idx + 3] = sq_alpha * std::pow(s_y / limit_rads, 4);

        // 3. Roll Barriers
        double r_limit_rad = 20.0 * deg2rad;
        fvec[base_idx + 4] = sq_alpha * std::pow(params[4] / r_limit_rad, 4);
        fvec[base_idx + 5] = sq_alpha * std::pow(params[5] / r_limit_rad, 4);

        // 4. Y-Offset Penalty (Squared difference is more stable than cubic)
        // Pulls opt_y_off back toward the physical known_y_off
        fvec[base_idx + 6] = sq_alpha * 0.01 * (opt_y_off - known_y_off);

        return 0;
    }

    int inputs() const { return 6 + ((int)observations.size() * 2); }
    // FIX: Must return + 4 to match the base_idx + 3 assignment
    int values() const { return ((int)observations.size() * 4) + 7; }
};

class PipeBundleAdjuster 
{
public:
    PipeBundleAdjuster(double _fx, double _fy, double _cx, double _cy);

    void solve(const std::vector<CylindricalFeature>& features,
        double initial_dz, double r1, double r2, double p, double y,
        double y_off, double enc_dist, double alpha);

    Eigen::Vector3d getOptimizedTranslation() const { return optimized_t; }
    Eigen::Vector3d getOptimizedRPY() const { return optimized_rpy; }
    double getOptimizedXOffset() const { return optimized_t.x(); }
    double getOptimizedYOffset() const { return optimized_t.y(); }

private:
    double fx, fy, cx, cy;
    Eigen::Vector3d optimized_t, optimized_rpy;
};
#endif