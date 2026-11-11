#include "point_constraints.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using Eigen::Vector3d;
using Matrix23 = Eigen::Matrix<double, 2, 3>;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    std::mt19937 rng(20261111);
    std::normal_distribution<double> normal;
    auto vector = [&]() {
        Vector3d v;
        for (int k = 0; k < 3; ++k) v(k) = normal(rng);
        return v;
    };
    std::vector<Vector3d> bearings;
    for (int k = 0; k < 3; ++k) {
        bearings.emplace_back(Vector3d::Unit(k));
        bearings.emplace_back(-Vector3d::Unit(k));
    }
    for (double h : {0.0, 1e-300, -1e-300, 1e-16, -1e-16, 1e-12,
                     -1e-12, 1e-8, -1e-8, 1e-3, -1e-3, 1.0, -1.0}) {
        for (int i = 0; i < 32; ++i) {
            const double angle = i * 0.197;
            bearings.emplace_back(Vector3d(std::cos(angle), std::sin(angle), h)
                                      .normalized());
        }
    }
    bearings.emplace_back(Vector3d::Ones().normalized());
    for (int i = 0; i < 10000; ++i) bearings.emplace_back(vector().normalized());

    double orthogonality = 0, null_residual = 0, projector_error = 0;
    double norm_identity = 0, min_sigma = 1, max_sigma = 1;
    for (const auto &f : bearings) {
        const auto Q = UPnPL::detail::pointBearingComplement(f);
        const Matrix23 D = Q.transpose();
        require(D.allFinite(), "Nonfinite complement");
        orthogonality = std::max(orthogonality,
            (Q.transpose() * Q - Eigen::Matrix2d::Identity()).norm());
        null_residual = std::max(null_residual, (D * f).norm());
        projector_error = std::max(projector_error,
            (D.transpose() * D - (Eigen::Matrix3d::Identity() - f * f.transpose())).norm());
        for (double scale : {0.1, 7.0, -3.0}) {
            const auto scaled_Q = UPnPL::detail::pointBearingComplement((scale * f).normalized());
            require((scaled_Q * scaled_Q.transpose() - Q * Q.transpose()).norm() < 2e-14,
                    "Homogeneous bearing scale changed the projector");
        }
        Eigen::JacobiSVD<Matrix23> svd(D);
        require(svd.rank() == 2, "Complement lost rank");
        min_sigma = std::min(min_sigma, svd.singularValues().minCoeff());
        max_sigma = std::max(max_sigma, svd.singularValues().maxCoeff());
        const Vector3d v = vector();
        const double cross = v.cross(f).squaredNorm();
        norm_identity = std::max(norm_identity,
            std::abs((D * v).squaredNorm() - cross) / std::max(1.0, v.squaredNorm()));
        // A point on the bearing through a nonzero camera center must vanish.
        const Vector3d camera = vector();
        const Vector3d point = camera + 7.3 * f;
        require((D * point - D * camera).norm() < 1e-13, "Incorrect RHS sign");
    }
    require(orthogonality < 2e-14 && null_residual < 2e-14 &&
            projector_error < 2e-14 && norm_identity < 2e-14,
            "Orthogonal projection identity failed");
    require(min_sigma > 1 - 2e-14 && max_sigma < 1 + 2e-14,
            "Complement is ill-conditioned");
    std::cout << std::setprecision(17)
              << "bearings=" << bearings.size()
              << " max_QtQ_error=" << orthogonality
              << " max_Qtf=" << null_residual
              << " max_projector_error=" << projector_error
              << " max_residual_identity_error=" << norm_identity
              << " singular_values=[" << min_sigma << ',' << max_sigma << "]\n";
    std::cout << "h,old_sigma_min,new_sigma_min,old_rank,new_rank\n";
    for (double h : {0.0, 1e-16, 1e-12, 1e-8, 1e-3}) {
        const Vector3d f = Vector3d(0.6, 0.8, h).normalized();
        Matrix23 old;
        old << -f.z(), 0, f.x(), 0, -f.z(), f.y();
        const Matrix23 D = UPnPL::detail::pointBearingComplement(f).transpose();
        Eigen::JacobiSVD<Matrix23> old_svd(old), new_svd(D);
        std::cout << h << ',' << old_svd.singularValues()(1) << ','
                  << new_svd.singularValues()(1) << ',' << old_svd.rank()
                  << ',' << new_svd.rank() << '\n';
    }
}
