#pragma once

#include <Eigen/Dense>

namespace UPnPL {
namespace detail {

// The caller supplies a unit bearing in the body frame. Choosing the least
// aligned axis bounds the cross-product norm below by sqrt(2/3), including
// bearings whose third component is zero.
inline Eigen::Matrix<double, 3, 2>
pointBearingComplement(const Eigen::Vector3d &f) {
    Eigen::Index axis;
    f.cwiseAbs().minCoeff(&axis);
    const Eigen::Vector3d reference = Eigen::Vector3d::Unit(axis);
    Eigen::Matrix<double, 3, 2> Q;
    Q.col(0) = f.cross(reference).normalized();
    Q.col(1) = f.cross(Q.col(0)).normalized();
    return Q;
}

} // namespace detail
} // namespace UPnPL
