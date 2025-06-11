#pragma once

#include <vector>

#include <Eigen/Dense>

#include <random>

namespace UPnPL {

using namespace std;

class UPnPL {
  public:
    UPnPL() {}

    void solveUPnPL_DLS(const vector<Eigen::Vector3d> &points_w,
                        const vector<Eigen::VectorXd> &lines_w,
                        const vector<Eigen::Vector3d> &uv_c,
                        const vector<Eigen::Vector3d> &normals_c,
                        const vector<int> &points_cam,
                        const vector<int> &lines_cam,
                        const vector<Eigen::Matrix3d> &Rbc,
                        const vector<Eigen::Vector3d> &tbc,
                        Eigen::Matrix3d &R_bw, Eigen::Vector3d &t_bw);

    void solveUPnPL_EPnPL(const vector<Eigen::Vector3d> &points_w,
                          const vector<Eigen::VectorXd> &lines_w,
                          const vector<Eigen::Vector3d> &uv_c,
                          const vector<Eigen::Vector3d> &normals_c,
                          const vector<int> &points_cam,
                          const vector<int> &lines_cam,
                          const vector<Eigen::Matrix3d> &Rbc,
                          const vector<Eigen::Vector3d> &tbc,
                          Eigen::Matrix3d &R_bw, Eigen::Vector3d &t_bw);

    void chooseControlPoints(const vector<Eigen::Vector3d> &points_w,
                             const vector<Eigen::VectorXd> &lines_w,
                             vector<Eigen::Vector3d> &control_points);

    void computeAlpha(const vector<Eigen::Vector3d> &points_w,
                      const vector<Eigen::VectorXd> &lines_w,
                      const vector<Eigen::Vector3d> &control_points,
                      vector<double> &alpha);

    void solveN1(const vector<Eigen::Vector3d> &control_points_w,
                 const Eigen::VectorXd &beta, const Eigen::VectorXd &y,
                 double &lambda);

    void normalization(const vector<Eigen::Vector3d> &points_w,
                       const vector<Eigen::VectorXd> &lines_w,
                       vector<Eigen::Vector3d> &points_w_n,
                       vector<Eigen::VectorXd> &lines_w_n, double &scale,
                       Eigen::Vector3d &center, bool normalize = false);

    double pointLineDistance(const Eigen::Vector3d &x, const Eigen::Vector3d &a,
                             const Eigen::Vector3d &b);

    Eigen::Matrix<double, 3, 9> phi(const Eigen::Vector3d &p);

    static void CGR2Rotation(const Eigen::Vector3d &s, Eigen::Matrix3d &R);

    static Eigen::Matrix3d crossProductMatrix(const Eigen::Vector3d &v) {
        Eigen::Matrix3d M;
        M << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
        return M;
    }
};

} // namespace UPnPL
