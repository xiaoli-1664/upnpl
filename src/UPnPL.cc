#include "UPnPL.h"
#include "constructM.h"
#include <iostream>

namespace UPnPL {

void UPnPL::solveUPnPL(const vector<Eigen::Vector3d> &points_w,
                       const vector<Eigen::VectorXd> &lines_w,
                       const vector<Eigen::Vector3d> &uv_c,
                       const vector<Eigen::Vector3d> &normals_c,
                       const vector<int> &points_cam,
                       const vector<int> &lines_cam,
                       const vector<Eigen::Matrix3d> &Rbc,
                       const vector<Eigen::Vector3d> &tbc,
                       Eigen::Matrix3d &R_bw, Eigen::Vector3d &t_bw) {
    int n = points_w.size(), m = lines_w.size();

    double scale;
    Eigen::Vector3d center;

    vector<Eigen::Vector3d> points_w_n;
    vector<Eigen::VectorXd> lines_w_n;

    normalization(points_w, lines_w, points_w_n, lines_w_n, scale, center,
                  true);

    vector<Eigen::Vector3d> tbc_n = tbc;
    for (auto &t : tbc_n) {
        t *= scale;
    }

    vector<Eigen::Vector3d> uv_b(n);
    for (int i = 0; i < n; ++i) {
        uv_b[i] = uv_c[i];
        uv_b[i].normalize();
        uv_b[i] = Rbc[points_cam[i]] * uv_b[i];
    }

    vector<Eigen::Vector3d> normals_b(m);
    for (int i = 0; i < m; ++i) {
        normals_b[i] = Rbc[lines_cam[i]] * normals_c[i];
        normals_b[i].normalize();
    }

    Eigen::Matrix<double, 1, 9> I0;
    I0 << Eigen::Vector3d::Zero().transpose(),
        Eigen::Vector3d::Ones().transpose(),
        Eigen::Vector3d::Zero().transpose();

    Eigen::Matrix3d H4 = n * Eigen::Matrix3d::Identity();
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(3, 2 * m + 3 * n);
    Eigen::Matrix<double, 3, 9> G_phi_i, G_phi_j;
    Eigen::Matrix<double, 3, 1> G_pi, G_pj;
    G_phi_i.setZero();
    G_phi_j.setZero();
    G_pi.setZero();
    G_pj.setZero();
    for (int i = 0; i < m; ++i) {
        Eigen::Matrix3d uuT = normals_b[i] * normals_b[i].transpose();
        H4 += 2 * uuT;
        G.block<3, 1>(0, 2 * i) = -normals_b[i];
        G.block<3, 1>(0, 2 * i + 1) = -normals_b[i];
        int cam = lines_cam[i];
        G_phi_i +=
            -uuT * (phi(lines_w_n[i].head<3>() + lines_w_n[i].tail<3>()) +
                    2 * tbc_n[cam] * I0);
        G_pi += -uuT * (lines_w_n[i].head<3>() + lines_w_n[i].tail<3>() -
                        2 * tbc_n[cam]);
    }
    for (int j = 0; j < n; ++j) {
        Eigen::Matrix3d UV = uv_b[j] * uv_b[j].transpose();
        H4 -= UV;
        Eigen::Matrix3d G_j = UV - Eigen::Matrix3d::Identity();
        G.block<3, 3>(0, 2 * m + 3 * j) = G_j;
        int cam = points_cam[j];
        G_phi_j += G_j * (phi(points_w_n[j]) + tbc_n[cam] * I0);
        G_pj += G_j * (points_w_n[j] - tbc_n[cam]);
    }
    if (H4.determinant() < 1e-6) {
        std::cerr << "H4 is singular, cannot proceed with UPnPL." << std::endl;
        return;
    }
    H4 = H4.inverse().eval();
    G = H4 * G;
    G_phi_i = H4 * G_phi_i;
    G_phi_j = H4 * G_phi_j;
    G_pi = H4 * G_pi;
    G_pj = H4 * G_pj;

    Eigen::Matrix<double, 10, 10> A = Eigen::Matrix<double, 10, 10>::Zero();

    for (int i = 0; i < m; ++i) {
        Eigen::Matrix<double, 1, 9> Ai1, Ai2;
        Ai1.setZero();
        Ai2.setZero();

        Eigen::Matrix<double, 1, 1> bi1, bi2;
        bi1.setZero();
        bi2.setZero();

        Eigen::Vector3d ui_b = normals_b[i];

        int cam = lines_cam[i];
        Ai1 = ui_b.transpose() * (phi(lines_w_n[i].head<3>()) +
                                  tbc_n[cam] * I0 + G_phi_i + G_phi_j);
        Ai2 = ui_b.transpose() * (phi(lines_w_n[i].tail<3>()) +
                                  tbc_n[cam] * I0 + G_phi_i + G_phi_j);
        bi1 = ui_b.transpose() *
              (lines_w_n[i].head<3>() - tbc_n[cam] + G_pi + G_pj);
        bi2 = ui_b.transpose() *
              (lines_w_n[i].tail<3>() - tbc_n[cam] + G_pi + G_pj);

        A.block<9, 9>(0, 0) += Ai1.transpose() * Ai1 + Ai2.transpose() * Ai2;
        A.block<9, 1>(0, 9) += Ai1.transpose() * bi1 + Ai2.transpose() * bi2;
        A.block<1, 9>(9, 0) += bi1 * Ai1 + bi2 * Ai2;
        A.block<1, 1>(9, 9) += bi1 * bi1 + bi2 * bi2;
    }

    for (int j = 0; j < n; ++j) {
        Eigen::Matrix<double, 3, 9> Aj;
        Aj.setZero();

        Eigen::Vector3d bj = Eigen::Vector3d::Zero();
        Eigen::Matrix3d PI =
            uv_b[j] * uv_b[j].transpose() - Eigen::Matrix3d::Identity();

        int cam = points_cam[j];
        Aj = PI * (G_phi_i + G_phi_j + phi(points_w_n[j]) + tbc_n[cam] * I0);
        bj = PI * (points_w_n[j] - tbc_n[cam] + G_pi + G_pj);

        A.block<9, 9>(0, 0) += Aj.transpose() * Aj;
        A.block<9, 1>(0, 9) += Aj.transpose() * bj;
        A.block<1, 9>(9, 0) += bj.transpose() * Aj;
        A.block<1, 1>(9, 9) += bj.transpose() * bj;
    }

    Eigen::Vector4d u;
    // random generation of u
    u.setRandom();
    u *= 100;

    Eigen::Matrix<double, 120, 120> M = Eigen::Matrix<double, 120, 120>::Zero();
    Eigen::Matrix<double, 27, 27> M0 = Eigen::Matrix<double, 27, 27>::Zero();
    constructM(A, u, M);

    M0 = M.block<27, 27>(0, 0) - M.block<27, 93>(0, 27) *
                                     M.block<93, 93>(27, 27).inverse() *
                                     M.block<93, 27>(27, 0);

    Eigen::EigenSolver<Eigen::Matrix<double, 27, 27>> es(M0);
    es.eigenvectors();

    auto V = es.eigenvectors();
    auto D = es.eigenvalues();
    vector<double> errors;
    vector<Eigen::Vector3d> r_cols;
    vector<int> r_indices;
    errors.reserve(27);
    r_cols.reserve(27);
    r_indices.reserve(27);
    int j = 0;
    for (int i = 0; i < 27; ++i) {
        if (abs(D(i).imag()) <= 1e-4) {
            r_indices.push_back(j++);

            auto v = V.col(i);
            Eigen::VectorXd v_real = v.real();

            v_real /= v_real(0);
            Eigen::Vector3d r;
            r << v_real(1), v_real(2), v_real(3);
            r_cols.push_back(r);

            Eigen::VectorXd r_dls;
            r_dls.resize(10);
            r_dls << r(0), r(1), r(2), r(0) * r(0), r(1) * r(1), r(2) * r(2),
                r(0) * r(1), r(0) * r(2), r(1) * r(2), 1;
            double error = r_dls.transpose() * A * r_dls;
            errors.push_back(error);
        }
    }

    sort(r_indices.begin(), r_indices.end(),
         [&errors](int a, int b) { return errors[a] < errors[b]; });

    // cout << "Real columns: " << r_cols.size() << endl;

    CGR2Rotation(r_cols[r_indices[0]], R_bw);

    t_bw.setZero();
    for (int i = 0; i < m; ++i) {
        Eigen::Vector3d p1 = lines_w_n[i].head<3>();
        Eigen::Vector3d p2 = lines_w_n[i].tail<3>();

        t_bw += G.block<3, 1>(0, 2 * i) * normals_b[i].transpose() *
                (R_bw * (p1 + p2) - 2 * tbc_n[lines_cam[i]]);
    }

    for (int j = 0; j < n; ++j) {
        Eigen::Vector3d p = points_w_n[j];
        t_bw +=
            G.block<3, 3>(0, 2 * m + 3 * j) * (R_bw * p - tbc_n[points_cam[j]]);
    }

    t_bw = t_bw / scale - R_bw * center;
}

void UPnPL::normalization(const vector<Eigen::Vector3d> &points_w,
                          const vector<Eigen::VectorXd> &lines_w,
                          vector<Eigen::Vector3d> &points_w_n,
                          vector<Eigen::VectorXd> &lines_w_n, double &scale,
                          Eigen::Vector3d &center, bool normalize) {
    int n = points_w.size(), m = lines_w.size();

    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();

    for (const auto &p : points_w) {
        A += Eigen::Matrix3d::Identity();
        b += p;
    }

    for (const auto &l : lines_w) {
        Eigen::Vector3d p1 = l.head<3>();
        Eigen::Vector3d p2 = l.tail<3>();
        Eigen::Vector3d d = p2 - p1;
        d.normalize();

        Eigen::Matrix3d Pi = Eigen::Matrix3d::Identity() - d * d.transpose();
        A += Pi;
        b += Pi * p1;
    }

    center = A.ldlt().solve(b);

    double dist = 0;
    for (const auto &p : points_w) {
        dist += (p - center).norm();
    }

    vector<double> line_dist;
    line_dist.reserve(m);

    for (const auto &l : lines_w) {
        Eigen::Vector3d p1 = l.head<3>();
        Eigen::Vector3d p2 = l.tail<3>();

        line_dist.emplace_back(pointLineDistance(center, p1, p2));
        dist += 2 * sqrt(2) * line_dist.back();
    }

    scale = sqrt(3) / (dist / (n + 2 * m));

    if (!normalize) {
        scale = 1;
        center.setZero();
    }

    points_w_n.resize(n);
    for (int i = 0; i < n; ++i) {
        points_w_n[i] = scale * (points_w[i] - center);
    }

    lines_w_n.resize(m);
    for (int i = 0; i < m; ++i) {
        Eigen::Vector3d p1 = lines_w[i].head<3>();
        Eigen::Vector3d p2 = lines_w[i].tail<3>();

        Eigen::Vector3d p1_n = scale * (p1 - center);
        Eigen::Vector3d p2_n = scale * (p2 - center);

        Eigen::Vector3d d = p2_n - p1_n;
        d.normalize();

        Eigen::Vector3d p_proj = p1_n + d * (-p1_n).dot(d);
        double dist_l = line_dist[i] * scale;

        lines_w_n[i].resize(6);
        lines_w_n[i].head<3>() = p_proj + d * dist_l;
        lines_w_n[i].tail<3>() = p_proj - d * dist_l;
    }
}

double UPnPL::pointLineDistance(const Eigen::Vector3d &x,
                                const Eigen::Vector3d &a,
                                const Eigen::Vector3d &b) {
    Eigen::Vector3d ab = b - a;
    Eigen::Vector3d ax = x - a;
    return (ab.cross(ax)).norm() / ab.norm();
}

Eigen::Matrix<double, 3, 9> UPnPL::phi(const Eigen::Vector3d &p) {
    Eigen::Matrix<double, 3, 9> J;
    J.setZero();
    double x = p(0), y = p(1), z = p(2);
    J << 0, 2 * z, -2 * y, x, -x, -x, 2 * y, 2 * z, 0, -2 * z, 0, 2 * x, -y, y,
        y, 2 * x, 0, 2 * z, 2 * y, -2 * x, 0, -z, -z, z, 0, 2 * x, 2 * y;
    return J;
}

void UPnPL::CGR2Rotation(const Eigen::Vector3d &s, Eigen::Matrix3d &R) {
    double norm_s = s.norm();
    if (norm_s < 1e-6) {
        R.setIdentity();
        return;
    }

    R = (1 - s.transpose() * s) * Eigen::Matrix3d::Identity() +
        2 * crossProductMatrix(s) + 2 * s * s.transpose();

    R /= 1 + s.transpose() * s;
}

} // namespace UPnPL
