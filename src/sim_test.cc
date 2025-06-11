#include <iostream>
#include <random>

#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

#include "UPnPL.h"

using namespace std;

using namespace Eigen;

struct Camera {
    Matrix3d R_bc; // camera-base 旋转
    Vector3d t_bc; // camera-base 平移
    double fx, fy, cx, cy;
    int width, height;
};

struct SimConfig {
    int num_cameras = 1;
    int num_points = 0;
    int num_lines = 50;
    double noise_std = 0.01;
    double outlier_ratio = 0.00000001;
    double outlier_std = 100;
};

Vector3d backProjectPixel(double u, double v, const Camera &camera) {
    double x = (u - camera.cx) / camera.fx;
    double y = (v - camera.cy) / camera.fy;
    return {x, y, 1.0};
}

Eigen::MatrixXd cvMatToEigen(const cv::Mat &mat) {
    Eigen::MatrixXd eigen_mat(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            eigen_mat(i, j) = mat.at<double>(i, j);
        }
    }
    return eigen_mat;
}

Eigen::Vector3d rotToCGR(const Eigen::Matrix3d &R) {
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d RpI = R + I;
    if (RpI.determinant() < 1e-6) {
        cerr << "Error: Rotation matrix is singular." << endl;
        return Eigen::Vector3d::Zero();
    }

    Eigen::Matrix3d A = (R - I) * RpI.inverse();

    Eigen::Vector3d caley;
    caley << A(2, 1), A(0, 2), A(1, 0);
    return caley;
}

int main() {
    SimConfig config;

    default_random_engine rng(42);

    Vector3d axis;
    axis << 0, 0, 1;
    axis.normalize();
    Matrix3d R_bw = AngleAxisd(M_PI / 6, axis).toRotationMatrix();
    Eigen::Vector3d r;
    r << 0, 0, 8;
    UPnPL::UPnPL::CGR2Rotation(r, R_bw);
    Vector3d t_bw(0, 1, 2);

    Vector3d axis_bc;
    axis_bc << 0, 1, 2;
    axis_bc.normalize();

    vector<Camera> cameras(config.num_cameras);
    vector<Matrix3d> Rbc(config.num_cameras);
    vector<Vector3d> tbc(config.num_cameras);
    for (int i = 0; i < config.num_cameras; ++i) {
        double angle = i * 2 * M_PI / config.num_cameras;
        angle = M_PI / 3;
        cameras[i].R_bc = AngleAxisd(angle, axis_bc).toRotationMatrix();
        // cameras[i].R_bc.setIdentity(); // Set to identity for simplicity
        cameras[i].t_bc = Vector3d(0.2 * cos(angle), 0, 0.2 * sin(angle));
        // cameras[i].t_bc.setZero(); // Set to zero for simplicity
        Rbc[i] = cameras[i].R_bc;
        tbc[i] = cameras[i].t_bc;
        cameras[i].fx = 1000;
        cameras[i].fy = 1000;
        cameras[i].cx = 320;
        cameras[i].cy = 240;
        cameras[i].width = 640;
        cameras[i].height = 480;
    }

    cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << cameras[0].fx, 0, cameras[0].cx, 0,
         cameras[0].fy, cameras[0].cy, 0, 0, 1);

    uniform_real_distribution<double> u_dist(0.0, cameras[0].width);
    uniform_real_distribution<double> v_dist(0.0, cameras[0].height);
    uniform_real_distribution<double> z_dist(0.5, 60.0);
    uniform_real_distribution<double> noise_dist(-config.noise_std,
                                                 config.noise_std);
    uniform_real_distribution<double> noise_outlier_dist(-config.outlier_std,
                                                         config.outlier_std);
    uniform_real_distribution<double> outlier_dist(0.0, 1.0);

    vector<Vector3d> points_w(config.num_points);
    vector<Vector3d> uv_c(config.num_points);
    vector<int> points_cam(config.num_points);

    // epnp vector
    vector<cv::Point3f> object_points;
    vector<cv::Point2f> image_points;

    for (int i = 0; i < config.num_points; ++i) {
        int cam_id = rng() % config.num_cameras;
        points_cam[i] = cam_id;
        const Camera &cam = cameras[cam_id];

        double u = u_dist(rng);
        double v = v_dist(rng);
        double z = z_dist(rng);

        double outlier = outlier_dist(rng);
        double noise = 0;
        if (outlier < config.outlier_ratio) {
            noise = noise_outlier_dist(rng);
        } else {
            noise = noise_dist(rng);
        }

        double noise_u = noise + u;
        double noise_v = noise + v;

        Vector3d p_c = backProjectPixel(u, v, cam);
        Vector3d p_c_noisy = backProjectPixel(noise_u, noise_v, cam);
        uv_c[i] = p_c_noisy;

        p_c *= z;
        Eigen::Vector3d p_b = cam.R_bc * p_c + cam.t_bc;
        Eigen::Vector3d p_w = R_bw.transpose() * (p_b - t_bw);
        points_w[i] = p_w;

        if (cam_id == 0) {
            object_points.emplace_back(p_w(0), p_w(1), p_w(2));
            image_points.emplace_back(noise_u, noise_v);
        }
    }

    for (int i = 0; i < config.num_points - object_points.size(); ++i) {
        const Camera &cam = cameras[0];

        double u = u_dist(rng);
        double v = v_dist(rng);
        double z = z_dist(rng);

        double outlier = outlier_dist(rng);
        double noise = 0;
        if (outlier < config.outlier_ratio) {
            noise = noise_outlier_dist(rng);
        } else {
            noise = noise_dist(rng);
        }
        double noise_u = noise + u;
        double noise_v = noise + v;

        Vector3d p_c = backProjectPixel(u, v, cam);

        p_c *= z;
        Eigen::Vector3d p_b = cam.R_bc * p_c + cam.t_bc;
        Eigen::Vector3d p_w = R_bw.transpose() * (p_b - t_bw);

        object_points.emplace_back(p_w(0), p_w(1), p_w(2));
        image_points.emplace_back(noise_u, noise_v);
    }

    vector<VectorXd> lines_w(config.num_lines);
    vector<Vector3d> normals_c(config.num_lines);
    vector<int> lines_cam(config.num_lines);

    for (int i = 0; i < config.num_lines; ++i) {
        int cam_id = rng() % config.num_cameras;
        lines_cam[i] = cam_id;
        const Camera &cam = cameras[cam_id];

        double u1 = u_dist(rng);
        double v1 = v_dist(rng);
        double z1 = z_dist(rng);

        double outlier1 = outlier_dist(rng);
        double noise_dist1 = 0;
        if (outlier1 < config.outlier_ratio) {
            noise_dist1 = noise_outlier_dist(rng);
        } else {
            noise_dist1 = noise_dist(rng);
        }
        double noise_u1 = noise_dist1 + u1;
        double noise_v1 = noise_dist1 + v1;

        double u2 = u_dist(rng);
        double v2 = v_dist(rng);
        double z2 = z_dist(rng);

        double outlier2 = outlier_dist(rng);
        double noise_dist2 = 0;
        if (outlier2 < config.outlier_ratio) {
            noise_dist2 = noise_outlier_dist(rng);
        } else {
            noise_dist2 = noise_dist(rng);
        }
        double noise_u2 = noise_dist2 + u2;
        double noise_v2 = noise_dist2 + v2;

        Vector3d p1_c = backProjectPixel(u1, v1, cam);
        Vector3d p2_c = backProjectPixel(u2, v2, cam);

        Vector3d p1_c_noisy = backProjectPixel(noise_u1, noise_v1, cam);
        Vector3d p2_c_noisy = backProjectPixel(noise_u2, noise_v2, cam);

        Vector3d normal_c_noisy =
            p1_c_noisy.cross(p2_c_noisy).normalized(); // Use noisy points

        normals_c[i] = normal_c_noisy;

        p1_c *= z1;
        p2_c *= z2;

        Vector3d p1_b = cam.R_bc * p1_c + cam.t_bc;
        Vector3d p2_b = cam.R_bc * p2_c + cam.t_bc;

        Vector3d p1_w = R_bw.transpose() * (p1_b - t_bw);
        Vector3d p2_w = R_bw.transpose() * (p2_b - t_bw);

        VectorXd line(6);
        line.head<3>() = p1_w;
        line.tail<3>() = p2_w;
        lines_w[i] = line;
    }

    Matrix3d R_bw_est;
    Vector3d t_bw_est;

    UPnPL::UPnPL upnpl_solver;
    upnpl_solver.solveUPnPL_EPnPL(points_w, lines_w, uv_c, normals_c,
                                  points_cam, lines_cam, Rbc, tbc, R_bw_est,
                                  t_bw_est);

    cout << "Estimated R_bw:\n" << R_bw_est << endl;
    cout << "Estimated t_bw:\n" << t_bw_est.transpose() << endl;

    cout << "Ground truth R_bw:\n" << R_bw << endl;
    cout << "Ground truth t_bw:\n" << t_bw.transpose() << endl;
    Eigen::Vector3d caley = rotToCGR(R_bw);
    cout << "Ground truth Caley vector:\n" << caley.transpose() << endl;

    // Convert R_bw_est and t_bw_est to Isometry3d
    Eigen::Isometry3d T_bw_est = Eigen::Isometry3d::Identity();
    T_bw_est.linear() = R_bw_est;
    T_bw_est.translation() = t_bw_est;

    Eigen::Isometry3d T_wb_est = T_bw_est.inverse();
    cout << "Estimated T_wb:\n" << T_wb_est.translation().transpose() << endl;

    cout << "Ground truth T_wb:\n"
         << (R_bw.transpose() * -t_bw).transpose() << endl;

    // Use OpenCV's EPnP to verify the results

    cv::Mat rvec, tvec;
    bool success =
        cv::solvePnP(object_points, image_points, camera_matrix, cv::Mat(),
                     rvec, tvec, false, cv::SOLVEPNP_EPNP);

    if (success) {
        Eigen::Isometry3d T_cw_epnp;
        cv::Mat R_epnp;
        cv::Rodrigues(rvec, R_epnp);
        T_cw_epnp.linear() = cvMatToEigen(R_epnp);
        T_cw_epnp.translation() = Eigen::Vector3d(
            tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

        Eigen::Isometry3d Tbc;
        Tbc.linear() = cameras[0].R_bc;
        Tbc.translation() = cameras[0].t_bc;

        Eigen::Isometry3d T_bw_epnp = Tbc * T_cw_epnp;
        cout << "EPnP Estimated R_bw:\n" << T_bw_epnp.linear() << endl;
        cout << "EPnP Estimated t_bw:\n"
             << T_bw_epnp.translation().transpose() << endl;
        cout << "EPnP Estimated T_wb:\n"
             << (T_bw_epnp.inverse().translation()).transpose() << endl;
    }

    return 0;
}
