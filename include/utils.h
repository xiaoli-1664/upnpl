#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

#include <opengv/absolute_pose/NoncentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/ccalib/omnidir.hpp>
#include <opencv2/opencv.hpp>

#include "UPnPL.h"

namespace fs = std::filesystem;

using namespace std;
using namespace opengv;
using namespace Eigen;

namespace utils {
struct Camera {
    int id;
    double fx, fy, cx, cy;
    string distortion_model;
    vector<double> distortion_coeffs;
    double xi;
    vector<double> projection_parameters;
    array<double, 2> image_size; // width, height
    Eigen::Isometry3d T_bc;

    cv::Mat map1, map2;

    bool need_rictified = false;

    void backProjectPixel(double u, double v, Eigen::Vector3d &point) const {
        point(0) = (u - cx) / fx;
        point(1) = (v - cy) / fy;
        point(2) = 1.0; // Assuming unit depth for back-projection
    }

    void getRectifiedFisheyeCamera() {
        cv::Size image_size_cv(image_size[0], image_size[1]);

        cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        cv::Mat D = cv::Mat(distortion_coeffs);

        cv::Mat R = cv::Mat::eye(3, 3, CV_64F);

        cv::Mat newK;

        double alpha = 1.0;

        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
            K, D, image_size_cv, R, newK, alpha);

        fx = newK.at<double>(0, 0);
        fy = newK.at<double>(1, 1);
        cx = newK.at<double>(0, 2);
        cy = newK.at<double>(1, 2);

        cv::fisheye::initUndistortRectifyMap(K, D, R, newK, image_size_cv,
                                             CV_32FC1, map1, map2);
        need_rictified = true;
    }

    void getRecitifiedPinholeCamera() {
        cv::Size image_size_cv(image_size[0], image_size[1]);
        cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        cv::Mat D = cv::Mat(distortion_coeffs);

        cv::Mat newK;
        double alpha = 1.0;

        newK = cv::getOptimalNewCameraMatrix(K, D, image_size_cv, alpha,
                                             image_size_cv);

        cv::initUndistortRectifyMap(K, distortion_coeffs, cv::Mat(), newK,
                                    image_size_cv, CV_32FC1, map1, map2);
        fx = newK.at<double>(0, 0);
        fy = newK.at<double>(1, 1);
        cx = newK.at<double>(0, 2);
        cy = newK.at<double>(1, 2);
        need_rictified = true;
    }

    void getRecitifiedMeiCamera() {
        cv::Size new_image_size(1400, 1400);
        cv::Mat K = (cv::Mat_<double>(3, 3) << projection_parameters[0], 0,
                     projection_parameters[2], 0, projection_parameters[1],
                     projection_parameters[3], 0, 0, 1);
        cv::Mat D = cv::Mat(distortion_coeffs);

        cv::Mat R = cv::Mat::eye(3, 3, CV_64F);

        cv::Mat newK = (cv::Mat_<double>(3, 3) << new_image_size.width / 4, 0,
                        new_image_size.width / 2, 0, new_image_size.height / 4,
                        new_image_size.height / 2, 0, 0, 1);
        fx = newK.at<double>(0, 0);
        fy = newK.at<double>(1, 1);
        cx = newK.at<double>(0, 2);
        cy = newK.at<double>(1, 2);
        need_rictified = true;
        cv::omnidir::initUndistortRectifyMap(K, D, xi, R, newK, new_image_size,
                                             CV_32FC1, map1, map2,
                                             cv::omnidir::RECTIFY_PERSPECTIVE);
    }
};

struct Point3D {
    Eigen::Vector3d pt;
    int id;
};

struct Line3D {
    Eigen::Vector3d p1, p2;
    int id;
};

void randomEndPoints(Eigen::VectorXd &lines, bool is_random) {
    if (!is_random)
        return;
    Eigen::Vector3d p1 = lines.head<3>();
    Eigen::Vector3d p2 = lines.tail<3>();
    Eigen::Vector3d v = (p2 - p1).normalized();
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(-10, 10);
    double t1 = dist(gen);
    double t2 = dist(gen);
    while (abs(t1 - t2) < 1e-6) {
        t2 = dist(gen);
    }
    Eigen::Vector3d pt_w_1 = p1 + t1 * v;
    Eigen::Vector3d pt_w_2 = p2 + t2 * v;
    lines.head<3>() = pt_w_1;
    lines.tail<3>() = pt_w_2;
}

class Simulator {
  public:
    Simulator(int num_cams, int n_points, int m_lines, double noise_std, double outlier_ratio=0.0)
        : num_cams_(num_cams), n_points_(n_points), m_lines_(m_lines),
          noise_std_(noise_std), outlier_ratio_(outlier_ratio) {
        rng_ = std::mt19937(rd_());
        noise_ = std::normal_distribution<>(0.0, noise_std);
        // generate random ground truth transformation
        uniform_real_distribution<double> angle_dist(0, 2 * M_PI);
        uniform_real_distribution<double> axis_dist(-1.0, 1.0);

        Eigen::Vector3d axis(axis_dist(rng_), axis_dist(rng_), axis_dist(rng_));
        axis.normalize();
        double angle = angle_dist(rng_);

        Eigen::Matrix3d R = Eigen::AngleAxisd(angle, axis).toRotationMatrix();

        uniform_real_distribution<double> trans_dist(-10.0, 10.0);
        Eigen::Vector3d translation;
        translation << trans_dist(rng_), trans_dist(rng_), trans_dist(rng_);
        T_bw_gt_.linear() = R;
        T_bw_gt_.translation() = translation;
    }

    void setupCameras();
    void generateScene();
    void generateData(vector<Eigen::Vector3d> &points_w,
                      vector<Eigen::VectorXd> &lines_w,
                      vector<Eigen::Vector3d> &uv_c,
                      vector<Eigen::Vector3d> &normals_c,
                      vector<Eigen::VectorXd> &lines_c, vector<int> &points_cam,
                      vector<int> &lines_cam);
    void clearScene() {
        for (int i = 0; i < 3; ++i) {
            cube_points_[i].clear();
            cube_lines_[i].clear();
        }
    }

  public:
    int num_cams_, n_points_, m_lines_;
    double noise_std_, outlier_ratio_;
    std::vector<Camera> cameras_;
    std::vector<Point3D> cube_points_[3];
    std::vector<Line3D> cube_lines_[3];
    Eigen::Isometry3d T_bw_gt_ = Eigen::Isometry3d::Identity();
    std::random_device rd_;
    std::mt19937 rng_;
    std::normal_distribution<> noise_;

    void generateCube(const Eigen::Vector3d &center, double size, int cube_id);
    Eigen::Vector2d projectWithNoise(const Eigen::Vector3d &pt,
                                     const Camera &cam);
    Eigen::Vector3d Unproject(const Eigen::Vector2d &uv, const Camera &cam);
};

void Simulator::setupCameras() {
    cameras_.resize(num_cams_);
    for (int i = 0; i < num_cams_; ++i) {
        Camera &cam = cameras_[i];
        cam.distortion_coeffs.resize(4, 0.0);
        cam.id = i;
        cam.fx = 450;
        cam.fy = 450;
        cam.cx = 376;
        cam.cy = 240;
        cam.image_size[0] = 752;
        cam.image_size[1] = 480;

        cam.T_bc = Eigen::Isometry3d::Identity();
        if (i == 1)
            cam.T_bc(0, 3) = 0.2;
        if (i == 2) {
            cam.T_bc(0, 3) = -0.2;
            cam.T_bc(2, 3) = -0.2;
            // rotate around y-axis
            cam.T_bc.linear() =
                Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitY())
                    .toRotationMatrix();
        }
        if (i == 3) {
            cam.T_bc.linear() =
                Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitY())
                    .toRotationMatrix();
            cam.T_bc(0, 3) = 0.4;
            cam.T_bc(2, 3) = -0.2;
        }
    }
}

void Simulator::generateCube(const Eigen::Vector3d &center, double size,
                             int cube_id) {
    for (int i = 0; i < n_points_; ++i) {
        double x = center.x() + size * (rand() / (double)RAND_MAX - 0.5);
        double y = center.y() + size * (rand() / (double)RAND_MAX - 0.5);
        double z = center.z() + size * (rand() / (double)RAND_MAX - 0.5);
        cube_points_[cube_id].push_back({Eigen::Vector3d(x, y, z), i});
    }
    for (int i = 0; i < m_lines_; ++i) {
        Eigen::Vector3d p1(
            center.x() + size * (rand() / (double)RAND_MAX - 0.5),
            center.y() + size * (rand() / (double)RAND_MAX - 0.5),
            center.z() + size * (rand() / (double)RAND_MAX - 0.5));
        Eigen::Vector3d p2(
            center.x() + size * (rand() / (double)RAND_MAX - 0.5),
            center.y() + size * (rand() / (double)RAND_MAX - 0.5),
            center.z() + size * (rand() / (double)RAND_MAX - 0.5));
        cube_lines_[cube_id].push_back({p1, p2, i}); // Store line with id
    }
}

void Simulator::generateScene() {
    int width = 4;
    int depth = 8;
    generateCube(Eigen::Vector3d(0.1, 0, depth), width, 0);
    if (num_cams_ > 2) {
        generateCube(Eigen::Vector3d(-depth - 0.2, 0, -0.2), width, 1);
    }
    if (num_cams_ > 3) {
        generateCube(Eigen::Vector3d(depth + 0.4, 0, -0.2), width, 2);
    }
}

Eigen::Vector2d Simulator::projectWithNoise(const Eigen::Vector3d &pt_b,
                                            const Camera &cam) {
    Eigen::Vector3d pt_c = cam.T_bc.inverse() * pt_b;

    Eigen::Vector2d uv;
    uv(0) = cam.fx * pt_c(0) / pt_c(2) + cam.cx;
    uv(1) = cam.fy * pt_c(1) / pt_c(2) + cam.cy;
    // Add noise
    uv(0) += noise_(rng_);
    uv(1) += noise_(rng_);
    return uv;
}

void Simulator::generateData(vector<Eigen::Vector3d> &points_w,
                             vector<Eigen::VectorXd> &lines_w,
                             vector<Eigen::Vector3d> &uv_c,
                             vector<Eigen::Vector3d> &normals_c,
                             vector<Eigen::VectorXd> &lines_c,
                             vector<int> &points_cam, vector<int> &lines_cam) {
    points_w.clear();
    lines_w.clear();
    uv_c.clear();
    normals_c.clear();
    points_cam.clear();
    lines_cam.clear();

    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

    for (int i = 0; i < num_cams_; ++i) {
        const Camera &cam = cameras_[i];
        std::uniform_real_distribution<double> rand_u(0, cam.image_size[0]);
        std::uniform_real_distribution<double> rand_v(0, cam.image_size[1]);
        int cube_id = 0;
        if (i < 2) {
            cube_id = 0; // First cube for first two cameras
        } else if (i == 2) {
            cube_id = 1; // Second cube for third camera
        } else if (i == 3) {
            cube_id = 2; // Third cube for fourth camera
        }
        for (const auto &pt_b : cube_points_[cube_id]) {
            Eigen::Vector2d uv;
            if (prob_dist(rng_) < outlier_ratio_) {
                uv << rand_u(rng_), rand_v(rng_);
            }
            else {
                uv = projectWithNoise(pt_b.pt, cam);
            }
            if (uv(0) < 0 || uv(0) >= cam.image_size[0] || uv(1) < 0 ||
                uv(1) >= cam.image_size[1]) {
                continue; // Skip points outside image bounds
            }
            Eigen::Vector3d pt_w = T_bw_gt_.inverse() * pt_b.pt;
            points_w.push_back(pt_w);
            uv_c.push_back(Unproject(uv, cam));
            points_cam.push_back(i);
        }
        for (const auto &line_b : cube_lines_[cube_id]) {
            // if (i != 2)
            //     continue;
            Eigen::Vector2d uv1, uv2;
            if (prob_dist(rng_) < outlier_ratio_) {
                uv1 << rand_u(rng_), rand_v(rng_);
                uv2 << rand_u(rng_), rand_v(rng_);
            } else {
                uv1 = projectWithNoise(line_b.p1, cam);
                uv2 = projectWithNoise(line_b.p2, cam);
            }
            if (uv1(0) < 0 || uv1(0) >= cam.image_size[0] || uv1(1) < 0 ||
                uv1(1) >= cam.image_size[1] || uv2(0) < 0 ||
                uv2(0) >= cam.image_size[0] || uv2(1) < 0 ||
                uv2(1) >= cam.image_size[1]) {
                continue; // Skip lines with endpoints outside image bounds
            }
            Eigen::Vector3d pt_w1 = T_bw_gt_.inverse() * line_b.p1;
            Eigen::Vector3d pt_w2 = T_bw_gt_.inverse() * line_b.p2;
            Eigen::VectorXd line;
            line.resize(6);
            line.head<3>() = pt_w1;
            line.tail<3>() = pt_w2;
            randomEndPoints(line, true);
            lines_w.push_back(line);

            Eigen::Vector3d pt_c1 = Unproject(uv1, cam);
            Eigen::Vector3d pt_c2 = Unproject(uv2, cam);
            Eigen::VectorXd line_c(6);
            line_c.head<3>() = pt_c1;
            line_c.tail<3>() = pt_c2;
            Eigen::Vector3d normal_c = (pt_c1.cross(pt_c2)).normalized();
            lines_c.push_back(line_c);
            normals_c.push_back(normal_c);
            lines_cam.push_back(i);
        }
    }
}

Eigen::Vector3d Simulator::Unproject(const Eigen::Vector2d &uv,
                                     const Camera &cam) {
    Eigen::Vector3d pt_c;
    pt_c(0) = (uv(0) - cam.cx) / cam.fx;
    pt_c(1) = (uv(1) - cam.cy) / cam.fy;
    pt_c(2) = 1.0; // Assume unit depth for simplicity
    return pt_c;
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

Eigen::Isometry3d myUPnPL(const vector<Eigen::Vector3d> &points_w_tmp,
                          const vector<Eigen::VectorXd> &lines_w_tmp,
                          const vector<Eigen::Vector3d> &uv_c_tmp,
                          const vector<Eigen::VectorXd> &lines_c_tmp,
                          const vector<int> &points_cam_tmp,
                          const vector<int> &lines_cam_tmp,
                          const vector<Camera> &cameras, int num_cam,
                          double &used_time) {
    UPnPL::UPnPL upnpl_solver(true);
    Eigen::Matrix3d R_bw;
    Eigen::Vector3d t_bw;

    vector<Eigen::Matrix3d> R_bc;
    vector<Eigen::Vector3d> t_bc;

    for (int i = 0; i < num_cam; ++i) {
        const Camera &cam = cameras[i];
        R_bc.push_back(cam.T_bc.linear());
        t_bc.push_back(cam.T_bc.translation());
    }

    vector<Eigen::Vector3d> points_w;
    vector<Eigen::VectorXd> lines_w;
    vector<Eigen::Vector3d> uv_c;
    vector<Eigen::VectorXd> lines_c;
    vector<int> points_cam;
    vector<int> lines_cam;

    for (int i = 0; i < points_w_tmp.size(); ++i) {
        if (points_cam_tmp[i] < num_cam) {
            points_w.push_back(points_w_tmp[i]);
            uv_c.push_back(uv_c_tmp[i]);
            points_cam.push_back(points_cam_tmp[i]);
        }
    }

    for (int i = 0; i < lines_w_tmp.size(); ++i) {
        if (lines_cam_tmp[i] < num_cam) {
            lines_w.push_back(lines_w_tmp[i]);
            lines_c.push_back(lines_c_tmp[i]);
            lines_cam.push_back(lines_cam_tmp[i]);
        }
    }

    Eigen::Isometry3d T_bw = Eigen::Isometry3d::Identity();
    auto start = chrono::high_resolution_clock::now();
    if (points_w.size() + lines_w.size() < 4) {
        used_time = 0.0;
        return T_bw;
    }
    upnpl_solver.solveMain(points_w, lines_w, uv_c, lines_c, points_cam,
                           lines_cam, R_bc, t_bc, R_bw, t_bw);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> elapsed = end - start;
    used_time = elapsed.count();
    T_bw.linear() = R_bw;
    T_bw.translation() = t_bw;
    return T_bw;
}

Eigen::Isometry3d cv_EPnP(const vector<Eigen::Vector3d> &points_w,
                          const vector<Eigen::Vector3d> &uv_c,
                          const vector<int> &points_cam,
                          const vector<Camera> &cameras, double &used_time) {
    // Prepare OpenCV data structures
    vector<cv::Point3f> object_points;
    vector<cv::Point2f> image_points;

    int cam = 0;
    for (size_t i = 0; i < points_w.size(); ++i) {
        if (points_cam[i] == cam) {
            object_points.emplace_back(points_w[i](0), points_w[i](1),
                                       points_w[i](2));
            double u =
                cameras[cam].fx * uv_c[i](0) / uv_c[i](2) + cameras[cam].cx;
            double v =
                cameras[cam].fy * uv_c[i](1) / uv_c[i](2) + cameras[cam].cy;
            // image_points.emplace_back(u, v);
            image_points.emplace_back(uv_c[i](0) / uv_c[i](2),
                                      uv_c[i](1) / uv_c[i](2));
        }
    }

    cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << cameras[cam].fx, 0, cameras[cam].cx, 0,
         cameras[cam].fy, cameras[cam].cy, 0, 0, 1);
    camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    Eigen::Isometry3d T_bw = Eigen::Isometry3d::Identity();

    if (object_points.size() < 4) {
        used_time = 0.0;
        return T_bw; // Not enough points for EPnP
    }

    cv::Mat rvec, tvec;
    auto start = chrono::high_resolution_clock::now();
    try {
        bool success =
            cv::solvePnP(object_points, image_points, camera_matrix,
                         dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_EPNP);
            // cv::solvePnPRansac(object_points, image_points, camera_matrix,
            //                  dist_coeffs, rvec, tvec, false, 100, 2.0, 0.99,
            //                  cv::noArray(), cv::SOLVEPNP_AP3P);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, std::milli> elapsed = end - start;
        used_time = elapsed.count();

        if (success) {
            Eigen::Isometry3d T_cw_epnp;
            cv::Mat R_epnp;
            cv::Rodrigues(rvec, R_epnp);
            T_cw_epnp.linear() = cvMatToEigen(R_epnp);
            T_cw_epnp.translation() = Eigen::Vector3d(
                tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
            Eigen::Isometry3d T_bc = cameras[cam].T_bc;

            T_bw = T_bc * T_cw_epnp;
        }
    } catch (const cv::Exception &e) {
        used_time = 0.0;
        return T_bw; // Handle exceptions gracefully
    }

    return T_bw;
}

Eigen::Isometry3d opengv_UPnP(const vector<Eigen::Vector3d> &points_w,
                              const vector<Eigen::Vector3d> &uv_c,
                              const vector<int> &points_cam,
                              const vector<Camera> &cameras,
                              double &used_time) {
    bearingVectors_t bearingVectors;
    points_t points;
    vector<int> cam_correspondences;
    vector<int> indices;
    translations_t translations;
    rotations_t rotations;

    for (int i = 0; i < points_w.size(); ++i) {
        bearingVectors.push_back(uv_c[i].normalized());
        points.push_back(points_w[i]);
        cam_correspondences.push_back(points_cam[i]);

        indices.push_back(i);
    }

    for (auto &cam : cameras) {
        translations.push_back(cam.T_bc.translation());
        rotations.push_back(cam.T_bc.linear());
    }

    try {
        auto start = chrono::high_resolution_clock::now();
        absolute_pose::NoncentralAbsoluteAdapter adapter(
            bearingVectors, cam_correspondences, points, translations,
            rotations);

        transformations_t T = absolute_pose::upnp(adapter, indices);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, std::milli> elapsed = end - start;
        used_time = elapsed.count();

        Eigen::Matrix<double, 3, 4> T_matrix = T[0];
        Eigen::Isometry3d T_bw = Eigen::Isometry3d::Identity();
        T_bw.linear() = T_matrix.block<3, 3>(0, 0);
        T_bw.translation() = T_matrix.block<3, 1>(0, 3);
        return T_bw.inverse();
    } catch (const std::exception &e) {
        used_time = 0.0;
        return Eigen::Isometry3d::Identity(); // Return identity if error occurs
    }
}

Eigen::Isometry3d opengv_GPnP(const vector<Eigen::Vector3d> &points_w,
                              const vector<Eigen::Vector3d> &uv_c,
                              const vector<int> &points_cam,
                              const vector<Camera> &cameras,
                              double &used_time) {
    bearingVectors_t bearingVectors;
    points_t points;
    vector<int> cam_correspondences;
    vector<int> indices;
    translations_t translations;
    rotations_t rotations;

    for (int i = 0; i < points_w.size(); ++i) {
        bearingVectors.push_back(uv_c[i].normalized());
        points.push_back(points_w[i]);
        cam_correspondences.push_back(points_cam[i]);

        indices.push_back(i);
    }

    for (auto &cam : cameras) {
        translations.push_back(cam.T_bc.translation());
        rotations.push_back(cam.T_bc.linear());
    }

    try {
        auto start = chrono::high_resolution_clock::now();
        absolute_pose::NoncentralAbsoluteAdapter adapter(
            bearingVectors, cam_correspondences, points, translations,
            rotations);

        if (indices.size() < 4) {
            used_time = 0.0;
            return Eigen::Isometry3d::Identity(); // Not enough points for GPnP
        }

        transformation_t T = absolute_pose::gpnp(adapter, indices);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, std::milli> elapsed = end - start;
        used_time = elapsed.count();

        Eigen::Matrix<double, 3, 4> T_matrix = T;
        Eigen::Isometry3d T_bw = Eigen::Isometry3d::Identity();
        T_bw.linear() = T_matrix.block<3, 3>(0, 0);
        T_bw.translation() = T_matrix.block<3, 1>(0, 3);
        return T_bw.inverse();
    } catch (const std::exception &e) {
        used_time = 0.0;
        return Eigen::Isometry3d::Identity(); // Return identity if error occurs
    }
}

void saveDataForMatlab(
    const vector<Eigen::Vector3d> &points_w, const vector<double> &points_sigma,
    const vector<Eigen::VectorXd> &lines_w, const vector<double> &lines_sigma,
    const vector<Eigen::Vector3d> &uv_c, const vector<Eigen::VectorXd> &lines_c,
    const vector<int> &points_cam, const vector<int> &lines_cam,
    const string &filename, const double time) {
    int cam = 0;
    fs::path output_path(filename);
    if (!fs::exists(output_path.parent_path())) {
        fs::create_directories(output_path.parent_path());
    }

    ofstream ofs(filename);
    if (!ofs.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }

    vector<Eigen::Vector3d> points_w_cam;
    vector<int> points_level_cam;
    vector<Eigen::VectorXd> lines_w_cam;
    vector<int> lines_level_cam;
    vector<Eigen::Vector3d> uv_c_cam;
    vector<Eigen::VectorXd> lines_c_cam;

    for (int i = 0; i < points_w.size(); ++i) {
        if (points_cam[i] == cam) {
            points_w_cam.push_back(points_w[i]);
            points_level_cam.push_back(points_sigma[i]);
            Eigen::Vector3d uv = uv_c[i];
            uv /= uv_c[i](2); // Normalize by depth
            uv_c_cam.push_back(uv);
        }
    }
    for (int i = 0; i < lines_w.size(); ++i) {
        if (lines_cam[i] == cam) {
            lines_w_cam.push_back(lines_w[i]);
            lines_level_cam.push_back(lines_sigma[i]);
            lines_c_cam.push_back(lines_c[i]);
        }
    }

    ofs << fixed << setprecision(10);

    ofs << time << endl; // Save timestamp

    ofs << points_w_cam.size() << endl;

    for (int i = 0; i < points_w_cam.size(); ++i) {
        ofs << points_w_cam[i].transpose() << " " << uv_c_cam[i].transpose()
            << " " << points_sigma[i] << endl;
    }

    ofs << lines_w_cam.size() << endl;

    for (int i = 0; i < lines_w_cam.size(); ++i) {
        ofs << lines_w_cam[i].transpose() << " " << lines_c_cam[i].transpose()
            << " " << lines_sigma[i] << endl;
    }
}

void saveEurocTraejectory(const string &output_file,
                          const vector<Eigen::Isometry3d> &poses,
                          const vector<double> &times) {
    if (poses.size() != times.size()) {
        cerr << "Error: poses and times vectors must have the same size."
             << endl;
        return;
    }

    fs::path output_path(output_file);

    if (!fs::exists(output_path.parent_path())) {
        fs::create_directories(output_path.parent_path());
    }

    ofstream ofs(output_file);
    if (!ofs.is_open()) {
        cerr << "Failed to open output file: " << output_file << endl;
        return;
    }

    ofs << fixed << setprecision(10);

    for (size_t i = 0; i < poses.size(); ++i) {
        const Eigen::Isometry3d &pose = poses[i];
        double timestamp = times[i];

        Eigen::Vector3d t = pose.translation();

        Eigen::Quaterniond q(pose.linear());

        ofs << (long long)(timestamp * 1e9) << "," << t.x() << "," << t.y()
            << "," << t.z() << "," << q.x() << "," << q.y() << "," << q.z()
            << "," << q.w() << "\n";
    }

    ofs.close();

    cout << "Trajectory saved to " << output_file << endl;
}

double computePointError(const Vector3d& P_w, const Vector3d& uv_obs, 
                         const Isometry3d& T_bc, const Isometry3d& T_wb) {
    Isometry3d T_cw = (T_wb * T_bc).inverse();
    Vector3d P_c = T_cw * P_w;
    
    if (P_c.z() <= 0) return 1e9; 
    
    Vector2d uv_proj = P_c.head<2>() / P_c.z();
    return (uv_proj - uv_obs.head<2>()).norm();
}

double computeLineError(const VectorXd& L_w, const VectorXd& L_c,
                        const Isometry3d& T_bc, const Isometry3d& T_wb) {
    Isometry3d T_cw = (T_wb * T_bc).inverse();
    Vector3d p_c1 = T_cw * Vector3d(L_w.head<3>());
    Vector3d p_c2 = T_cw * Vector3d(L_w.tail<3>());
    if (p_c1.z() <= 0 || p_c2.z() <= 0) return 1e9;

    Vector2d proj1 = p_c1.head<2>() / p_c1.z();
    Vector2d proj2 = p_c2.head<2>() / p_c2.z();

    Vector3d n = (Vector3d(L_c.head<3>()).cross(Vector3d(L_c.tail<3>()))).normalized();

    double scale = sqrt(n.x() * n.x() + n.y() * n.y());
    if (scale < 1e-6) return 1e9;
    
    double d1 = std::abs(Vector3d(proj1.x(), proj1.y(), 1.0).dot(n)) / scale;
    double d2 = std::abs(Vector3d(proj2.x(), proj2.y(), 1.0).dot(n)) / scale;
    
    return (d1 + d2) / 2.0;
}

Eigen::Isometry3d myUPnPL_RANSAC(const vector<Eigen::Vector3d> &points_w_tmp,
                                 const vector<Eigen::VectorXd> &lines_w_tmp,
                                 const vector<Eigen::Vector3d> &uv_c_tmp,
                                 const vector<Eigen::VectorXd> &lines_c_tmp,
                                 const vector<int> &points_cam_tmp,
                                 const vector<int> &lines_cam_tmp,
                                 const vector<Camera> &cameras, int num_cam,
                                 int n_min, int m_min, int max_iters, double threshold,
                                 double &used_time) {
    
    vector<Eigen::Matrix3d> R_bc;
    vector<Eigen::Vector3d> t_bc;
    for (int i = 0; i < num_cam; ++i) {
        R_bc.push_back(cameras[i].T_bc.linear());
        t_bc.push_back(cameras[i].T_bc.translation());
    }

    vector<Eigen::Vector3d> p_w_all, uv_c_all;
    vector<Eigen::VectorXd> l_w_all, l_c_all;
    vector<int> p_cam_all, l_cam_all;
    for (size_t i = 0; i < points_w_tmp.size(); ++i) {
        if (points_cam_tmp[i] < num_cam) {
            p_w_all.push_back(points_w_tmp[i]);
            uv_c_all.push_back(uv_c_tmp[i]);
            p_cam_all.push_back(points_cam_tmp[i]);
        }
    }
    for (size_t i = 0; i < lines_w_tmp.size(); ++i) {
        if (lines_cam_tmp[i] < num_cam) {
            l_w_all.push_back(lines_w_tmp[i]);
            l_c_all.push_back(lines_c_tmp[i]);
            l_cam_all.push_back(lines_cam_tmp[i]);
        }
    }

    Eigen::Isometry3d best_T = Eigen::Isometry3d::Identity();
    int max_inliers = -1;
    vector<int> best_p_inliers, best_l_inliers;
    
    std::random_device rd;
    std::mt19937 g(rd());
    UPnPL::UPnPL upnpl_solver(true);

    auto start_time = chrono::high_resolution_clock::now();
    for (int iter = 0; iter < max_iters; ++iter) {
        vector<Eigen::Vector3d> p_w_s, uv_c_s;
        vector<Eigen::VectorXd> l_w_s, l_c_s;
        vector<int> p_cam_s, l_cam_s;

        if (n_min > 0 && p_w_all.size() >= n_min) {
            vector<int> idx(p_w_all.size()); iota(idx.begin(), idx.end(), 0);
            shuffle(idx.begin(), idx.end(), g);
            for(int j=0; j<n_min; ++j) {
                p_w_s.push_back(p_w_all[idx[j]]); uv_c_s.push_back(uv_c_all[idx[j]]); p_cam_s.push_back(p_cam_all[idx[j]]);
            }
        }
        if (m_min > 0 && l_w_all.size() >= m_min) {
            vector<int> idx(l_w_all.size()); iota(idx.begin(), idx.end(), 0);
            shuffle(idx.begin(), idx.end(), g);
            for(int j=0; j<m_min; ++j) {
                l_w_s.push_back(l_w_all[idx[j]]); l_c_s.push_back(l_c_all[idx[j]]); l_cam_s.push_back(l_cam_all[idx[j]]);
            }
        }

        Eigen::Matrix3d R_tmp; Eigen::Vector3d t_tmp;
        upnpl_solver.solveMain(p_w_s, l_w_s, uv_c_s, l_c_s, p_cam_s, l_cam_s, R_bc, t_bc, R_tmp, t_tmp);
        
        Eigen::Isometry3d T_curr = Eigen::Isometry3d::Identity();
        T_curr.linear() = R_tmp; T_curr.translation() = t_tmp;

        vector<int> curr_p_inliers, curr_l_inliers;
        for (int i = 0; i < p_w_all.size(); ++i){
            double pe = computePointError(p_w_all[i], uv_c_all[i], cameras[p_cam_all[i]].T_bc, T_curr.inverse());
            // cout << "point error: " << pe << endl;
            if (pe < threshold)
                curr_p_inliers.push_back(i);

        }
        for (int i = 0; i < l_w_all.size(); ++i) {
            double le = computeLineError(l_w_all[i], l_c_all[i], cameras[l_cam_all[i]].T_bc, T_curr.inverse());
            // cout << "line error: " << le << endl;
            if (le < threshold)
                curr_l_inliers.push_back(i);
        }

        int total_inliers = curr_p_inliers.size() + curr_l_inliers.size();
        if (total_inliers > max_inliers) {
            max_inliers = total_inliers;
            best_T = T_curr;
            best_p_inliers = curr_p_inliers;
            best_l_inliers = curr_l_inliers;
        }
        // cout << "points number: " << points_w_tmp.size() << ", line number: " << lines_w_tmp.size() << endl;
        // cout << "RANSAC Iteration " << iter + 1 << ": Inliers = " << total_inliers << endl;
        // cout << "point inliers: " << curr_p_inliers.size() << ", line inliers: " << curr_l_inliers.size() << endl;
    }

    if (max_inliers >= 4) {
        vector<Eigen::Vector3d> p_w_r, uv_c_r;
        vector<Eigen::VectorXd> l_w_r, l_c_r;
        vector<int> p_cam_r, l_cam_r;
        for(int idx : best_p_inliers) {
            p_w_r.push_back(p_w_all[idx]); uv_c_r.push_back(uv_c_all[idx]); p_cam_r.push_back(p_cam_all[idx]);
        }
        for(int idx : best_l_inliers) {
            l_w_r.push_back(l_w_all[idx]); l_c_r.push_back(l_c_all[idx]); l_cam_r.push_back(l_cam_all[idx]);
        }
        Eigen::Matrix3d R_ref; Eigen::Vector3d t_ref;
        upnpl_solver.solveMain(p_w_r, l_w_r, uv_c_r, l_c_r, p_cam_r, l_cam_r, R_bc, t_bc, R_ref, t_ref);
        best_T.linear() = R_ref; best_T.translation() = t_ref;
    }

    auto end_time = chrono::high_resolution_clock::now();
    used_time = chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return best_T;
}



}; // namespace utils
