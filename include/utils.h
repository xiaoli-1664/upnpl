#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

#include <opengv/absolute_pose/NoncentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

#include "UPnPL.h"

namespace fs = std::filesystem;

using namespace std;
using namespace opengv;
using namespace Eigen;

namespace utils {
struct Camera {
    int id;
    bool isStereo;
    cv::Matx33d K;
    Eigen::Isometry3d T_bc;
    cv::Size image_size;
};

struct Point3D {
    Eigen::Vector3d pt;
    int id;
};

struct Line3D {
    Eigen::Vector3d p1, p2;
    int id;
};

class Simulator {
  public:
    Simulator(int num_cams, int n_points, int m_lines, double noise_std)
        : num_cams_(num_cams), n_points_(n_points), m_lines_(m_lines),
          noise_std_(noise_std) {
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
    double noise_std_;
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
        cam.id = i;
        cam.isStereo = (i < 2);
        cam.K = cv::Matx33d(450, 0, 376, 0, 450, 240, 0, 0, 1);
        cam.image_size = cv::Size(752, 480);

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
    uv(0) = cam.K(0, 0) * pt_c(0) / pt_c(2) + cam.K(0, 2);
    uv(1) = cam.K(1, 1) * pt_c(1) / pt_c(2) + cam.K(1, 2);
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
    for (int i = 0; i < num_cams_; ++i) {
        const Camera &cam = cameras_[i];
        int cube_id = 0;
        if (i < 2) {
            cube_id = 0; // First cube for first two cameras
        } else if (i == 2) {
            cube_id = 1; // Second cube for third camera
        } else if (i == 3) {
            cube_id = 2; // Third cube for fourth camera
        }
        for (const auto &pt_b : cube_points_[cube_id]) {
            // if (i != 2)
            //     continue;
            Eigen::Vector2d uv = projectWithNoise(pt_b.pt, cam);
            if (uv(0) < 0 || uv(0) >= cam.image_size.width || uv(1) < 0 ||
                uv(1) >= cam.image_size.height) {
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
            Eigen::Vector2d uv1 = projectWithNoise(line_b.p1, cam);
            Eigen::Vector2d uv2 = projectWithNoise(line_b.p2, cam);
            if (uv1(0) < 0 || uv1(0) >= cam.image_size.width || uv1(1) < 0 ||
                uv1(1) >= cam.image_size.height || uv2(0) < 0 ||
                uv2(0) >= cam.image_size.width || uv2(1) < 0 ||
                uv2(1) >= cam.image_size.height) {
                continue; // Skip lines with endpoints outside image bounds
            }
            Eigen::Vector3d pt_w1 = T_bw_gt_.inverse() * line_b.p1;
            Eigen::Vector3d pt_w2 = T_bw_gt_.inverse() * line_b.p2;
            Eigen::VectorXd line(6);
            line.head<3>() = pt_w1;
            line.tail<3>() = pt_w2;
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
    pt_c(0) = (uv(0) - cam.K(0, 2)) / cam.K(0, 0);
    pt_c(1) = (uv(1) - cam.K(1, 2)) / cam.K(1, 1);
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
                          const vector<Camera> &cameras, int num_cam) {
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

    upnpl_solver.solveMain(points_w, lines_w, uv_c, lines_c, points_cam,
                           lines_cam, R_bc, t_bc, R_bw, t_bw);
    Eigen::Isometry3d T_bw = Eigen::Isometry3d::Identity();
    T_bw.linear() = R_bw;
    T_bw.translation() = t_bw;
    return T_bw;
}

Eigen::Isometry3d cv_EPnP(const vector<Eigen::Vector3d> &points_w,
                          const vector<Eigen::VectorXd> &lines_w,
                          const vector<Eigen::Vector3d> &uv_c,
                          const vector<Eigen::Vector3d> &normals_c,
                          const vector<int> &points_cam,
                          const vector<int> &lines_cam,
                          const vector<Camera> &cameras) {
    // Prepare OpenCV data structures
    vector<cv::Point3f> object_points;
    vector<cv::Point2f> image_points;

    int cam = 0;
    for (size_t i = 0; i < points_w.size(); ++i) {
        if (points_cam[i] == cam) {
            object_points.emplace_back(points_w[i](0), points_w[i](1),
                                       points_w[i](2));
            double u = cameras[cam].K(0, 0) * uv_c[i](0) / uv_c[i](2) +
                       cameras[cam].K(0, 2);
            double v = cameras[cam].K(1, 1) * uv_c[i](1) / uv_c[i](2) +
                       cameras[cam].K(1, 2);
            // image_points.emplace_back(u, v);
            image_points.emplace_back(uv_c[i](0) / uv_c[i](2),
                                      uv_c[i](1) / uv_c[i](2));
        }
    }

    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << cameras[cam].K(0, 0), 0,
                             cameras[cam].K(0, 2), 0, cameras[cam].K(1, 1),
                             cameras[cam].K(1, 2), 0, 0, 1);
    camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    Eigen::Isometry3d T_bw = Eigen::Isometry3d::Identity();

    if (object_points.size() < 4) {
        return T_bw; // Not enough points for EPnP
    }

    cv::Mat rvec, tvec;
    bool success =
        cv::solvePnP(object_points, image_points, camera_matrix, dist_coeffs,
                     rvec, tvec, false, cv::SOLVEPNP_EPNP);

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

    return T_bw;
}

Eigen::Isometry3d opengv_UPnP(const vector<Eigen::Vector3d> &points_w,
                              const vector<Eigen::VectorXd> &lines_w,
                              const vector<Eigen::Vector3d> &uv_c,
                              const vector<Eigen::Vector3d> &normals_c,
                              const vector<int> &points_cam,
                              const vector<int> &lines_cam,
                              const vector<Camera> &cameras) {
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

    absolute_pose::NoncentralAbsoluteAdapter adapter(
        bearingVectors, cam_correspondences, points, translations, rotations);

    transformations_t T = absolute_pose::upnp(adapter, indices);
    Eigen::Matrix<double, 3, 4> T_matrix = T[0];
    Eigen::Isometry3d T_bw = Eigen::Isometry3d::Identity();
    T_bw.linear() = T_matrix.block<3, 3>(0, 0);
    T_bw.translation() = T_matrix.block<3, 1>(0, 3);
    return T_bw.inverse();
}

void saveDataForMatlab(const vector<Eigen::Vector3d> &points_w,
                       const vector<Eigen::VectorXd> &lines_w,
                       const vector<Eigen::Vector3d> &uv_c,
                       const vector<Eigen::VectorXd> &lines_c,
                       const vector<int> &points_cam,
                       const vector<int> &lines_cam, const string &filename,
                       const double time) {
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
    vector<Eigen::VectorXd> lines_w_cam;
    vector<Eigen::Vector3d> uv_c_cam;
    vector<Eigen::VectorXd> lines_c_cam;

    for (int i = 0; i < points_w.size(); ++i) {
        if (points_cam[i] == cam) {
            points_w_cam.push_back(points_w[i]);
            Eigen::Vector3d uv = uv_c[i];
            uv /= uv_c[i](2); // Normalize by depth
            uv_c_cam.push_back(uv);
        }
    }
    for (int i = 0; i < lines_w.size(); ++i) {
        if (lines_cam[i] == cam) {
            lines_w_cam.push_back(lines_w[i]);
            lines_c_cam.push_back(lines_c[i]);
        }
    }

    ofs << fixed << setprecision(10);

    ofs << time << endl; // Save timestamp

    ofs << points_w_cam.size() << endl;

    for (int i = 0; i < points_w_cam.size(); ++i) {
        ofs << points_w_cam[i].transpose() << " " << uv_c_cam[i].transpose()
            << endl;
    }

    ofs << lines_w_cam.size() << endl;

    for (int i = 0; i < lines_w_cam.size(); ++i) {
        ofs << lines_w_cam[i].transpose() << " " << lines_c_cam[i].transpose()
            << endl;
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
}; // namespace utils
