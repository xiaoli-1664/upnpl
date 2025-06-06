#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

#include "UPnPL.h"

using namespace std;

struct Camera {
    cv::Mat P;       // 3x4 Projection Matrix
    cv::Mat K, R, t; // Decomposed
};

Eigen::MatrixXd cvMatToEigen(const cv::Mat &mat) {
    Eigen::MatrixXd eigen_mat(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            eigen_mat(i, j) = mat.at<double>(i, j);
        }
    }
    return eigen_mat;
}

Eigen::Vector3d backProjectPixel(double u, double v, const Camera &camera) {
    Eigen::Vector3d pixel(u, v, 1.0);
    Eigen::Vector3d world_point;
    world_point(0) =
        (pixel(0) - camera.K.at<double>(0, 2)) / camera.K.at<double>(0, 0);
    world_point(1) =
        (pixel(1) - camera.K.at<double>(1, 2)) / camera.K.at<double>(1, 1);
    world_point(2) = 1.0; // Assuming unit depth for back-projection
    return world_point;
}

vector<Eigen::Vector3d> readPointsFromFile(const std::string &path,
                                           const Camera &camera) {
    ifstream ifs(path);
    vector<Eigen::Vector3d> points;
    double x, y;
    while (ifs >> x >> y) {
        Eigen::Vector3d p_c = backProjectPixel(x, y, camera);
        points.push_back(p_c);
    }
    return points;
}

vector<Eigen::Vector2d> readPixelsFromFile(const std::string &path) {
    ifstream ifs(path);
    vector<Eigen::Vector2d> pixels;
    double x, y;
    while (ifs >> x >> y) {
        Eigen::Vector2d pixel(x, y);
        pixels.push_back(pixel);
    }
    return pixels;
}

vector<Eigen::Vector3d> readLinesFromFile(const std::string &path,
                                          const Camera &camera) {
    ifstream ifs(path);
    vector<Eigen::Vector3d> normals;
    double x1, y1, x2, y2;
    while (ifs >> x1 >> y1 >> x2 >> y2) {
        Eigen::Vector3d p1 = backProjectPixel(x1, y1, camera);
        Eigen::Vector3d p2 = backProjectPixel(x2, y2, camera);
        Eigen::Vector3d normal = p1.cross(p2);
        normal.normalize();
        normals.push_back(normal);
    }
    return normals;
}

cv::Mat readMatrixFromFile(const std::string &path, int rows, int cols) {
    std::ifstream ifs(path);
    cv::Mat mat(rows, cols, CV_64F);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            ifs >> mat.at<double>(i, j);
    return mat;
}

int main() {
    string dataset_path = "/home/ljj/dataset/vgg/multi-view/";
    string dataset_name = "Corridor";
    dataset_path += dataset_name + "/";
    string dir_2d = dataset_path + "2D/bt.";
    string dir_3d = dataset_path + "3D/bt.";

    // read points
    vector<cv::Point3f> object_points;
    ifstream points_file(dir_3d + "p3d");
    double x, y, z;
    while (points_file >> x >> y >> z) {
        object_points.emplace_back(x, y, z);
    }

    // read lines
    vector<Eigen::VectorXd> object_lines;
    ifstream lines_file(dir_3d + "l3d");
    double x1, y1, z1, x2, y2, z2;
    while (lines_file >> x1 >> y1 >> z1 >> x2 >> y2 >> z2) {
        Eigen::VectorXd line(6);
        line << x1, y1, z1, x2, y2, z2;
        object_lines.push_back(line);
    }

    int num_views = 1;

    vector<Camera> cameras(num_views);
    vector<Eigen::Matrix3d> Rbc(num_views);
    vector<Eigen::Vector3d> tbc(num_views);

    for (int i = 0; i < num_views; ++i) {
        char buf[16];
        sprintf(buf, "%03d", i);

        string P_path = dir_2d + buf + ".P";

        cv::Mat P = readMatrixFromFile(P_path, 3, 4);
        cv::Mat temp_K, temp_R, temp_t;
        cv::decomposeProjectionMatrix(P, temp_K, temp_R, temp_t);

        temp_K = temp_K * cv::Mat::diag((cv::Mat_<double>(3, 1) << 1, 1, -1));
        temp_R = cv::Mat::diag((cv::Mat_<double>(3, 1) << -1, -1, 1)) * temp_R;
        temp_t =
            cv::Mat::diag((cv::Mat_<double>(4, 1) << -1, -1, 1, 1)) * temp_t;

        cameras[i].P = P;
        cameras[i].K = temp_K;
        cameras[i].R = temp_R;
        cameras[i].t = cv::Mat::zeros(3, 1, CV_64F);

        cv::Mat C = cv::Mat::zeros(3, 1, CV_64F);
        C.at<double>(0) = temp_t.at<double>(0) / temp_t.at<double>(3);
        C.at<double>(1) = temp_t.at<double>(1) / temp_t.at<double>(3);
        C.at<double>(2) = temp_t.at<double>(2) / temp_t.at<double>(3);

        cameras[i].t = -temp_R * C;
        cout << "Camera " << i << ":\n"
             << "K:\n"
             << cameras[i].K << "\nR:\n"
             << cameras[i].R << "\nt:\n"
             << cameras[i].t.t() << endl;

        Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
        Tcw.linear() = cvMatToEigen(cameras[i].R);
        Tcw.translation() = Eigen::Vector3d(cameras[i].t.at<double>(0),
                                            cameras[i].t.at<double>(1),
                                            cameras[i].t.at<double>(2));
        Eigen::Isometry3d Tbw = Eigen::Isometry3d::Identity();
        Tbw.linear() = cvMatToEigen(cameras[0].R);
        Tbw.translation() = Eigen::Vector3d(cameras[0].t.at<double>(0),
                                            cameras[0].t.at<double>(1),
                                            cameras[0].t.at<double>(2));

        Eigen::Isometry3d Tbc = Tbw * Tcw.inverse();
        Rbc[i] = Tbc.linear();
        tbc[i] = Tbc.translation();
    }

    vector<vector<Eigen::Vector3d>> uv(num_views);
    vector<vector<Eigen::Vector3d>> normals(num_views);
    vector<vector<Eigen::Vector2d>> pixels(num_views);

    for (int i = 0; i < num_views; ++i) {
        char buf[16];
        sprintf(buf, "%03d", i);

        string corners_path = dir_2d + buf + ".corners";
        string normals_path = dir_2d + buf + ".lines";

        // uv[i] = readPointsFromFile(corners_path, cameras[i]);
        normals[i] = readLinesFromFile(normals_path, cameras[i]);
        pixels[i] = readPixelsFromFile(corners_path);
        uv[i].resize(pixels[i].size());
        for (size_t j = 0; j < pixels[i].size(); ++j) {
            double u = pixels[i][j].x();
            double v = pixels[i][j].y();
            double x = (u - cameras[i].K.at<double>(0, 2)) /
                       cameras[i].K.at<double>(0, 0);
            double y = (v - cameras[i].K.at<double>(1, 2)) /
                       cameras[i].K.at<double>(1, 1);
            uv[i][j] = Eigen::Vector3d(x, y, 1.0);
        }
    }

    // {
    //     Eigen::MatrixXd P2 = cvMatToEigen(cameras[2].P);
    //     Eigen::Vector3d p =
    //         P2 * Eigen::Vector4d(object_points[0].x, object_points[0].y,
    //                              object_points[0].z, 1);
    //     p /= p(2); // Normalize by z
    //     cout << "First point in camera 0 coordinates: " << p.transpose()
    //          << endl;
    // }

    vector<Eigen::Vector3d> uv_c;
    vector<Eigen::Vector3d> normals_c;
    vector<int> points_cam, lines_cam;

    vector<Eigen::Vector3d> points_w;
    vector<Eigen::VectorXd> lines_w;

    vector<cv::Point3f> object_points_cv;
    vector<cv::Point2f> image_points_cv;

    ifstream ifs_pmatch(dir_2d + "nview-corners");
    string line;
    int row = 0;
    while (getline(ifs_pmatch, line)) {
        stringstream ss(line);
        string token;
        int cam_id = 0;
        while (ss >> token) {
            if (cam_id >= num_views) {
                break; // Avoid out of bounds
            }
            if (token != "*") {
                int point_id = stoi(token);
                points_w.push_back(Eigen::Vector3d(object_points[row].x,
                                                   object_points[row].y,
                                                   object_points[row].z));
                points_cam.push_back(cam_id);
                uv_c.push_back(uv[cam_id][point_id]);
                if (cam_id == 0) {
                    object_points_cv.emplace_back(object_points[row].x,
                                                  object_points[row].y,
                                                  object_points[row].z);
                    image_points_cv.emplace_back(pixels[cam_id][point_id].x(),
                                                 pixels[cam_id][point_id].y());
                }
            }
            ++cam_id;
        }
        ++row;
    }

    ifstream ifs_lmatch(dir_2d + "nview-lines");
    row = 0;
    // while (getline(ifs_lmatch, line)) {
    //     stringstream ss(line);
    //     string token;
    //     int cam_id = 0;
    //     while (ss >> token) {
    //         if (cam_id >= num_views) {
    //             break; // Avoid out of bounds
    //         }
    //         if (token != "*") {
    //             int line_id = stoi(token);
    //             lines_w.push_back(object_lines[row]);
    //             lines_cam.push_back(cam_id);
    //             normals_c.push_back(normals[cam_id][line_id]);
    //         }
    //         ++cam_id;
    //     }
    //     ++row;
    // }

    Eigen::Matrix3d R_bw = cvMatToEigen(cameras[0].R);
    Eigen::Vector3d t_bw =
        Eigen::Vector3d(cameras[0].t.at<double>(0), cameras[0].t.at<double>(1),
                        cameras[0].t.at<double>(2));

    Eigen::Matrix3d R_bw_est;
    Eigen::Vector3d t_bw_est;

    UPnPL::UPnPL upnpl;
    upnpl.solveUPnPL(points_w, lines_w, uv_c, normals_c, points_cam, lines_cam,
                     Rbc, tbc, R_bw_est, t_bw_est);

    cout << "ground truth Rotation:\n" << R_bw << endl;
    cout << "ground truth Translation:\n" << t_bw.transpose() << endl;

    cout << "Estimated Rotation:\n" << R_bw_est << endl;
    cout << "Estimated Translation:\n" << t_bw_est.transpose() << endl;

    cv::Mat r_vec, t_vec;
    bool success =
        cv::solvePnP(object_points_cv, image_points_cv, cameras[0].K, cv::Mat(),
                     r_vec, t_vec, false, cv::SOLVEPNP_EPNP);

    if (success) {
        cv::Mat R_est;
        cv::Rodrigues(r_vec, R_est);
        Eigen::Matrix3d R_est_eigen = cvMatToEigen(R_est);
        Eigen::Vector3d t_est_eigen(t_vec.at<double>(0), t_vec.at<double>(1),
                                    t_vec.at<double>(2));

        cout << "OpenCV Estimated Rotation:\n" << R_est_eigen << endl;
        cout << "OpenCV Estimated Translation:\n"
             << t_est_eigen.transpose() << endl;
    } else {
        cout << "OpenCV solvePnP failed." << endl;
    }

    return 0;
}
