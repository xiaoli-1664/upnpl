#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "UPnPL.h"
#include "generate.h"
#include "rapidcsv.h"

using namespace std;

void loadTUMCameraParameters(const string &cam_file, Camera &camera1,
                             Camera &camera2) {
    YAML::Node config = YAML::LoadFile(cam_file);
    vector<double> intrinsics1 =
        config["cam0"]["intrinsics"].as<vector<double>>();
    vector<double> intrinsics2 =
        config["cam1"]["intrinsics"].as<vector<double>>();
    camera1.fx = intrinsics1[0];
    camera1.fy = intrinsics1[1];
    camera1.cx = intrinsics1[2];
    camera1.cy = intrinsics1[3];
    camera2.fx = intrinsics2[0];
    camera2.fy = intrinsics2[1];
    camera2.cx = intrinsics2[2];
    camera2.cy = intrinsics2[3];

    camera1.distortion_model = config["cam0"]["distortion_model"].as<string>();
    camera2.distortion_model = config["cam1"]["distortion_model"].as<string>();

    camera1.distortion_coeffs =
        config["cam0"]["distortion_coeffs"].as<vector<double>>();
    camera2.distortion_coeffs =
        config["cam1"]["distortion_coeffs"].as<vector<double>>();

    camera1.image_size = config["cam0"]["resolution"].as<vector<double>>();
    camera2.image_size = config["cam1"]["resolution"].as<vector<double>>();

    YAML::Node Tbc1_node = config["cam0"]["T_cam_imu"];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            camera1.Tbc(i, j) = Tbc1_node[i][j].as<double>();
        }
    }
    camera1.Tbc = camera1.Tbc.inverse();

    YAML::Node Tbc2_node = config["cam1"]["T_cam_imu"];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            camera2.Tbc(i, j) = Tbc2_node[i][j].as<double>();
        }
    }
    camera2.Tbc = camera2.Tbc.inverse();

    camera1.getRectifiedFisheyeCamera();
    camera2.getRectifiedFisheyeCamera();
}

void loadTUMImage(const string &image_path, const string &time_file,
                  const string &gt_file, vector<string> &image_files,
                  vector<double> &times, vector<Eigen::Isometry3d> &poses_gt) {
    ifstream time_ifs(time_file);
    ifstream gt_ifs(gt_file);
    if (!time_ifs.is_open()) {
        cerr << "Failed to open time file: " << time_file << endl;
        return;
    }

    if (!gt_ifs.is_open()) {
        cerr << "Failed to open ground truth file: " << gt_file << endl;
        return;
    }

    rapidcsv::Document time_doc(time_file, rapidcsv::LabelParams(0, -1),
                                rapidcsv::SeparatorParams(','));

    rapidcsv::Document gt_doc(gt_file, rapidcsv::LabelParams(0, -1),
                              rapidcsv::SeparatorParams(','));

    size_t time_row_count = time_doc.GetRowCount();
    size_t gt_row_count = gt_doc.GetRowCount();

    int j = 0;
    for (int i = 0; i < time_row_count; ++i) {
        double time = time_doc.GetCell<double>(0, i);
        time /= 1e9; // Convert nanoseconds to seconds
        string image_file = image_path + time_doc.GetCell<string>(1, i);

        Eigen::Isometry3d pose_gt;

        while (j < gt_row_count) {
            double gt_time =
                gt_doc.GetCell<double>(0, j) / 1e9; // Convert to seconds
            if (fabs(gt_time - time) < 0.002) {
                pose_gt.translation() << gt_doc.GetCell<double>(1, j),
                    gt_doc.GetCell<double>(2, j), gt_doc.GetCell<double>(3, j);
                Eigen::Quaterniond q(
                    gt_doc.GetCell<double>(4, j), gt_doc.GetCell<double>(5, j),
                    gt_doc.GetCell<double>(6, j), gt_doc.GetCell<double>(7, j));
                q.normalize();
                pose_gt.linear() = q.toRotationMatrix();
                break;
            } else if (gt_time > time && j > 0) {
                double gt_time_last = gt_doc.GetCell<double>(0, j - 1) / 1e9;
                Eigen::Quaterniond q_last(gt_doc.GetCell<double>(4, j - 1),
                                          gt_doc.GetCell<double>(5, j - 1),
                                          gt_doc.GetCell<double>(6, j - 1),
                                          gt_doc.GetCell<double>(7, j - 1));
                q_last.normalize();
                Eigen::Vector3d t_last(gt_doc.GetCell<double>(1, j - 1),
                                       gt_doc.GetCell<double>(2, j - 1),
                                       gt_doc.GetCell<double>(3, j - 1));
                Eigen::Quaterniond q_curr(
                    gt_doc.GetCell<double>(4, j), gt_doc.GetCell<double>(5, j),
                    gt_doc.GetCell<double>(6, j), gt_doc.GetCell<double>(7, j));
                q_curr.normalize();
                Eigen::Vector3d t_curr(gt_doc.GetCell<double>(1, j),
                                       gt_doc.GetCell<double>(2, j),
                                       gt_doc.GetCell<double>(3, j));
                double alpha = (time - gt_time_last) / (gt_time - gt_time_last);

                Eigen::Vector3d t_inter = t_last + alpha * (t_curr - t_last);
                Eigen::Quaterniond q_inter = q_last.slerp(alpha, q_curr);
                q_inter.normalize();
                pose_gt.translation() = t_inter;
                pose_gt.linear() = q_inter.toRotationMatrix();
                break;
            }
            ++j;
        }

        if (j < gt_row_count) {
            image_files.push_back(image_file);
            times.push_back(time);
            poses_gt.push_back(pose_gt);
        }
    }
}

int main() {
    string tum_path = "/home/ljj/dataset/tum/room2/";
    string cam_file = tum_path + "dso/camchain.yaml";
    string gt_file = tum_path + "mav0/mocap0/data.csv";

    string image1_path = tum_path + "mav0/cam0/data/";
    string time1_file = tum_path + "mav0/cam0/data.csv";

    string image2_path = tum_path + "mav0/cam1/data/";
    string time2_file = tum_path + "mav0/cam1/data.csv";

    string calib_file = tum_path + "dso/camchain.yaml";

    vector<string> image1_files, image2_files;
    vector<double> time1, time2;
    vector<Eigen::Isometry3d> pose_gt1, pose_gt2;

    loadTUMImage(image1_path, time1_file, gt_file, image1_files, time1,
                 pose_gt1);
    loadTUMImage(image2_path, time2_file, gt_file, image2_files, time2,
                 pose_gt2);

    vector<Camera> cameras(2);
    loadTUMCameraParameters(calib_file, cameras[0], cameras[1]);

    vector<vector<string>> image_files(2);
    vector<vector<double>> times(2);
    vector<vector<Eigen::Isometry3d>> poses_gt(2);

    for (size_t i = 0; i < image1_files.size(); ++i) {
        image_files[0].push_back(image1_files[i]);
        times[0].push_back(time1[i]);
        poses_gt[0].push_back(pose_gt1[i]);
    }

    for (size_t i = 0; i < image2_files.size(); ++i) {
        image_files[1].push_back(image2_files[i]);
        times[1].push_back(time2[i]);
        poses_gt[1].push_back(pose_gt2[i]);
    }

    vector<Eigen::Matrix3d> R_bc;
    vector<Eigen::Vector3d> t_bc;
    for (int j = 0; j < cameras.size(); ++j) {
        R_bc.push_back(cameras[j].Tbc.linear());
        t_bc.push_back(cameras[j].Tbc.translation());
    }

    for (int i = 0; i < image_files[0].size() - 1; ++i) {
        vector<Eigen::Vector3d> points_w;
        vector<Eigen::VectorXd> lines_w;
        vector<Eigen::Vector3d> uv_c;
        vector<Eigen::Vector3d> normals_c;
        vector<int> points_cam;
        vector<int> lines_cam;

        generatePnPLData(i, image_files, times, poses_gt, cameras, points_w,
                         lines_w, uv_c, normals_c, points_cam, lines_cam);
        Eigen::Matrix3d R_bw;
        Eigen::Vector3d t_bw;
        UPnPL::UPnPL upnpl;
        upnpl.solveUPnPL(points_w, lines_w, uv_c, normals_c, points_cam,
                         lines_cam, R_bc, t_bc, R_bw, t_bw);
        Eigen::Isometry3d T_bw = Eigen::Isometry3d::Identity();
        T_bw.linear() = R_bw;
        T_bw.translation() = t_bw;
        Eigen::Isometry3d T_wb = T_bw.inverse();

        // cv EPnP
        vector<cv::Point3f> object_points;
        vector<cv::Point2f> image_points;
        cv::Mat camera_matrix =
            (cv::Mat_<double>(3, 3) << cameras[0].fx, 0, cameras[0].cx, 0,
             cameras[0].fy, cameras[0].cy, 0, 0, 1);
        cv::Mat dist_coeffs = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0);
        cv::Mat rvec, tvec;

        for (int i = 0; i < points_w.size(); ++i) {
            if (points_cam[i] == 0) {
                object_points.emplace_back(points_w[i](0), points_w[i](1),
                                           points_w[i](2));
                Eigen::Vector3d uv = uv_c[i];
                uv /= uv(2);
                cv::Point2f uv_image;
                uv_image.x = uv(0) * cameras[0].fx + cameras[0].cx;
                uv_image.y = uv(1) * cameras[0].fy + cameras[0].cy;
                image_points.emplace_back(uv_image.x, uv_image.y);
            }
        }

        cv::solvePnP(object_points, image_points, camera_matrix, dist_coeffs,
                     rvec, tvec, false, cv::SOLVEPNP_EPNP);
        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);
        Eigen::Matrix3d R_cv_eigen = cvMatToEigen(R_cv);
        Eigen::Vector3d t_cv_eigen(tvec.at<double>(0), tvec.at<double>(1),
                                   tvec.at<double>(2));
        Eigen::Isometry3d T_cw = Eigen::Isometry3d::Identity();
        T_cw.linear() = R_cv_eigen;
        T_cw.translation() = t_cv_eigen;
        Eigen::Isometry3d T_bw_cv = cameras[0].Tbc * T_cw;
        Eigen::Isometry3d T_wb_cv = T_bw_cv.inverse();

        // cout << "Pose for image index " << i << ": " << endl;
        // cout << "points num: " << points_w.size() << endl;
        // cout << "lines num: " << lines_w.size() << endl;
        // cout << "opencv EPnP Pose:\n";
        // cout << "Translation:\n" << T_wb_cv.translation().transpose() <<
        // endl; cout << "UPnPL Pose:\n"; cout << "Translation:\n" <<
        // T_wb.translation().transpose() << endl; cout << "Ground truth
        // Pose:\n"; cout << "Translation:\n"
        //      << poses_gt[0][i + 1].translation().transpose() << endl;
    }

    return 0;
}
