#include <Eigen/Dense>
#include <chrono>
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

void loadEurocCameraParameters(const string &cam1_file, const string &cam2_file,
                               Camera &camera1, Camera &camera2) {
    YAML::Node config1 = YAML::LoadFile(cam1_file);
    YAML::Node config2 = YAML::LoadFile(cam2_file);

    vector<double> intrinsics1 = config1["intrinsics"].as<vector<double>>();
    vector<double> intrinsics2 = config2["intrinsics"].as<vector<double>>();
    camera1.fx = intrinsics1[0];
    camera1.fy = intrinsics1[1];
    camera1.cx = intrinsics1[2];
    camera1.cy = intrinsics1[3];
    camera2.fx = intrinsics2[0];
    camera2.fy = intrinsics2[1];
    camera2.cx = intrinsics2[2];
    camera2.cy = intrinsics2[3];

    camera1.distortion_model = config1["distortion_model"].as<string>();
    camera2.distortion_model = config2["distortion_model"].as<string>();

    camera1.distortion_coeffs =
        config1["distortion_coefficients"].as<vector<double>>();
    camera2.distortion_coeffs =
        config2["distortion_coefficients"].as<vector<double>>();

    camera1.image_size = config1["resolution"].as<array<double, 2>>();
    camera2.image_size = config2["resolution"].as<array<double, 2>>();

    YAML::Node Tbc1_node = config1["T_BS"]["data"];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            camera1.T_bc(i, j) = Tbc1_node[i * 4 + j].as<double>();
        }
    }

    YAML::Node Tbc2_node = config2["T_BS"]["data"];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            camera2.T_bc(i, j) = Tbc2_node[i * 4 + j].as<double>();
        }
    }

    camera1.getRecitifiedPinholeCamera();
    camera2.getRecitifiedPinholeCamera();
}

void loadEurocImage(const string &image_path, const string &time_file,
                    const string &gt_file, vector<string> &image_files,
                    vector<double> &times,
                    vector<Eigen::Isometry3d> &poses_gt) {
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

int main(int argc, char **argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <sequence_name>" << endl;
        return 1;
    }
    string seq = argv[1];
    string euroc_path = "~/dataset/euroc/" + seq + "/mav0/";

    string upnpl_out_file = euroc_path + "upnpl_trajectory_" + seq + ".csv";
    string cv_epnp_out_file = euroc_path + "cv_epnp_trajectory_" + seq + ".csv";
    string upnp_out_file = euroc_path + "upnp_trajectory_" + seq + ".csv";
    string upnpl_points_out_file = euroc_path + "upnpl_points_" + seq + ".csv";
    string upnpl_lines_out_file = euroc_path + "upnpl_lines_" + seq + ".csv";
    string gpnp_out_file = euroc_path + "gpnp" + ".txt";
    string gt_out_file = euroc_path + "gt_trajectory_" + seq + ".csv";

    string cam1_file = euroc_path + "cam0/sensor.yaml";
    string cam2_file = euroc_path + "cam1/sensor.yaml";
    string gt_file = euroc_path + "state_groundtruth_estimate0/data.csv";

    string image1_path = euroc_path + "cam0/data/";
    string time1_file = euroc_path + "cam0/data.csv";

    string image2_path = euroc_path + "cam1/data/";
    string time2_file = euroc_path + "cam1/data.csv";

    vector<vector<string>> image_files(2);
    vector<vector<double>> times(2);
    vector<vector<Eigen::Isometry3d>> poses_gt(2);

    loadEurocImage(image1_path, time1_file, gt_file, image_files[0], times[0],
                   poses_gt[0]);
    loadEurocImage(image2_path, time2_file, gt_file, image_files[1], times[1],
                   poses_gt[1]);

    vector<Camera> cameras(2);
    loadEurocCameraParameters(cam1_file, cam2_file, cameras[0], cameras[1]);

    vector<double> times_save;
    vector<Eigen::Isometry3d> Twb_upnpl;
    vector<Eigen::Isometry3d> Twb_upnpl_points;
    vector<Eigen::Isometry3d> Twb_cv;
    vector<Eigen::Isometry3d> Twb_gt;
    vector<Eigen::Isometry3d> Twb_upnp;
    vector<Eigen::Isometry3d> Twb_upnpl_lines;
    vector<Eigen::Isometry3d> Twb_gpnp;
    double avg_time_upnpl = 0.0;
    double avg_time_cv = 0.0;
    double avg_time_upnp = 0.0;
    double avg_time_upnpl_points = 0.0;
    double avg_time_gpnp = 0.0;
    for (int i = 0; i < image_files[0].size() - 91; ++i) {
        string matlab_data =
            euroc_path + "data/" + "data_" + to_string(i) + ".txt";

        vector<Eigen::Vector3d> points_w;
        vector<double> points_sigma;
        vector<Eigen::VectorXd> lines_w;
        vector<double> lines_sigma;
        vector<Eigen::Vector3d> uv_c;
        vector<Eigen::VectorXd> lines_c;
        vector<int> points_cam;
        vector<int> lines_cam;

        try {
            generatePnPLData(i, image_files, times, poses_gt, cameras, points_w,
                             points_sigma, lines_w, lines_sigma, uv_c, lines_c,
                             points_cam, lines_cam, 800);
        } catch (const std::exception &e) {
            cerr << "Error generating data for frame " << i << ": " << e.what()
                 << endl;
            continue;
        }
        saveDataForMatlab(points_w, points_sigma, lines_w, lines_sigma, uv_c,
                          lines_c, points_cam, lines_cam, matlab_data,
                          times[0][i + 1]);
        Twb_gt.push_back(poses_gt[0][i + 1]);
        times_save.push_back(times[0][i + 1]);
        Eigen::Isometry3d Tbw = Eigen::Isometry3d::Identity();
        double used_time = 0.0;
        Tbw = utils::myUPnPL(points_w, lines_w, uv_c, lines_c, points_cam,
                             lines_cam, cameras, 2, used_time);
        // upnpl.solveUPnPL_EPnPL(points_w, lines_w, uv_c, normals_c,
        // points_cam,
        //                        lines_cam, R_bc, t_bc, R_bw, t_bw);
        avg_time_upnpl += used_time;

        Twb_upnpl.push_back(Tbw.inverse());

        // // OpenCV EPnP
        // cv::Mat camera_matrix =
        //     (cv::Mat_<double>(3, 3) << cameras[0].fx, 0, cameras[0].cx, 0,
        //      cameras[0].fy, cameras[0].cy, 0, 0, 1);
        // cv::Mat dist_coeffs = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0);
        //
        // vector<cv::Point3f> object_points;
        // vector<cv::Point2f> image_points;
        //
        // for (int j = 0; j < points_w.size(); ++j) {
        //     if (points_cam[j] == 0) {
        //         object_points.emplace_back(points_w[j](0), points_w[j](1),
        //                                    points_w[j](2));
        //         Eigen::Vector3d uv = uv_c[j];
        //         uv /= uv(2);
        //         cv::Point2f uv_image;
        //         uv_image.x = uv(0) * cameras[0].fx + cameras[0].cx;
        //         uv_image.y = uv(1) * cameras[0].fy + cameras[0].cy;
        //         image_points.emplace_back(uv_image.x, uv_image.y);
        //     }
        // }
        // cv::Mat rvec, tvec;
        // start = chrono::high_resolution_clock::now();
        // bool success = false;
        // if (object_points.size() >= 4 && image_points.size() >= 4)
        //     success =
        //         cv::solvePnP(object_points, image_points, camera_matrix,
        //                      dist_coeffs, rvec, tvec, false,
        //                      cv::SOLVEPNP_EPNP);
        // end = chrono::high_resolution_clock::now();
        // elapsed = end - start;

        Tbw = utils::cv_EPnP(points_w, uv_c, points_cam, cameras, used_time);
        avg_time_cv += used_time;
        Twb_cv.push_back(Tbw.inverse());

        Tbw =
            utils::opengv_UPnP(points_w, uv_c, points_cam, cameras, used_time);
        avg_time_upnp += used_time;
        Twb_upnp.push_back(Tbw.inverse());

        Tbw =
            utils::opengv_GPnP(points_w, uv_c, points_cam, cameras, used_time);
        avg_time_gpnp += used_time;
        Twb_gpnp.push_back(Tbw.inverse());

        vector<int> points_cam_tmp;
        vector<Eigen::Vector3d> points_w_tmp;
        vector<Eigen::Vector3d> uv_c_tmp;

        Tbw = utils::myUPnPL(points_w_tmp, lines_w, uv_c_tmp, lines_c,
                             points_cam_tmp, lines_cam, cameras, 2, used_time);
        Twb_upnpl_lines.push_back(Tbw.inverse());

        lines_w.clear();
        lines_c.clear();
        lines_cam.clear();

        Tbw = utils::myUPnPL(points_w, lines_w, uv_c, lines_c, points_cam,
                             lines_cam, cameras, 2, used_time);
        Twb_upnpl_points.push_back(Tbw.inverse());
        avg_time_upnpl_points += used_time;

        cout << "Processed frame " << i << endl;
        // cout << "opencv EPnP Transformation: "
        //      << Twb_cv.back().translation().transpose() << endl;
        // cout << "UPnPL Transformation: "
        //      << Twb_upnpl.back().translation().transpose() << endl;
        // cout << "Ground Truth Transformation: "
        //      << Twb_gt.back().translation().transpose() << endl;
        // Eigen::Matrix3d R_bw_gt = Twb_gt.back().inverse().linear();
        // cout << "gt R_bw: " << R_bw_gt << endl;
        // Eigen::Vector3d caley_gt = rotToCGR(R_bw_gt);
        // cout << "Ground Truth Caley vector: " << caley_gt.transpose() <<
        // endl; cout << "tbw: " <<
        // Twb_gt.back().inverse().translation().transpose()
        //      << endl;
    }

    avg_time_upnpl /= Twb_upnpl.size();
    avg_time_cv /= Twb_cv.size();
    avg_time_upnp /= Twb_upnp.size();
    avg_time_upnpl_points /= Twb_upnpl_points.size();
    avg_time_gpnp /= Twb_gpnp.size();

    cout << "Average UPnPL time: " << avg_time_upnpl << " ms" << endl;
    cout << "Average OpenCV EPnP time: " << avg_time_cv << " ms" << endl;
    cout << "Average OpenGV UPnP time: " << avg_time_upnp << " ms" << endl;
    cout << "Average UPnPL only points time: " << avg_time_upnpl_points << " ms"
         << endl;
    cout << "Average OpenGV GPnP time: " << avg_time_gpnp << " ms" << endl;

    saveEurocTraejectory(upnpl_out_file, Twb_upnpl, times_save);
    saveEurocTraejectory(cv_epnp_out_file, Twb_cv, times_save);
    saveEurocTraejectory(upnp_out_file, Twb_upnp, times_save);
    saveEurocTraejectory(upnpl_points_out_file, Twb_upnpl_points, times_save);
    saveEurocTraejectory(upnpl_lines_out_file, Twb_upnpl_lines, times_save);
    saveEurocTraejectory(gt_out_file, Twb_gt, times_save);
    saveEurocTraejectory(gpnp_out_file, Twb_gpnp, times_save);
}
