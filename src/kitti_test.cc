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

string formatSerialNumber(int index) {
    ostringstream oss;
    oss << setw(10) << setfill('0') << index;
    return oss.str();
}

double timestampStringToDouble(const string &timestamp) {
    if (timestamp.size() < 27) {
        cerr << "Error: Timestamp string is too short." << endl;
        return 0.0;
    }

    tm tm = {};
    istringstream ss(timestamp.substr(0, 19));

    ss >> get_time(&tm, "%Y-%m-%d %H:%M:%S");
    if (ss.fail()) {
        cerr << "Error: Failed to parse timestamp." << endl;
        return 0.0;
    }

    auto tp = chrono::system_clock::from_time_t(mktime(&tm));

    string nano_str = timestamp.substr(20);
    if (nano_str.size() > 0 && nano_str.size() <= 9) {
        nano_str.append(9 - nano_str.size(), '0'); // Pad with zeros
        long nanoseconds = stol(nano_str);
        tp += chrono::nanoseconds(nanoseconds);
    }

    auto duration = tp.time_since_epoch();
    return chrono::duration_cast<chrono::duration<double>>(duration).count();
}

vector<double> parseLineDoubles(const string &line) {
    istringstream iss(line.substr(line.find(":") + 1));
    vector<double> values;
    double val;
    while (iss >> val) {
        values.push_back(val);
    }
    return values;
}

void saveIsometry3dList(const vector<Eigen::Isometry3d> &T_list,
                        const string &filename) {
    YAML::Emitter out;
    out << YAML::BeginMap;

    for (size_t i = 0; i < T_list.size(); ++i) {
        string key = "T_" + to_string(i);
        out << YAML::Key << key;
        out << YAML::Value << YAML::BeginMap;

        out << YAML::Key << "data" << YAML::Value << YAML::Flow
            << YAML::BeginSeq;
        Eigen::Matrix4d mat = T_list[i].matrix();
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col) {
                out << mat(row, col);
            }
        out << YAML::EndSeq; // End of data
        out << YAML::EndMap;
    }

    out << YAML::EndMap;

    ofstream fout(filename);
    fout << out.c_str();
}

void saveCameraParameters(const string &filename, const Camera &camera) {
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "T_BS";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "cols" << YAML::Value << 4;
    out << YAML::Key << "rows" << YAML::Value << 4;

    out << YAML::Key << "data" << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            out << camera.T_bc.matrix()(i, j);
    out << YAML::EndSeq;
    out << YAML::EndMap;
    out << YAML::EndMap;

    out << YAML::BeginMap;
    out << YAML::Key << "intrinsics" << YAML::Value << YAML::Flow
        << YAML::BeginSeq;
    out << camera.fx << camera.fy << camera.cx << camera.cy;
    out << YAML::EndSeq;
    out << YAML::EndMap;

    fs::path output_path(filename);
    if (!fs::exists(output_path.parent_path())) {
        fs::create_directories(output_path.parent_path());
    }

    ofstream fout(filename);
    fout << out.c_str();
    fout.close();
}

Eigen::Isometry3d vectorToIsometry3d(const vector<double> &vec) {
    if (vec.size() != 16 && vec.size() != 12) {
        cerr << "Error: Vector size must be 16 or 12 for Isometry3d conversion."
             << endl;
        return Eigen::Isometry3d::Identity();
    }
    Eigen::Isometry3d T;
    T.setIdentity();
    T.linear() << vec[0], vec[1], vec[2], vec[4], vec[5], vec[6], vec[8],
        vec[9], vec[10];
    T.translation() << vec[3], vec[7], vec[11];
    return T;
}

void loadKittiStereoCameraParameters(const string &cam_file, Camera &camera0,
                                     Camera &camera1) {
    ifstream file(cam_file);

    if (!file.is_open()) {
        cerr << "Error: Could not open camera file " << cam_file << endl;
        return;
    }

    camera0.T_bc.setIdentity();
    camera1.T_bc.setIdentity();
    camera0.distortion_coeffs = {0.0, 0.0, 0.0, 0.0};
    camera1.distortion_coeffs = {0.0, 0.0, 0.0, 0.0};

    string line;
    while (getline(file, line)) {
        if (line.find("S_rect_00") != string::npos) {
            vector<double> size = parseLineDoubles(line);
            camera0.image_size = {size[0], size[1]};
        } else if (line.find("P_rect_00") != string::npos) {
            vector<double> P = parseLineDoubles(line);
            camera0.fx = P[0];
            camera0.cx = P[2];
            camera0.fy = P[5];
            camera0.cy = P[6];
        } else if (line.find("S_rect_01") != string::npos) {
            vector<double> size = parseLineDoubles(line);
            camera1.image_size = {size[0], size[1]};
        } else if (line.find("P_rect_01") != string::npos) {
            vector<double> P = parseLineDoubles(line);
            camera1.fx = P[0];
            camera1.cx = P[2];
            camera1.fy = P[5];
            camera1.cy = P[6];

            double b = -P[3] / P[0];               // Baseline
            camera1.T_bc.translation() << b, 0, 0; // Set translation
        }
    }
}

void loadKittiMeiCameraParameters(const string &cam2_file,
                                  const string &cam3_file,
                                  const string &extrinsic_file, Camera &camera2,
                                  Camera &camera3) {
    YAML::Node config2 = YAML::LoadFile(cam2_file);
    YAML::Node config3 = YAML::LoadFile(cam3_file);

    camera2.image_size[0] = config2["image_width"].as<double>();
    camera2.image_size[1] = config2["image_height"].as<double>();
    camera3.image_size[0] = config3["image_width"].as<double>();
    camera3.image_size[1] = config3["image_height"].as<double>();

    camera2.xi = config2["mirror_parameters"]["xi"].as<double>();
    camera3.xi = config3["mirror_parameters"]["xi"].as<double>();

    camera2.distortion_coeffs.resize(4);
    camera2.distortion_coeffs[0] =
        config2["distortion_parameters"]["k1"].as<double>();
    camera2.distortion_coeffs[1] =
        config2["distortion_parameters"]["k2"].as<double>();
    camera2.distortion_coeffs[2] =
        config2["distortion_parameters"]["p1"].as<double>();
    camera2.distortion_coeffs[3] =
        config2["distortion_parameters"]["p2"].as<double>();

    camera3.distortion_coeffs.resize(4);
    camera3.distortion_coeffs[0] =
        config3["distortion_parameters"]["k1"].as<double>();
    camera3.distortion_coeffs[1] =
        config3["distortion_parameters"]["k2"].as<double>();
    camera3.distortion_coeffs[2] =
        config3["distortion_parameters"]["p1"].as<double>();
    camera3.distortion_coeffs[3] =
        config3["distortion_parameters"]["p2"].as<double>();

    camera2.projection_parameters.resize(4);
    camera2.projection_parameters[0] =
        config2["projection_parameters"]["gamma1"].as<double>();
    camera2.projection_parameters[1] =
        config2["projection_parameters"]["gamma2"].as<double>();
    camera2.projection_parameters[2] =
        config2["projection_parameters"]["u0"].as<double>();
    camera2.projection_parameters[3] =
        config2["projection_parameters"]["v0"].as<double>();

    camera3.projection_parameters.resize(4);
    camera3.projection_parameters[0] =
        config3["projection_parameters"]["gamma1"].as<double>();
    camera3.projection_parameters[1] =
        config3["projection_parameters"]["gamma2"].as<double>();
    camera3.projection_parameters[2] =
        config3["projection_parameters"]["u0"].as<double>();
    camera3.projection_parameters[3] =
        config3["projection_parameters"]["v0"].as<double>();

    ifstream extrinsic_file_stream(extrinsic_file);

    if (!extrinsic_file_stream.is_open()) {
        cerr << "Error: Could not open extrinsic file " << extrinsic_file
             << endl;
        return;
    }

    string line;

    vector<Eigen::Isometry3d> extrinsics(4);
    while (getline(extrinsic_file_stream, line)) {
        if (line.find("image_02") != string::npos) {
            extrinsics[2] = vectorToIsometry3d(parseLineDoubles(line));
        } else if (line.find("image_03") != string::npos) {
            extrinsics[3] = vectorToIsometry3d(parseLineDoubles(line));
        } else if (line.find("image_00") != string::npos) {
            extrinsics[0] = vectorToIsometry3d(parseLineDoubles(line));
        } else if (line.find("image_01") != string::npos) {
            extrinsics[1] = vectorToIsometry3d(parseLineDoubles(line));
        }
    }

    camera2.T_bc = extrinsics[0].inverse() * extrinsics[2];
    camera3.T_bc = extrinsics[0].inverse() * extrinsics[3];

    camera2.getRecitifiedMeiCamera();
    camera3.getRecitifiedMeiCamera();
}

void loadKittiImage(const string &image_path, const string &time_file,
                    const string &gt_file, vector<string> &image_files,
                    vector<double> &times, vector<Eigen::Isometry3d> &poses_gt,
                    bool is_02 = false) {
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

    string line;
    int i = 0;
    if (is_02)
        i = 4391;
    vector<double> times_tmp;
    vector<string> image_files_tmp;
    while (getline(time_ifs, line)) {
        double time = timestampStringToDouble(line);
        times_tmp.push_back(time);
        string image_file = image_path + formatSerialNumber(i++) + ".png";
        image_files_tmp.push_back(image_file);
    }

    vector<double> gt_times;
    vector<Eigen::Isometry3d> gt_poses;
    while (getline(gt_ifs, line)) {
        istringstream iss(line);
        int index = 0;
        iss >> index;
        vector<double> T(16);
        for (int i = 0; i < 16; ++i) {
            iss >> T[i];
        }
        gt_times.push_back(times_tmp[index - 1]);
        gt_poses.push_back(vectorToIsometry3d(T));
    }

    int gt_count = gt_poses.size() - 1;
    int time_count = times_tmp.size() - 1;

    int j = 0;
    for (int i = 0; i < time_count; ++i) {
        double time = times_tmp[i];

        Eigen::Isometry3d pose_gt;

        while (j < gt_count) {
            double gt_time = gt_times[j];
            if (fabs(gt_time - time) < 0.002) {
                pose_gt = gt_poses[j];
                break;
            } else if (gt_time > time && j > 0) {
                double gt_time_last = gt_times[j - 1];
                Eigen::Quaterniond q_last(gt_poses[j - 1].linear());
                Eigen::Vector3d t_last = gt_poses[j - 1].translation();
                Eigen::Quaterniond q_curr(gt_poses[j].linear());
                Eigen::Vector3d t_curr = gt_poses[j].translation();
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

        if (j < gt_count) {
            image_files.push_back(image_files_tmp[i]);
            times.push_back(time);
            poses_gt.push_back(pose_gt);
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <kitti_path>" << endl;
        return 1;
    }
    string seq = "2013_05_28_drive_" + string(argv[1]) + "_sync";
    bool is_02 = stoi(argv[1]) == 2;
    string kitti_path = "~/dataset/kitti-360/";

    string upnpl_out_file = kitti_path + seq + "/upnpl" + ".txt";
    string cv_epnp_out_file = kitti_path + seq + "/cv_epnp" + ".txt";
    string upnp_out_file = kitti_path + seq + "/upnp" + ".txt";
    string upnpl_points_out_file = kitti_path + seq + "/upnpl_points" + ".txt";
    string upnpl_lines_out_file = kitti_path + seq + "/upnpl_lines" + ".txt";
    string gpnp_out_file = kitti_path + seq + "/gpnp" + ".txt";
    string gt_out_file = kitti_path + seq + "/gt" + ".txt";

    string stereo_file = kitti_path + "calibration/perspective.txt";
    string cam2_file = kitti_path + "calibration/image_02.yaml";
    string cam3_file = kitti_path + "calibration/image_03.yaml";
    string extrinsic_file = kitti_path + "calibration/calib_cam_to_pose.txt";
    string gt_file = kitti_path + "data_poses/" + seq + "/poses.txt";

    string time_file = kitti_path + "KITTI-360/data_2d_raw/" + seq +
                       "/image_00/timestamps.txt";
    string image0_path =
        kitti_path + "KITTI-360/data_2d_raw/" + seq + "/image_00/data_rect/";
    string image1_path =
        kitti_path + "KITTI-360/data_2d_raw/" + seq + "/image_01/data_rect/";
    string image2_path =
        kitti_path + "KITTI-360/data_2d_raw/" + seq + "/image_02/data_rgb/";
    string image3_path =
        kitti_path + "KITTI-360/data_2d_raw/" + seq + "/image_03/data_rgb/";

    vector<vector<string>> image_files(4);
    vector<vector<double>> times(4);
    vector<vector<Eigen::Isometry3d>> poses_gt(4);

    loadKittiImage(image0_path, time_file, gt_file, image_files[0], times[0],
                   poses_gt[0], is_02);
    loadKittiImage(image1_path, time_file, gt_file, image_files[1], times[1],
                   poses_gt[1], is_02);
    loadKittiImage(image2_path, time_file, gt_file, image_files[2], times[2],
                   poses_gt[2], is_02);
    loadKittiImage(image3_path, time_file, gt_file, image_files[3], times[3],
                   poses_gt[3], is_02);

    vector<Camera> cameras(4);
    loadKittiStereoCameraParameters(stereo_file, cameras[0], cameras[1]);
    loadKittiMeiCameraParameters(cam2_file, cam3_file, extrinsic_file,
                                 cameras[2], cameras[3]);

    vector<Eigen::Isometry3d> Tbc(4);
    for (int i = 0; i < 4; ++i) {
        Tbc[i] = cameras[i].T_bc;
    }

    string camera_file = kitti_path + seq + "/cam0/sensor.yaml";
    saveCameraParameters(camera_file, cameras[0]);

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
    for (int i = 0; i < image_files[0].size() - 91; ++i) {
        string matlab_data =
            kitti_path + seq + "/data/" + "data_" + to_string(i) + ".txt";

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
                             points_cam, lines_cam, 1000);
        } catch (const std::exception &e) {
            cerr << "Error generating data for frame " << i << ": " << e.what()
                 << endl;
            continue;
        }
        cout << "Processed frame " << i << endl;
        cout << "point size: " << points_w.size()
             << ", line size: " << lines_w.size() << endl;
        saveDataForMatlab(points_w, points_sigma, lines_w, lines_sigma, uv_c,
                          lines_c, points_cam, lines_cam, matlab_data,
                          times[0][i + 1]);
        Twb_gt.push_back(poses_gt[0][i + 1]);
        times_save.push_back(times[0][i + 1]);
        Eigen::Isometry3d Tbw = Eigen::Isometry3d::Identity();
        double used_time = 0.0;
        Tbw = utils::myUPnPL(points_w, lines_w, uv_c, lines_c, points_cam,
                             lines_cam, cameras, 4, used_time);
        avg_time_upnpl += used_time;

        Twb_upnpl.push_back(Tbw.inverse());

        Tbw = utils::cv_EPnP(points_w, uv_c, points_cam, cameras, used_time);
        avg_time_cv += used_time;
        Twb_cv.push_back(Tbw.inverse());

        Tbw =
            utils::opengv_UPnP(points_w, uv_c, points_cam, cameras, used_time);
        avg_time_upnp += used_time;
        Twb_upnp.push_back(Tbw.inverse());

        Tbw =
            utils::opengv_GPnP(points_w, uv_c, points_cam, cameras, used_time);
        Twb_gpnp.push_back(Tbw.inverse());

        vector<int> points_cam_tmp;
        vector<Eigen::Vector3d> points_w_tmp;
        vector<Eigen::Vector3d> uv_c_tmp;

        Tbw = utils::myUPnPL(points_w_tmp, lines_w, uv_c_tmp, lines_c,
                             points_cam_tmp, lines_cam, cameras, 4, used_time);
        Twb_upnpl_lines.push_back(Tbw.inverse());

        lines_w.clear();
        lines_c.clear();
        lines_cam.clear();

        Tbw = utils::myUPnPL(points_w, lines_w, uv_c, lines_c, points_cam,
                             lines_cam, cameras, 4, used_time);
        Twb_upnpl_points.push_back(Tbw.inverse());
        avg_time_upnpl_points += used_time;
    }

    avg_time_upnpl /= Twb_upnpl.size();
    avg_time_cv /= Twb_cv.size();
    avg_time_upnp /= Twb_upnp.size();
    avg_time_upnpl_points /= Twb_upnpl_points.size();

    cout << "Average UPnPL time: " << avg_time_upnpl << " ms" << endl;
    cout << "Average OpenCV EPnP time: " << avg_time_cv << " ms" << endl;
    cout << "Average OpenGV UPnP time: " << avg_time_upnp << " ms" << endl;
    cout << "Average UPnPL only points time: " << avg_time_upnpl_points << " ms"
         << endl;

    saveEurocTraejectory(upnpl_out_file, Twb_upnpl, times_save);
    saveEurocTraejectory(cv_epnp_out_file, Twb_cv, times_save);
    saveEurocTraejectory(upnp_out_file, Twb_upnp, times_save);
    saveEurocTraejectory(upnpl_points_out_file, Twb_upnpl_points, times_save);
    saveEurocTraejectory(upnpl_lines_out_file, Twb_upnpl_lines, times_save);
    saveEurocTraejectory(gpnp_out_file, Twb_gpnp, times_save);
    saveEurocTraejectory(gt_out_file, Twb_gt, times_save);
}
