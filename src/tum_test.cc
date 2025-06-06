#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "rapidcsv.h"

using namespace std;

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
    string tum_path = "/home/ljj/dataset/tum/room1/";
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

    return 0;
}
