
#include "utils.h"
#include <Eigen/src/Geometry/Transform.h>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
    bool save = true;
    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " <num_points> <num_lines> <noise_std> <outlier_ratio>"
             << endl;
        return 1;
    }
    int point_num = atoi(argv[1]);
    int line_num = atoi(argv[2]);
    int noise_std = atoi(argv[3]);
    double outlier_ratio = atof(argv[4]);

    int iter = atoi(argv[5]);
    string output_dir = "./simulated/" + to_string(point_num) + "_" +
                        to_string(line_num) + "_" + to_string(noise_std) + "/";
    string upnpl_4cam_output_file =
        output_dir + "upnpl_4cam_simulated_trajectory.txt";
    string upnpl_3cam_output_file =
        output_dir + "upnpl_3cam_simulated_trajectory.txt";
    string upnpl_2cam_output_file =
        output_dir + "upnpl_2cam_simulated_trajectory.txt";
    string upnpl_1cam_output_file =
        output_dir + "upnpl_1cam_simulated_trajectory.txt";
    string epnp_output_file = output_dir + "epnp_simulated_trajectory.txt";
    string upnp_output_file = output_dir + "upnp_simulated_trajectory.txt";
    string gt_output_file = output_dir + "gt_simulated_trajectory.txt";
    string gpnp_output_file = output_dir + "gpnp.txt";
    string upnpl_ransac1_output_file =
        output_dir + "upnpl_ransac1_simulated_trajectory.txt";
    string upnpl_ransac2_output_file =
        output_dir + "upnpl_ransac2_simulated_trajectory.txt";
    string upnpl_ransac4_output_file =
        output_dir + "upnpl_ransac4_simulated_trajectory.txt";

    vector<Eigen::Isometry3d> Tbw_upnpl_4cam;
    vector<Eigen::Isometry3d> Tbw_upnpl_3cam;
    vector<Eigen::Isometry3d> Tbw_upnpl_2cam;
    vector<Eigen::Isometry3d> Tbw_upnpl_1cam;
    vector<Eigen::Isometry3d> Tbw_epnp;
    vector<Eigen::Isometry3d> Tbw_gt;
    vector<Eigen::Isometry3d> Tbw_upnp;
    vector<Eigen::Isometry3d> Tbw_gpnp;
    vector<Eigen::Isometry3d> Tbw_upnpl_ransac1;
    vector<Eigen::Isometry3d> Tbw_upnpl_ransac2;
    vector<Eigen::Isometry3d> Tbw_upnpl_ransac4;
    vector<double> times;

    double avg_upupl_1cam = 0.0;
    double avg_upupl_2cam = 0.0;
    double avg_upupl_3cam = 0.0;
    double avg_upupl_4cam = 0.0;
    double avg_upnpl_points = 0.0;
    double avg_epnp = 0.0;
    double avg_upnp = 0.0;
    double avg_gpnp = 0.0;
    for (int i = 0; i < iter; ++i) {
        string simulated_data =
            output_dir + "data/data_" + to_string(i) + ".txt";
        times.push_back(i * 0.1); // Simulate timestamps
        utils::Simulator simulator(4, point_num, line_num, noise_std, outlier_ratio);
        simulator.setupCameras();

        Tbw_gt.push_back(simulator.T_bw_gt_);

        simulator.generateScene();
        vector<Eigen::Vector3d> points_w;
        vector<Eigen::VectorXd> lines_w;
        vector<Eigen::Vector3d> uv_c;
        vector<Eigen::Vector3d> normals_c;
        vector<Eigen::VectorXd> lines_c;
        vector<int> points_cam;
        vector<int> lines_cam;
        simulator.generateData(points_w, lines_w, uv_c, normals_c, lines_c,
                               points_cam, lines_cam);
        // cout << "Generated " << points_w.size() << " points and " <<
        // lines_w.size()
        //      << " lines." << endl;
        double noise_std_dev = simulator.noise_std_;
        double sigma = noise_std_dev / simulator.cameras_[0].fx;
        vector<double> points_sigma(points_w.size(), sigma * sigma);
        vector<double> lines_sigma(lines_w.size(), sigma * sigma);

        utils::saveDataForMatlab(points_w, points_sigma, lines_w, lines_sigma,
                                 uv_c, lines_c, points_cam, lines_cam,
                                 simulated_data, times.back());

        double used_time = 0.0;
        Eigen::Isometry3d T_bw_est_isometry;
        T_bw_est_isometry =
            myUPnPL(points_w, lines_w, uv_c, lines_c, points_cam, lines_cam,
                    simulator.cameras_, 4, used_time);
        Tbw_upnpl_4cam.push_back(T_bw_est_isometry);
        avg_upupl_4cam += used_time;

        T_bw_est_isometry =
            myUPnPL(points_w, lines_w, uv_c, lines_c, points_cam, lines_cam,
                    simulator.cameras_, 3, used_time);
        Tbw_upnpl_3cam.push_back(T_bw_est_isometry);
        avg_upupl_3cam += used_time;

        T_bw_est_isometry =
            myUPnPL(points_w, lines_w, uv_c, lines_c, points_cam, lines_cam,
                    simulator.cameras_, 2, used_time);
        Tbw_upnpl_2cam.push_back(T_bw_est_isometry);
        avg_upupl_2cam += used_time;

        T_bw_est_isometry =
            myUPnPL(points_w, lines_w, uv_c, lines_c, points_cam, lines_cam,
                    simulator.cameras_, 1, used_time);
        Tbw_upnpl_1cam.push_back(T_bw_est_isometry);
        avg_upupl_1cam += used_time;

        Eigen::Isometry3d T_ransac1 = utils::myUPnPL_RANSAC(
            points_w, lines_w, uv_c, lines_c, points_cam, lines_cam,
            simulator.cameras_, 1, 2, 2, 100, 0.1, used_time
        );
        Tbw_upnpl_ransac1.push_back(T_ransac1);

        Eigen::Isometry3d T_ransac2 = utils::myUPnPL_RANSAC(
            points_w, lines_w, uv_c, lines_c, points_cam, lines_cam,
            simulator.cameras_, 2, 2, 2, 100, 0.1, used_time
        );
        Tbw_upnpl_ransac2.push_back(T_ransac2);

        Eigen::Isometry3d T_ransac4 = utils::myUPnPL_RANSAC(
            points_w, lines_w, uv_c, lines_c, points_cam, lines_cam,
            simulator.cameras_, 4, 2, 2, 100, 0.1, used_time
        );
        Tbw_upnpl_ransac4.push_back(T_ransac4);

        lines_c.clear();
        lines_cam.clear();
        lines_w.clear();

        T_bw_est_isometry =
            myUPnPL(points_w, lines_w, uv_c, lines_c, points_cam, lines_cam,
                    simulator.cameras_, 4, used_time);
        avg_upnpl_points += used_time;

        Eigen::Isometry3d T_bw_upnp;
        T_bw_upnp = opengv_UPnP(points_w, uv_c, points_cam, simulator.cameras_,
                                used_time);
        Tbw_upnp.push_back(T_bw_upnp);
        avg_upnp += used_time;

        Eigen::Isometry3d T_bw_upnpl_gpnp;
        T_bw_upnpl_gpnp = opengv_GPnP(points_w, uv_c, points_cam,
                                      simulator.cameras_, used_time);
        Tbw_gpnp.push_back(T_bw_upnpl_gpnp);
        avg_gpnp += used_time;

        // Use OpenCV's EPnP to verify the results
        Eigen::Isometry3d T_bw_epnp =
            cv_EPnP(points_w, uv_c, points_cam, simulator.cameras_, used_time);
        Tbw_epnp.push_back(T_bw_epnp);
        avg_epnp += used_time;

        

        if (save)
            continue;
        cout << "Estimated R_bw:\n" << T_bw_est_isometry.linear() << endl;
        cout << "Estimated t_bw:\n"
             << T_bw_est_isometry.translation().transpose() << endl;

        cout << "Ground truth R_bw:\n" << simulator.T_bw_gt_.linear() << endl;
        cout << "Ground truth t_bw:\n"
             << simulator.T_bw_gt_.translation().transpose() << endl;

        cout << "EPnP Estimated R_bw:\n" << T_bw_epnp.linear() << endl;
        cout << "EPnP Estimated t_bw:\n"
             << T_bw_epnp.translation().transpose() << endl;

        cout << "UPnP Estimated R_bw:\n" << T_bw_upnp.linear() << endl;
        cout << "UPnP Estimated t_bw:\n"
             << T_bw_upnp.translation().transpose() << endl;
    }

    avg_upupl_1cam /= iter;
    avg_upupl_2cam /= iter;
    avg_upupl_3cam /= iter;
    avg_upupl_4cam /= iter;
    avg_upnpl_points /= iter;
    avg_epnp /= iter;
    avg_upnp /= iter;
    avg_gpnp /= iter;

    // cout << "Average time for UPnPL with 1 camera: " << avg_upupl_1cam << "
    // ms"
    //      << endl;
    // cout << "Average time for UPnPL with 2 cameras: " << avg_upupl_2cam << "
    // ms"
    //      << endl;
    // cout << "Average time for UPnPL with 3 cameras: " << avg_upupl_3cam << "
    // ms"
    //      << endl;
    cout << "Average time for UPnPL with 4 cameras: " << avg_upupl_4cam << " ms"
         << endl;
    cout << "Average time for UPnPL with points: " << avg_upnpl_points << " ms"
         << endl;
    cout << "Average time for OpenCV EPnP: " << avg_epnp << " ms" << endl;
    cout << "Average time for OpenGV UPnP: " << avg_upnp << " ms" << endl;
    cout << "Average time for OpenGV GPnP: " << avg_gpnp << " ms" << endl;

    if (save) {
        utils::saveEurocTraejectory(upnpl_4cam_output_file, Tbw_upnpl_4cam,
                                    times);
        utils::saveEurocTraejectory(upnpl_3cam_output_file, Tbw_upnpl_3cam,
                                    times);
        utils::saveEurocTraejectory(upnpl_2cam_output_file, Tbw_upnpl_2cam,
                                    times);
        utils::saveEurocTraejectory(upnpl_1cam_output_file, Tbw_upnpl_1cam,
                                    times);
        utils::saveEurocTraejectory(gpnp_output_file, Tbw_gpnp, times);
        utils::saveEurocTraejectory(epnp_output_file, Tbw_epnp, times);
        utils::saveEurocTraejectory(gt_output_file, Tbw_gt, times);
        utils::saveEurocTraejectory(upnp_output_file, Tbw_upnp, times);
        utils::saveEurocTraejectory(upnpl_ransac1_output_file,
                                    Tbw_upnpl_ransac1, times);
        utils::saveEurocTraejectory(upnpl_ransac2_output_file,
                                    Tbw_upnpl_ransac2, times);
        utils::saveEurocTraejectory(upnpl_ransac4_output_file,
                                    Tbw_upnpl_ransac4, times);
    }

    cout << "Simulation completed." << endl;

    return 0;
}
