
#include "utils.h"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
    bool save = true;
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <num_points> <num_lines> <noise_std>"
             << endl;
        return 1;
    }
    int point_num = atoi(argv[1]);
    int line_num = atoi(argv[2]);
    int noise_std = atoi(argv[3]);
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

    vector<Eigen::Isometry3d> Tbw_upnpl_4cam;
    vector<Eigen::Isometry3d> Tbw_upnpl_3cam;
    vector<Eigen::Isometry3d> Tbw_upnpl_2cam;
    vector<Eigen::Isometry3d> Tbw_upnpl_1cam;
    vector<Eigen::Isometry3d> Tbw_epnp;
    vector<Eigen::Isometry3d> Tbw_gt;
    vector<Eigen::Isometry3d> Tbw_upnp;
    vector<double> times;

    int iter = 10000;
    for (int i = 0; i < iter; ++i) {
        string simulated_data =
            output_dir + "data/simulated_data_" + to_string(i) + ".txt";
        times.push_back(i * 0.1); // Simulate timestamps
        utils::Simulator simulator(4, point_num, line_num, noise_std);
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

        utils::saveDataForMatlab(points_w, lines_w, uv_c, lines_c, points_cam,
                                 lines_cam, simulated_data, times.back());

        Eigen::Isometry3d T_bw_est_isometry;
        T_bw_est_isometry =
            myUPnPL(points_w, lines_w, uv_c, lines_c, points_cam, lines_cam,
                    simulator.cameras_, 4);
        Tbw_upnpl_4cam.push_back(T_bw_est_isometry);

        T_bw_est_isometry =
            myUPnPL(points_w, lines_w, uv_c, lines_c, points_cam, lines_cam,
                    simulator.cameras_, 3);
        Tbw_upnpl_3cam.push_back(T_bw_est_isometry);

        T_bw_est_isometry =
            myUPnPL(points_w, lines_w, uv_c, lines_c, points_cam, lines_cam,
                    simulator.cameras_, 2);
        Tbw_upnpl_2cam.push_back(T_bw_est_isometry);

        T_bw_est_isometry =
            myUPnPL(points_w, lines_w, uv_c, lines_c, points_cam, lines_cam,
                    simulator.cameras_, 1);
        Tbw_upnpl_1cam.push_back(T_bw_est_isometry);

        Eigen::Isometry3d T_bw_upnp;
        T_bw_upnp = opengv_UPnP(points_w, lines_w, uv_c, normals_c, points_cam,
                                lines_cam, simulator.cameras_);
        Tbw_upnp.push_back(T_bw_upnp);

        // Use OpenCV's EPnP to verify the results
        Eigen::Isometry3d T_bw_epnp =
            cv_EPnP(points_w, lines_w, uv_c, normals_c, points_cam, lines_cam,
                    simulator.cameras_);
        Tbw_epnp.push_back(T_bw_epnp);

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

    if (save) {
        utils::saveEurocTraejectory(upnpl_4cam_output_file, Tbw_upnpl_4cam,
                                    times);
        utils::saveEurocTraejectory(upnpl_3cam_output_file, Tbw_upnpl_3cam,
                                    times);
        utils::saveEurocTraejectory(upnpl_2cam_output_file, Tbw_upnpl_2cam,
                                    times);
        utils::saveEurocTraejectory(upnpl_1cam_output_file, Tbw_upnpl_1cam,
                                    times);
        utils::saveEurocTraejectory(epnp_output_file, Tbw_epnp, times);
        utils::saveEurocTraejectory(gt_output_file, Tbw_gt, times);
        utils::saveEurocTraejectory(upnp_output_file, Tbw_upnp, times);
    }

    cout << "Simulation completed." << endl;

    return 0;
}
