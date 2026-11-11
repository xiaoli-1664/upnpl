#include "UPnPL.h"
#include "simulation_data.h"
#ifdef UPNPL_TEST_RANSAC
#include "utils.h"
#endif

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>

int main(int argc, char **argv) {
    using namespace simulation;
    const std::string mode = argc > 1 ? argv[1] : "--full";
    const bool check = mode == "--check";
    const bool sampling = mode == "--sampling";
    const bool ransac = mode == "--ransac";
    if (!check && !sampling && !ransac && mode != "--full") return 2;
#ifndef UPNPL_TEST_RANSAC
    if (ransac) return 2;
#endif
    const int trials = argc > 2 ? std::stoi(argv[2]) : (check ? 3 : 10);
    const int repeats = argc > 3 ? std::stoi(argv[3]) : (sampling || ransac ? 5 : 1);
    std::vector<Config> configs;
    if (sampling || ransac) {
        configs = {{"points_4cam", 4, 48, 0}, {"mixed_4cam", 4, 32, 16}};
    } else {
        configs = {{"points_1cam", 1, 16, 0}, {"points_4cam", 4, 16, 0},
                   {"mixed_1cam", 1, 12, 8}, {"mixed_4cam", 4, 12, 8},
                   {"mixed_no_normalize", 4, 12, 8, -1, false},
                   {"mixed_scale_100", 4, 12, 8, -1, true, 100},
                   {"mixed_bearing_scale_7", 4, 12, 8, -1, true, 1, 7},
                   {"h_zero", 4, 16, 0, 0},
                   {"h_near_1e-12", 4, 16, 0, 1e-12},
                   {"h_near_1e-6", 4, 16, 0, 1e-6},
                   {"mixed_h_zero", 4, 12, 8, 0},
                   {"lines_only", 4, 0, 16}};
    }
    std::cout << std::setprecision(17)
              << "mode,case,seed,repeat,sigma,outliers,input_hash,rotation_deg,translation_relative,point_residual,finite,pose_valid,success"
              << ",r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz\n";
    int failures = 0;
    for (const auto &config : configs) {
        for (int trial = 0; trial < trials; ++trial) {
            const unsigned seed = 20261111 + trial;
            for (double sigma : {0.0, 0.002}) {
                const std::vector<int> outlier_counts = check ? std::vector<int>{0} :
                    (ransac ? std::vector<int>{0, 10} : std::vector<int>{0, 1});
                for (int outliers : outlier_counts) {
                    const Scene scene = makeScene(config, seed, sigma, outliers);
                    for (int repeat = 0; repeat < repeats; ++repeat) {
                        // Eigen::setRandom uses rand(). random_device-based
                        // scoring/RANSAC remains unchanged and is repeated.
                        std::srand(seed + repeat);
                        Matrix3d R = Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
                        Vector3d t = Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
#ifdef UPNPL_TEST_RANSAC
                        if (ransac) {
                            std::vector<utils::Camera> cameras(config.cameras);
                            for (int c = 0; c < config.cameras; ++c) {
                                cameras[c].T_bc = Eigen::Isometry3d::Identity();
                                cameras[c].T_bc.linear() = scene.Rbc[c];
                                cameras[c].T_bc.translation() = scene.tbc[c];
                            }
                            double milliseconds;
                            const auto T = utils::myUPnPL_RANSAC(
                                scene.points, scene.lines, scene.bearings, scene.endpoints,
                                scene.point_cameras, scene.line_cameras, cameras, config.cameras,
                                config.lines ? 4 : 8, config.lines ? 4 : 0,
                                60, 0.015, milliseconds);
                            R = T.linear();
                            t = T.translation();
                        } else
#endif
                        {
                            UPnPL::UPnPL solver(config.normalize);
                            solver.solveMain(scene.points, scene.lines, scene.bearings,
                                             scene.endpoints, scene.point_cameras,
                                             scene.line_cameras, scene.Rbc, scene.tbc, R, t);
                        }
                        const bool finite = R.allFinite() && t.allFinite();
                        const double rotation = finite ? Eigen::AngleAxisd(R * scene.R.transpose()).angle()
                            * 180 / 3.14159265358979323846 : std::numeric_limits<double>::infinity();
                        const double translation = (t - scene.t).norm() / config.coordinate_scale;
                        const bool valid = finite && (R.transpose() * R - Matrix3d::Identity()).norm() < 1e-8
                            && std::abs(R.determinant() - 1) < 1e-8;
                        // Broad practical success thresholds, not a claim of
                        // improved accuracy. Translation uses the scene's units.
                        const bool success = valid && rotation < 1 && translation < 0.1;
                        std::cout << mode.substr(2) << ',' << config.name << ',' << seed << ','
                                  << repeat << ',' << sigma << ',' << outliers << ',' << fingerprint(scene)
                                  << ',' << rotation << ',' << translation << ','
                                  << pointResidual(scene, R, t) << ',' << finite << ',' << valid
                                  << ',' << success;
                        for (int row = 0; row < 3; ++row)
                            for (int col = 0; col < 3; ++col) std::cout << ',' << R(row, col);
                        std::cout << ',' << t.x() << ',' << t.y() << ',' << t.z() << '\n';
                        if (check && (!success || (sigma == 0 && (rotation > 1e-5 || translation > 1e-6)))) {
                            ++failures;
                            std::cerr << "FAILED " << config.name << " seed=" << seed << " sigma=" << sigma
                                      << " rotation=" << rotation << " translation=" << translation << '\n';
                        }
                    }
                }
            }
        }
    }
    return check && failures ? 1 : 0;
}
