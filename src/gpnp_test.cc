#include "utils.h"
#include <yaml-cpp/yaml.h>

int load_camera_intrinsics(const std::string &yaml_path,
                           std::vector<double> &intrinsics, cv::Mat &Tbc,
                           const std::vector<double> &default_intrinsics = {
                               450.0, 450.0, 376.0, 240.0}) {
    // 设置默认内参
    intrinsics = default_intrinsics;
    // 设置默认 Tbc 为单位矩阵
    Tbc = cv::Mat::eye(4, 4, CV_64F);

    // 检查文件是否存在
    std::ifstream file(yaml_path);
    if (!file.good()) {
        return 0;
    }
    file.close(); // 关闭文件，因为 YAML::LoadFile 会重新打开

    try {
        YAML::Node config = YAML::LoadFile(yaml_path);

        // 提取内参
        if (config["intrinsics"]) {
            if (config["intrinsics"].IsSequence() &&
                config["intrinsics"].size() == 4) {
                intrinsics.clear();
                for (size_t i = 0; i < 4; ++i) {
                    intrinsics.push_back(config["intrinsics"][i].as<double>());
                }
            } else {
                std::cerr
                    << "Warning: 'intrinsics' key found but not a sequence of "
                       "4 doubles in YAML. Using default values."
                    << std::endl;
            }
        } else {
            std::cerr << "Warning: 'intrinsics' key not found in YAML. Using "
                         "default values."
                      << std::endl;
        }

        // 提取 Tbc (T_BS)
        if (config["T_BS"] && config["T_BS"]["data"]) {
            if (config["T_BS"]["data"].IsSequence() &&
                config["T_BS"]["data"].size() == 16) {
                std::vector<double> Tbc_data;
                for (size_t i = 0; i < 16; ++i) {
                    Tbc_data.push_back(config["T_BS"]["data"][i].as<double>());
                }
                Tbc = cv::Mat(4, 4, CV_64F, Tbc_data.data())
                          .clone(); // 使用 clone 确保数据独立
            } else {
                std::cerr
                    << "Warning: 'T_BS.data' found but not a sequence of 16 "
                       "doubles in YAML. Using identity matrix for Tbc."
                    << std::endl;
            }
        } else {
            std::cerr << "Warning: 'T_BS' or 'T_BS.data' key not found in "
                         "YAML. Using identity matrix for Tbc."
                      << std::endl;
        }

        return 1; // 成功加载
    } catch (const YAML::BadFile &e) {
        std::cerr << "Error: Could not open or parse YAML file '" << yaml_path
                  << "': " << e.what() << ". Using default intrinsics."
                  << std::endl;
        return 0;
    } catch (const YAML::Exception &e) {
        std::cerr << "Error loading YAML file: " << e.what()
                  << ". Using default intrinsics." << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "An unexpected error occurred: " << e.what()
                  << ". Using default intrinsics." << std::endl;
        return 0;
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    string data_path = argv[1];
    int iter_num = atoi(argv[2]);

    string output_dir = data_path + "gpnp.txt";
    string data_dir = data_path + "data/";
    string yaml_file = data_path + "cam0/sensor.yaml";

    for (int i = 0; i < iter_num; ++i) {
        string input_file = data_dir + "data_" + to_string(i) + ".txt";
    }
}
