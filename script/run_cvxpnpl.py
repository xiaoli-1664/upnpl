import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
from cvxpnpl import pnpl
import yaml
import os
from pathlib import Path


def project_points(pts_2d, K):
    for i in range(len(pts_2d)):
        pts_2d[i] = np.dot(K, pts_2d[i])
        pts_2d[i] /= pts_2d[i][2]
    return pts_2d[:, :2]


def project_lines(line_2d, K):
    for i in range(len(line_2d)):
        line_2d[i][0] = np.dot(K, line_2d[i][0])
        line_2d[i][1] = np.dot(K, line_2d[i][1])
        line_2d[i][0] /= line_2d[i][0][2]
        line_2d[i][1] /= line_2d[i][1][2]
    return line_2d[:, :2, :2]


def load_sim_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    idx = 0
    time = float(lines[idx].strip())
    idx = 1
    n_pts = int(lines[idx].strip())
    idx += 1
    pts_3d = []
    pts_2d = []
    for _ in range(n_pts):
        vals = list(map(float, lines[idx].strip(" ").split()))
        pts_3d.append(vals[:3])
        pts_2d.append(vals[3:6])
        idx += 1

    n_lines = int(lines[idx].strip())
    idx += 1
    line_3d = []
    line_2d = []
    for _ in range(n_lines):
        vals = list(map(float, lines[idx].strip(" ").split()))
        line_3d.append(vals[0:3])
        line_3d.append(vals[3:6])
        line_2d.append([vals[6:9], vals[9:12]])
        idx += 1

    return (
        time,
        np.array(pts_3d),
        np.array(pts_2d),
        np.array(line_3d),
        np.array(line_2d),
    )


def save_trajectory_euroc(R_list, t_list, time_list, output_path):
    with open(output_path, 'w') as f:
        for R_mat, t_vec, time in zip(R_list, t_list, time_list):
            # 将旋转矩阵转换为四元数（w在最后）
            quat = R.from_matrix(R_mat).as_quat()  # x, y, z, w
            qx, qy, qz, qw = quat  # 保持为 xyzw 顺序
            tx, ty, tz = t_vec
            timestamp_ns = int(time * 1e9)
            f.write(
                f"{timestamp_ns},{tx:.6f},{ty:.6f},{tz:.6f},{qx:.6f},{qy:.6f},{qz:.6f},{qw:.6f}\n")


def load_camera_intrinsics(yaml_path, default_intrinsics=None):
    """
    从 YAML 文件加载相机内参，如果文件不存在则返回默认值。

    Args:
        yaml_path (str): YAML 文件路径
        default_intrinsics (list): 默认内参值 [fu, fv, cu, cv]

    Returns:
        list: 相机内参 [fu, fv, cu, cv]
    """
    # 默认内参（如果未提供）
    if default_intrinsics is None:
        default_intrinsics = [450, 450, 376, 240]

    # 检查文件是否存在
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        print(
            f"Warning: YAML file '{yaml_path}' not found. Using default intrinsics.")
        return 0, default_intrinsics, np.eye(4)

    # 读取 YAML 文件
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        # 提取内参
        intrinsics = data.get('intrinsics')
        if intrinsics is None:
            print("Warning: 'intrinsics' key not found in YAML. Using default values.")
            return 0, default_intrinsics, np.eye(4)

        Tbc = data['T_BS']['data']
        Tbc = np.array(Tbc).reshape(4, 4)

        # 检查内参格式是否正确
        if len(intrinsics) != 4:
            print(
                f"Warning: Expected 4 values in 'intrinsics', got {len(intrinsics)}. Using defaults.")
            return 0, default_intrinsics, np.eye(4)

        return 1, intrinsics, Tbc

    except Exception as e:
        print(f"Error loading YAML file: {e}. Using default intrinsics.")
        return 0, default_intrinsics, np.eye(4)


def main():
    parser = argparse.ArgumentParser(
        description="Run PNPL algorithm on sim data")
    # parser.add_argument("n_points", type=int, help="Number of points")
    # parser.add_argument("n_lines", type=int, help="Number of lines")
    # parser.add_argument("noise", type=float, help="Noise level or index")
    # parser.add_argument("is_dataset", type=int, help="")
    parser.add_argument("data_path", type=str,
                        help="Path to the data directory")
    parser.add_argument("count", type=int,
                        help="Number of data files to process")
    args = parser.parse_args()

    R_list = []
    t_list = []
    times = []

    data_path = args.data_path
    output = "cvxpnpl.txt"
    yaml_file = data_path + "cam0/sensor.yaml"
    sim, intrinsics, Tbc = load_camera_intrinsics(yaml_file)
    K = np.array([[intrinsics[0], 0, intrinsics[2]],
                  [0, intrinsics[1], intrinsics[3]], [0, 0, 1]])

    index = 0
    for i in range(0, args.count):
        filename = os.path.join(data_path, f"data/data_{index}.txt")

        while not os.path.exists(filename):
            index = index + 1
            filename = os.path.join(data_path, f"data/data_{index}.txt")

        index = index + 1

        time, pts_3d, pts_2d, line_3d, line_2d = load_sim_data(filename)

        try:
            pts_2d = project_points(pts_2d, K)
            line_2d = project_lines(line_2d, K)

            poses = pnpl(pts_2d=pts_2d, line_2d=line_2d,
                         pts_3d=pts_3d, line_3d=line_3d, K=K, max_iters=2500)
            R, t = poses[0]
            Tcw = np.eye(4)
            Tcw[:3, :3] = R
            Tcw[:3, 3] = t
            Tbw = Tbc @ Tcw
            Twb = np.linalg.inv(Tbw)
            if sim == 1:
                R = Twb[:3, :3]
                t = Twb[:3, 3]
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            R = np.eye(3)
            t = np.zeros(3)

        R_list.append(R)
        t_list.append(t)
        times.append(time)
    save_trajectory_euroc(R_list, t_list, times, data_path + output)
    print(f"Trajectory saved to {data_path + output}")


if __name__ == "__main__":
    main()
