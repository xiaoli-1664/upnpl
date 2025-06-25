import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
from cvxpnpl import pnpl

K = np.array([[450, 0, 376], [0, 450, 240], [0, 0, 1]])


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


def main():
    parser = argparse.ArgumentParser(
        description="Run PNPL algorithm on sim data")
    parser.add_argument("n_points", type=int, help="Number of points")
    parser.add_argument("n_lines", type=int, help="Number of lines")
    parser.add_argument("noise", type=float, help="Noise level or index")
    args = parser.parse_args()

    R_list = []
    t_list = []
    times = []

    data_path = f"/home/ljj/source_code/PL-MCVO/third-party/upnpl/simulated/{args.n_points}_{args.n_lines}_{int(args.noise)}/"
    output = "cvxpnpl.txt"

    for i in range(0, 10000):
        filename = data_path + f"data/simulated_data_{i}.txt"

        time, pts_3d, pts_2d, line_3d, line_2d = load_sim_data(filename)

        pts_2d = project_points(pts_2d, K)
        line_2d = project_lines(line_2d, K)

        poses = pnpl(pts_2d=pts_2d, line_2d=line_2d,
                     pts_3d=pts_3d, line_3d=line_3d, K=K, max_iters=2500)

        if poses:
            R, t = poses[0]
        else:
            R = np.eye(3)
            t = np.zeros(3)
            print("No pose found!")

        R_list.append(R)
        t_list.append(t)
        times.append(time)
    save_trajectory_euroc(R_list, t_list, times, data_path + output)
    print(f"Trajectory saved to {output}")


if __name__ == "__main__":
    main()
