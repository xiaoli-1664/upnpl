#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse


def read_euroc_format(filepath):
    """
    读取 EuRoC/TUM 格式的轨迹文件。
    格式: timestamp tx ty tz qx qy qz qw
    返回: 位姿列表（每个位姿是 [tx, ty, tz, qx, qy, qz, qw]）
    """
    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            data = [float(x) for x in line.split(',')]
            poses.append(np.array(data[1:]))  # 忽略时间戳
    return np.array(poses)


def calculate_ate_translation_error(gt_poses, est_poses, threshold=1000):
    """计算绝对轨迹误差 (ATE) 的平移部分 (无对齐)，跳过 NaN 和数值过大的帧。"""
    est_trans = est_poses[:, :3]
    gt_trans = gt_poses[:, :3]

    # valid_mask = (
    #     ~np.isnan(est_trans).any(axis=1) &
    #     (np.linalg.norm(est_trans, axis=1) < threshold)
    # )

    errors = np.linalg.norm(gt_trans - est_trans, axis=1)
    valid_mask = (~np.isnan(errors)) & (errors < threshold)
    errors = errors[valid_mask]

    max_error = np.max(errors)
    max_index = np.argmax(errors)
    print(
        f"最大平移误差: {max_error:.4f} 米 (发生在索引 {np.where(valid_mask)[0][max_index]})")
    return np.mean(errors)


def calculate_are_rotation_error(gt_poses, est_poses):
    """计算绝对旋转误差 (ARE) (无对齐)。"""
    rotation_errors_deg = []
    for i in range(len(gt_poses)):
        R_true = R.from_quat(gt_poses[i, 3:]).as_matrix()
        try:
            R_est = R.from_quat(est_poses[i, 3:]).as_matrix()
        except ValueError:
            R_est = np.eye(3)  # 如果估计的四元数无效，使用单位矩阵
        R_error = R_true.T @ R_est
        trace = np.trace(R_error)
        arg = np.clip(0.5 * (trace - 1.0), -1.0, 1.0)
        angle_rad = np.arccos(arg)
        rotation_errors_deg.append(np.rad2deg(angle_rad))
    return np.mean(rotation_errors_deg)


def main():
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="计算两条轨迹的ATE, ARE, 和 RPE 误差 (无对齐版本)。\n"
                    "文件格式应为: timestamp tx ty tz qx qy qz qw",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('groundtruth_file', type=str, help='真值轨迹文件的路径')
    parser.add_argument('estimated_file', type=str, help='估计轨迹文件的路径')
    args = parser.parse_args()

    try:
        # --- 读取数据 ---
        gt_poses = read_euroc_format(args.groundtruth_file)
        est_poses = read_euroc_format(args.estimated_file)

        if len(gt_poses) == 0 or len(est_poses) == 0:
            raise ValueError("文件为空或格式不正确。")
        if len(gt_poses) != len(est_poses):
            raise ValueError("文件行数不匹配，请确保两个文件中的位姿一一对应。")

        print(f"\n成功读取文件:")
        print(f"  真值轨迹: '{args.groundtruth_file}' ({len(gt_poses)} 个位姿)")
        print(f"  估计轨迹: '{args.estimated_file}' ({len(est_poses)} 个位姿)")
        print("-" * 30)

        # --- 计算各项误差 (均无对齐) ---

        # 1. ATE (平移)
        ate_mae = calculate_ate_translation_error(gt_poses, est_poses)
        print("1. 平均绝对平移误差 (ATE - Unaligned)")
        # print(f"   描述: 衡量原始轨迹间的直接平移距离，结果受初始位姿和累积漂移共同影响。")
        print(f"   结果: {ate_mae:.4f} 米\n")

        # 2. ARE (旋转)
        are_mae = calculate_are_rotation_error(gt_poses, est_poses)
        print("2. 平均绝对旋转误差 (ARE - Unaligned)")
        # print(f"   描述: 衡量原始轨迹间每个时间点上姿态的直接旋转误差。")
        print(f"   结果: {are_mae:.4f} 度\n")

        # # 3. RPE (旋转)
        # rpe_mae_rot = calculate_rpe_rotation_error(gt_poses, est_poses)
        # print("3. 平均相对旋转误差 (RPE)")
        # print(f"   描述: 衡量连续位姿之间相对运动的旋转精度（局部精度）。")
        # print(f"   结果: {rpe_mae_rot:.4f} 度\n")

    except FileNotFoundError as e:
        print(f"\n[错误] 文件未找到: {e.filename}")
    except Exception as e:
        print(f"\n[错误] 发生异常: {e}")


if __name__ == "__main__":
    main()
