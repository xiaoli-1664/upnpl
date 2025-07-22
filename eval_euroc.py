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
            # 支持逗号或空格分隔
            parts = line.replace(',', ' ').split()
            data = [float(x) for x in parts]
            poses.append(np.array(data[1:]))  # 忽略时间戳
    return np.array(poses)


def calculate_ate_translation_error(gt_poses, est_poses, threshold=1000):
    """
    计算绝对轨迹误差 (ATE) 的平移部分 (无对齐)，并返回用于筛选的掩码。
    """
    est_trans = est_poses[:, :3]
    gt_trans = gt_poses[:, :3]
    # est_norms = np.linalg.norm(est_trans, axis=1)

    errors = np.linalg.norm(gt_trans - est_trans, axis=1)
    valid_mask = ((~np.isnan(errors)) & (
        errors < threshold))
    valid_errors = errors[valid_mask]

    if len(valid_errors) == 0:
        print("警告: 在给定的阈值下，没有找到有效的平移数据点。")
        return np.nan, valid_mask

    max_error = np.max(valid_errors)
    # 找到最大误差在有效集中的索引
    max_index_in_valid_set = np.argmax(valid_errors)
    # 映射回原始数据集的索引
    original_index = np.where(valid_mask)[0][max_index_in_valid_set]

    print(f"最大平移误差: {max_error:.4f} 米 (发生在原始索引 {original_index})")

    # 返回平均误差和掩码
    return np.mean(valid_errors), valid_mask


def calculate_are_rotation_error(gt_poses, est_poses, valid_mask):
    """
    计算绝对旋转误差 (ARE) (无对齐)，仅使用通过掩码筛选的有效子集。
    """
    # 应用掩码筛选位姿
    gt_poses_filtered = gt_poses[valid_mask]
    est_poses_filtered = est_poses[valid_mask]

    if len(gt_poses_filtered) == 0:
        return np.nan

    rotation_errors_deg = []
    for i in range(len(gt_poses_filtered)):
        # 提取有效的四元数
        gt_quat = gt_poses_filtered[i, 3:]
        est_quat = est_poses_filtered[i, 3:]

        # 检查四元数是否归一化，以防万一
        if np.linalg.norm(gt_quat) < 1e-6 or np.linalg.norm(est_quat) < 1e-6:
            continue  # 跳过零四元数

        try:
            R_true = R.from_quat(gt_quat).as_matrix()
            R_est = R.from_quat(est_quat).as_matrix()
        except ValueError:
            # 如果估计的四元数无效 (例如，不是单位四元数)，跳过这一帧
            continue

        R_error = R_true.T @ R_est
        trace = np.trace(R_error)

        # 将 arccos 的参数裁剪到 [-1, 1] 范围内以避免数值错误
        arg = np.clip(0.5 * (trace - 1.0), -1.0, 1.0)
        angle_rad = np.arccos(arg)
        rotation_errors_deg.append(np.rad2deg(angle_rad))

    if not rotation_errors_deg:
        return np.nan

    return np.mean(rotation_errors_deg)


def main():
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="计算两条轨迹的ATE, ARE 误差 (无对齐版本)。\n"
                    "文件格式应为: timestamp tx ty tz qx qy qz qw (逗号或空格分隔)",
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
            raise ValueError(
                f"文件行数不匹配 (真值 {len(gt_poses)} 行, 估计 {len(est_poses)} 行)，请确保位姿一一对应。")

        print(f"\n成功读取文件:")
        print(f"  真值轨迹: '{args.groundtruth_file}' ({len(gt_poses)} 个位姿)")
        print(f"  估计轨迹: '{args.estimated_file}' ({len(est_poses)} 个位姿)")
        print("-" * 30)

        # --- 计算各项误差 (均无对齐) ---

        # 1. ATE (平移)，并获取有效位姿的掩码
        ate_mae, valid_mask = calculate_ate_translation_error(
            gt_poses, est_poses)
        valid_poses_count = np.sum(valid_mask)

        print("1. 平均绝对平移误差 (ATE - Unaligned)")
        print(f"   (基于平移误差阈值，使用了 {valid_poses_count}/{len(gt_poses)} 个有效位姿)")
        if not np.isnan(ate_mae):
            print(f"   结果: {ate_mae:.4f} 米\n")
        else:
            print("   结果: 无法计算\n")

        # 2. ARE (旋转)，传入掩码以使用相同的子集
        are_mae = calculate_are_rotation_error(gt_poses, est_poses, valid_mask)
        print("2. 平均绝对旋转误差 (ARE - Unaligned)")
        print(f"   (使用与平移误差相同的 {valid_poses_count} 个位姿进行计算)")
        if not np.isnan(are_mae):
            print(f"   结果: {are_mae:.4f} 度\n")
        else:
            print("   结果: 无法计算\n")

    except FileNotFoundError as e:
        print(f"\n[错误] 文件未找到: {e.filename}")
    except Exception as e:
        print(f"\n[错误] 发生异常: {e}")


if __name__ == "__main__":
    main()
