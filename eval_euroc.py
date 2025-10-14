#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse


def read_euroc_format(filepath):
    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.replace(',', ' ').split()
            data = [float(x) for x in parts]
            poses.append(np.array(data[1:]))
    return np.array(poses)


def calculate_ate_translation_error(gt_poses, est_poses, threshold=1000):
    est_trans = est_poses[:, :3]
    gt_trans = gt_poses[:, :3]
    # est_norms = np.linalg.norm(est_trans, axis=1)

    errors = np.linalg.norm(gt_trans - est_trans, axis=1)
    valid_mask = ((~np.isnan(errors)) & (
        errors < threshold))
    valid_errors = errors[valid_mask]

    if len(valid_errors) == 0:
        print("warning: no valid poses found within the threshold.")
        return np.nan, valid_mask

    max_error = np.max(valid_errors)
    max_index_in_valid_set = np.argmax(valid_errors)
    original_index = np.where(valid_mask)[0][max_index_in_valid_set]

    print(
        f"max translation error: {max_error:.4f} m (at original index {original_index})")

    return np.mean(valid_errors), valid_mask


def calculate_are_rotation_error(gt_poses, est_poses, valid_mask):
    gt_poses_filtered = gt_poses[valid_mask]
    est_poses_filtered = est_poses[valid_mask]

    if len(gt_poses_filtered) == 0:
        return np.nan

    rotation_errors_deg = []
    for i in range(len(gt_poses_filtered)):
        gt_quat = gt_poses_filtered[i, 3:]
        est_quat = est_poses_filtered[i, 3:]

        if np.linalg.norm(gt_quat) < 1e-6 or np.linalg.norm(est_quat) < 1e-6:
            continue

        try:
            R_true = R.from_quat(gt_quat).as_matrix()
            R_est = R.from_quat(est_quat).as_matrix()
        except ValueError:
            continue

        R_error = R_true.T @ R_est
        trace = np.trace(R_error)

        arg = np.clip(0.5 * (trace - 1.0), -1.0, 1.0)
        angle_rad = np.arccos(arg)
        rotation_errors_deg.append(np.rad2deg(angle_rad))

    if not rotation_errors_deg:
        return np.nan

    return np.mean(rotation_errors_deg)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('groundtruth_file', type=str)
    parser.add_argument('estimated_file', type=str)
    args = parser.parse_args()

    try:
        gt_poses = read_euroc_format(args.groundtruth_file)
        est_poses = read_euroc_format(args.estimated_file)

        if len(gt_poses) == 0 or len(est_poses) == 0:
            raise ValueError("file is empty or contains no valid poses.")
        if len(gt_poses) != len(est_poses):
            raise ValueError(
                "ground truth and estimated files must have the same number of poses.")

        ate_mae, valid_mask = calculate_ate_translation_error(
            gt_poses, est_poses)
        valid_poses_count = np.sum(valid_mask)

        print("1. absolute translation error (ATE - Unaligned)")
        if not np.isnan(ate_mae):
            print(f"result: {ate_mae:.4f} m")
        else:
            print("cannot calculate ATE, no valid poses found.")

        are_mae = calculate_are_rotation_error(gt_poses, est_poses, valid_mask)
        print("2. absolute rotation error (ARE - Unaligned)")
        print(f"valid poses count: {valid_poses_count}")
        if not np.isnan(are_mae):
            print(f"result: {are_mae:.4f} degrees")
        else:
            print("cannot calculate ARE, no valid poses found.")

    except FileNotFoundError as e:
        print(f"\nerror: file not found: {e.filename}")
    except Exception as e:
        print(f"\nerror: {str(e)}")


if __name__ == "__main__":
    main()
