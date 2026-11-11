#!/usr/bin/env python3
"""Compare unchanged HEAD and current solvers on identical synthetic inputs.

Build with tests/CMakeLists.txt first; outputs go to a new directory only.
No repository datasets or existing result files are read or overwritten.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from collections import defaultdict


KEY = ("mode", "case", "seed", "repeat", "sigma", "outliers")


def run(executable, mode, trials, repeats, destination):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    start = time.monotonic()
    with destination.open("x") as stdout, destination.with_suffix(".stderr").open("x") as stderr:
        subprocess.run([str(executable), f"--{mode}", str(trials), str(repeats)],
                       stdout=stdout, stderr=stderr, env=env, check=True)
    with destination.open() as source:
        rows = list(csv.DictReader(source))
    if not rows or any(not row.get("input_hash") for row in rows):
        raise RuntimeError(f"Malformed/empty simulation output: {destination}")
    print(f"{destination.name}: {len(rows)} poses, {time.monotonic() - start:.2f}s", flush=True)
    return rows


def quantile(values, probability):
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(len(values) * probability) - 1)]


def summarize(rows, variant):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ("mode", "case", "sigma", "outliers"))].append(row)
    for key, group in sorted(groups.items()):
        result = dict(zip(("mode", "case", "sigma", "outliers"), key))
        result.update(variant=variant, count=len(group),
                      nonfinite=sum(r["finite"] != "1" for r in group),
                      failures=sum(r["success"] != "1" for r in group))
        for column in ("rotation_deg", "translation_relative", "point_residual"):
            values = [float(r[column]) for r in group]
            values = [v if math.isfinite(v) else math.inf for v in values]
            result[column + "_median"] = statistics.median(values)
            result[column + "_p95"] = quantile(values, 0.95)
            result[column + "_max"] = max(values)
        yield result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--sample-repeats", type=int, default=5)
    parser.add_argument("--ransac-trials", type=int, default=5)
    parser.add_argument("--ransac-repeats", type=int, default=5)
    parser.add_argument("--skip-ransac", action="store_true")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Recompute summaries/checks from existing CSVs without rerunning solvers")
    args = parser.parse_args()
    build = args.build.resolve()
    if not args.analyze_only:
        args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    metadata = {"platform": platform.platform(), "trials": args.trials,
                "sample_repeats": args.sample_repeats,
                "ransac_trials": args.ransac_trials, "ransac_repeats": args.ransac_repeats,
                "baseline_commit": "861db5d68637d105f90aed6a8154188bf880b80f",
                "seed_start": 20261111,
                "randomness": "Fixed inputs and srand; original random_device/shuffle unchanged. "
                              "<=20 observations score all observations in shuffled order; "
                              ">20 and RANSAC use independent repeated draws for each variant.",
                "success": "Finite proper rotation; rotation error <1 degree; "
                           "translation error / scene coordinate_scale <0.1.",
                "source_sha256": hashlib.sha256((root / "src/UPnPL.cc").read_bytes()).hexdigest()}
    if not args.analyze_only:
        baseline_source = next(line.split("=", 1)[1] for line in
                               (build / "CMakeCache.txt").read_text().splitlines()
                               if line.startswith("UPNPL_BASELINE_SOURCE:FILEPATH="))
        exported = Path(baseline_source).read_bytes()
        original = subprocess.check_output(["git", "show", metadata["baseline_commit"] + ":src/UPnPL.cc"], cwd=root)
        if exported != original:
            raise AssertionError("Baseline source is not the unchanged starting HEAD")
        metadata["baseline_source_sha256"] = hashlib.sha256(exported).hexdigest()
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    suites = [("full", args.trials, 1), ("sampling", args.trials, args.sample_repeats)]
    if not args.skip_ransac:
        suites.append(("ransac", args.ransac_trials, args.ransac_repeats))
    summaries = []
    results = {}
    for mode, trials, repeats in suites:
        for variant in ("baseline", "current"):
            executable = (f"pose_ransac_{variant}" if mode == "ransac" else
                          "pose_simulation" + ("_baseline" if variant == "baseline" else ""))
            destination = args.output / f"{mode}_{variant}.csv"
            if args.analyze_only:
                with destination.open() as source:
                    rows = list(csv.DictReader(source))
            else:
                rows = run(build / executable, mode, trials, repeats, destination)
            results[mode, variant] = {tuple(r[k] for k in KEY): r for r in rows}
            summaries.extend(summarize(rows, variant))
        old, new = results[mode, "baseline"], results[mode, "current"]
        if old.keys() != new.keys() or any(old[k]["input_hash"] != new[k]["input_hash"] for k in old):
            raise AssertionError(f"Inputs differ between variants for {mode}")
    with (args.output / "summary.csv").open("w" if args.analyze_only else "x") as target:
        writer = csv.DictWriter(target, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    # Regression evidence for unchanged line-only behavior and bearing scale
    # invariance (no candidate subsets in the full suite).
    pose_columns = [f"r{i}{j}" for i in range(3) for j in range(3)] + ["tx", "ty", "tz"]
    old, new = results["full", "baseline"], results["full", "current"]
    line_delta, bearing_delta, clean_bearing_delta = 0.0, 0.0, 0.0
    for key, row in new.items():
        if row["case"] == "lines_only":
            line_delta = max(line_delta, *(abs(float(row[c]) - float(old[key][c])) for c in pose_columns))
        if row["case"] == "mixed_bearing_scale_7":
            normal_key = list(key)
            normal_key[1] = "mixed_4cam"
            reference = new[tuple(normal_key)]
            delta = max(abs(float(row[c]) - float(reference[c])) for c in pose_columns)
            bearing_delta = max(bearing_delta, delta)
            if row["outliers"] == "0":
                clean_bearing_delta = max(clean_bearing_delta, delta)
    checks = {"identical_inputs": True, "line_only_max_pose_component_delta": line_delta,
              "current_bearing_scale_7_max_pose_component_delta": bearing_delta,
              "current_bearing_scale_7_no_outlier_max_pose_component_delta": clean_bearing_delta}
    (args.output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    # Check end-to-end invariance on clean/noisy inliers. Contaminated fits may
    # amplify roundoff through candidate selection or nonlinear termination;
    # report their delta separately. Projector invariance is tested directly.
    if line_delta > 1e-8 or clean_bearing_delta > 1e-8:
        raise AssertionError("Unexpected line-only or homogeneous-bearing regression; inspect CSV files")


if __name__ == "__main__":
    main()
