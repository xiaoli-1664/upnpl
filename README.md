# Unified Pose Estimation for Multi-Camera Systems Using Point and Line Correspondences

This repository contains the official implementation of the method proposed in our paper:

> **Unified Pose Estimation for Multi-Camera Systems Using Point and Line Correspondences**  

## 🌐 Overview

We present a unified and efficient approach to estimate the full 6-DoF pose of a multi-camera system from 2D–3D point and line correspondences. Our method supports arbitrary camera configurations and localization using points, lines, or a combination of both.

## Build Instructions

### Prerequisites

- C++ 17
- CMake 3.20 or higher
- Eigen 3.3.9
- OpenCV 4
- Ceres 2.2.0
- yamlcpp
- OpenGV (required by UPnP and GPnP)

### Build

```
mkdir build && cd build
cmake ..
make -j
```

## Test

### Synthetic Experiments

```
./build/sim_test point_num line_num noise_sigma count
```

### Real-world Experiments

- EuRoC

```
./build/euroc_test MH_01_easy
```

- KITTI-360

```
./build/kitti_test 0000
```
