# 点约束正交补验证

验证入口为 `solveMain`，基线是开始开发时的提交
`861db5d68637d105f90aed6a8154188bf880b80f`。测试只生成内存中的仿真输入，
输出写入指定的新目录，不使用或覆盖仓库已有实验数据、轨迹和结果。

## 构建和复现

在新 worktree 根目录运行：

```bash
mkdir -p build
git show 861db5d68637d105f90aed6a8154188bf880b80f:src/UPnPL.cc > build/baseline_UPnPL.cc
cmake -S tests -B build/point-constraints \
  -DUPNPL_BASELINE_SOURCE="$PWD/build/baseline_UPnPL.cc" \
  -DUPNPL_TEST_RANSAC=ON \
  -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
cmake --build build/point-constraints -j2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ctest --test-dir build/point-constraints --output-on-failure
python3 tests/compare_simulation.py \
  --build build/point-constraints --output build/validation-new \
  --trials 30 --sample-repeats 10 --ransac-trials 20 --ransac-repeats 10
```

`--output` 在采集时必须不存在，防止覆盖结果。脚本检查导出的基线源码与上述
提交完全一致，并逐条核对两版输入指纹。已有 CSV 可使用 `--analyze-only` 重新
生成汇总和检查文件；此模式不重跑求解器，不修改原始 CSV。

独立代数测试和直接位姿仿真仅需要 Eigen、Ceres。若不需要外层 RANSAC，可省略
`UPNPL_TEST_RANSAC` 和 OpenCV 参数，并在比较脚本中使用 `--skip-ransac`。
RANSAC 测试直接调用现有 `utils::myUPnPL_RANSAC`，需要 OpenCV 和 OpenGV；
`OPENGV_PATH` 默认与项目一致，可用 CMake 参数调整。

本机默认 `/usr/local` OpenCV 4.9 配置需要指定
`-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8`，且缺少 `line_descriptor`、`ccalib`
链接目标。全项目构建可使用已经安装的系统 OpenCV 4.5.4：

```bash
cmake -S . -B build/system-opencv \
  -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
cmake --build build/system-opencv -j2
```

没有为此修改项目原有 CMake 或依赖。数值对比使用 GCC 11.4、Eigen 3.3.9、
Ceres 2.2.0 和原项目的 Release 编译选项；两版使用相同工具链。

## 测试设计

- `point_constraints_test.cc`：10,423 个方向，包含随机单位向量、正负坐标轴、
  分量相等的方向、`h=0`、正负 `1e-300` 至 `1e-3`。验证秩、奇异值、
  `QᵀQ=I`、`Qᵀf=0`、`DᵀD=I−ffᵀ`、
  `||Dv||²=||v×f||²`、非零相机中心的零残差，以及齐次缩放后的投影矩阵不变性。
- `simulation_data.h`：固定 `mt19937` 种子，从已知世界到机体的位姿反向生成
  点和线。四相机中心不共面；普通场景含非单位相机旋转。边界场景使用单位相机
  旋转，确保求解器实际接收的机体系第三分量在无噪时恰为零或指定近零值。
- 普通场景：在相机归一化像平面坐标上加入独立高斯噪声，σ 为 0 或 0.002。
  边界点：在单位视线切平面加入同标准差的噪声再归一化；带噪后的 `h` 不再严格
  等于零。线端点始终使用像平面噪声。没有把两种噪声模型混称为像素噪声。
- 外点：确定性错配观测，点像平面偏移 `(0.7,−0.5)`，线两端偏移 `(0.6,−0.4)`；
  点线混合时交替污染点、线。测试覆盖的是这一外点模型。
- 位姿记录包括完整 `R_bw,t_bw`、旋转误差（度）、平移误差除以场景尺度、
  所有点的单位视线叉积残差、有限性、旋转矩阵合法性和成功标记。
  残差包含外点，不能把带外点总残差直接当成内点拟合质量。
- 成功阈值为旋转误差 `<1°` 且归一化平移误差 `<0.1`，同时要求有限、合法的旋转。
  CTest 的无噪无外点位姿检查更严格：旋转误差 `≤1e-5°`、平移误差 `≤1e-6`。

三组对比：

| 分组 | 输入 | 外点 | 重复方式 |
| --- | --- | --- | --- |
| full | 12 种场景；16 点、12 点+8 线或16 线 | 0、1 | 30 个固定场景种子，每场景/噪声/外点条件1次 |
| sampling | 48 点或32 点+16 线，四相机 | 0、1 | 30 个固定场景种子，每条件10次 |
| ransac | 同上，调用原外层 RANSAC | 0、10（20.83%） | 20 个固定场景种子，每条件10次 |

full 的 12 种场景还覆盖单相机、多相机、归一化开关、世界长度尺度 ×100、
点视线齐次尺度 ×7、`h=0/1e-12/1e-6`、边界点线混合和纯线。
归一化开启且含线时，实际执行 `solveMain` 的两遍求解和线端点校正。

full 的观测数 `n+m≤20`，原程序虽然打乱顺序，但使用全部观测，没有随机选取
评分子集；其辅助多项式中的 `Eigen::setRandom` 通过每次调用前的 `srand` 固定。
sampling 的 `n+m=48`，保留原程序随机取 20 个观测评分的行为。两组都是全部
输入构造线性系统，内部抽样不负责剔除外点。

外层 RANSAC 保持原实现，纯点每次抽8点，混合抽4点+4线，60次迭代，阈值0.015
（归一化像平面距离）。外层和内部 `random_device` 未改动，所以原/新版共享
固定输入但不共享抽样序列；报告重复运行的分布，不把差异全部归因于点约束。
RANSAC 组限于前向相机观测；边界视线由直接 `solveMain` 组验证。

## 文件

- `summary.csv`：各条件误差中位数、P95（nearest rank）、最大值、失败数、非有限数。
- `{full,sampling,ransac}_{baseline,current}.csv`：逐次输入指纹、误差和完整位姿。
- 对应 `.stderr`：求解器诊断输出。
- `checks.json`：输入一致性、纯线不变性、视线尺度不变性的端到端检查。
- `metadata.json`：输入/辅助随机种子约定、基线提交、源码校验和、阈值和实验规模。

具体测得的数据及限制见 [validation_report.md](validation_report.md)。
