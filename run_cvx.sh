#!/bin/bash

python3 script/run_cvxpnpl.py /home/ljj/dataset/kitti-360/2013_05_28_drive_0000_sync/ 11396
# python3 script/run_cvxpnpl.py /home/ljj/dataset/kitti-360/2013_05_28_drive_0002_sync/ 19138
python3 script/run_cvxpnpl.py /home/ljj/dataset/kitti-360/2013_05_28_drive_0003_sync/ 938
python3 script/run_cvxpnpl.py /home/ljj/dataset/kitti-360/2013_05_28_drive_0004_sync/ 11270
# python3 script/run_cvxpnpl.py /home/ljj/dataset/kitti-360/2013_05_28_drive_0005_sync/ 6627
# python3 script/run_cvxpnpl.py /home/ljj/dataset/kitti-360/2013_05_28_drive_0006_sync/ 9606
# python3 script/run_cvxpnpl.py /home/ljj/dataset/kitti-360/2013_05_28_drive_0007_sync/ 3060
# python3 script/run_cvxpnpl.py /home/ljj/dataset/kitti-360/2013_05_28_drive_0009_sync/ 13852
# python3 script/run_cvxpnpl.py /home/ljj/dataset/kitti-360/2013_05_28_drive_0010_sync/ 3593
#
# python3 script/run_cvxpnpl.py /home/ljj/dataset/euroc/MH_01_easy/mav0/ 3569
# python3 script/run_cvxpnpl.py /home/ljj/dataset/euroc/MH_03_medium/mav0/ 2587
# python3 script/run_cvxpnpl.py /home/ljj/dataset/euroc/MH_04_difficult/mav0/ 1919
# python3 script/run_cvxpnpl.py /home/ljj/dataset/euroc/V1_01_easy/mav0/ 2801
# python3 script/run_cvxpnpl.py /home/ljj/dataset/euroc/V1_02_medium/mav0/ 1600
# python3 script/run_cvxpnpl.py /home/ljj/dataset/euroc/V1_03_difficult/mav0/ 2039
# python3 script/run_cvxpnpl.py /home/ljj/dataset/euroc/V2_02_medium/mav0/ 2243

# for ((i = 35; i <= 50; i += 5)) do
#     j=$((2 * i))
#     python3 script/run_cvxpnpl.py /home/ljj/source_code/PL-MCVO/third-party/upnpl/simulated/"$i"_"$j"_2/ 10000
# done

# for i in {9..10} 
# do
#     python3 script/run_cvxpnpl.py /home/ljj/source_code/PL-MCVO/third-party/upnpl/simulated/50_100_"$i"/ 10000
# done

