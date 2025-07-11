#!/bin/bash

if [ "$1" = "1" ]; then

    for ((i = 10; i <= 100; i += 10)) do
        python3 eval_euroc.py ./simulated/"$i"_0_2/gt_simulated_trajectory.txt ./simulated/"$i"_0_2/"$2".txt
    done

    for i in {1..10}
        do
            python3 eval_euroc.py ./simulated/100_0_"$i"/gt_simulated_trajectory.txt ./simulated/100_0_"$i"/"$2".txt
        done

elif [ "$1" = "2" ]; then

    for ((i = 5; i <= 50; i += 5)) do
        j=$((2 * i))
        python3 eval_euroc.py ./simulated/"$i"_"$j"_2/gt_simulated_trajectory.txt ./simulated/"$i"_"$j"_2/"$2".txt
    done

    for i in {1..10}
    do
        python3 eval_euroc.py ./simulated/50_100_"$i"/gt_simulated_trajectory.txt ./simulated/50_100_"$i"/"$2".txt
    done

elif [ "$1" = "3" ]; then

    python3 eval_euroc.py /home/ljj/dataset/euroc/MH_01_easy/mav0/gt.csv /home/ljj/dataset/euroc/MH_01_easy/mav0/"$2"

    python3 eval_euroc.py /home/ljj/dataset/euroc/MH_03_medium/mav0/gt.csv /home/ljj/dataset/euroc/MH_03_medium/mav0/"$2"

    python3 eval_euroc.py /home/ljj/dataset/euroc/V1_01_easy/mav0/gt.csv /home/ljj/dataset/euroc/V1_01_easy/mav0/"$2"

    python3 eval_euroc.py /home/ljj/dataset/euroc/V1_02_medium/mav0/gt.csv /home/ljj/dataset/euroc/V1_02_medium/mav0/"$2"

else
    python3 eval_euroc.py /home/ljj/dataset/kitti-360/2013_05_28_drive_0000_sync/gt.txt /home/ljj/dataset/kitti-360/2013_05_28_drive_0000_sync/"$2".txt

    python3 eval_euroc.py /home/ljj/dataset/kitti-360/2013_05_28_drive_0003_sync/gt.txt /home/ljj/dataset/kitti-360/2013_05_28_drive_0003_sync/"$2".txt

    python3 eval_euroc.py /home/ljj/dataset/kitti-360/2013_05_28_drive_0004_sync/gt.txt /home/ljj/dataset/kitti-360/2013_05_28_drive_0004_sync/"$2".txt

fi
