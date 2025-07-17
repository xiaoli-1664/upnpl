#!/bin/bash

# ./build/euroc_test MH_01_easy
# ./build/euroc_test MH_03_medium
# ./build/euroc_test MH_04_difficult
# ./build/euroc_test V1_01_easy
# ./build/euroc_test V1_02_medium
# ./build/euroc_test V1_03_difficult
# ./build/euroc_test V2_02_medium

# for ((i = 5; i <= 50; i += 5)) do
#     j=$((2 * i))
#     echo "Running ./build/sim_test $j $i 2"
#     ./build/sim_test "$i" "$j" 2 10000
# done

# for i in {1..10}
# do
#     echo "Running ./build/sim_test 100 0 $i"
#     ./build/sim_test 100 0 "$i" 10000
# done
#
for ((i = 10; i <= 90; i += 10)) do
    echo "Running ./build/sim_test $i 0 2"
    ./build/sim_test "$i" 0 2 10000
done

# for i in {1..10}
# do
#     echo "Running ./build/sim_test 50 50 $i"
#     ./build/sim_test 50 100 "$i" 10000
# done

