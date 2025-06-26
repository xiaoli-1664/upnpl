#!/bin/bash

for ((i = 10; i <= 50; i += 5)) do
    j=$((2 * i))
    echo "Running ./build/sim_test $j $i 2"
    ./build/sim_test "$j" "$i" 2
done

for i in {1..10}
do
    echo "Running ./build/sim_test 100 0 $i"
    ./build/sim_test 100 0 "$i"
done

for ((i = 10; i <= 90; i += 10)) do
    echo "Running ./build/sim_test $i 0 2"
    ./build/sim_test "$i" 0 2
done

for i in {1..10}
do
    echo "Running ./build/sim_test 50 50 $i"
    ./build/sim_test 50 50 "$i"
done

