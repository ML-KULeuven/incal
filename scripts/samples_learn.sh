#!/bin/bash

cd ../smtlearn

declare -a options=(25 50 75 100 250 500 750 1000 2500 5000 7500 10000)
declare -a options_e=(25 50 75 100 250 500 750 1000 2500)

for i in "${options[@]}"; do
    python experiments.py ../synthetic/ss/10000/ "" ../output/synthetic/ss/$i cnf -s $i -t 200 &
done

for i in "${options_e[@]}"; do
    python experiments.py ../synthetic/ss/10000/ "" ../output/synthetic/ss/e$i cnf -s $i -t 200 -a &
done
wait
