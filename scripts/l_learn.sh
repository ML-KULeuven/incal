#!/bin/bash

cd ../smtlearn

declare -a options=(1 2 3 4 5)

for i in "${options[@]}"; do
    python experiments.py ../synthetic/ll/$i "" ../output/synthetic/ll/$i cnf -t 200 &
done
wait