#!/bin/bash

cd ../smtlearn

declare -a options=(3 4 5 6 7 8 9 10)

for i in "${options[@]}"; do
    python experiments.py ../synthetic/hh/$i "" ../output/synthetic/hh/$i cnf -t 200 &
done
wait