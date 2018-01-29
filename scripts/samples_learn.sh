#!/bin/bash

function join_by { local IFS="$1"; shift; echo "$*"; }

cd ../smtlearn

declare -a options=(25 50 75 100 250 500 750 1000 2500 5000 7500 10000)

for i in "${options[@]}"; do
    python experiments.py ../synthetic/ss/10000/ "" ../output/synthetic/ss/$i cnf -s $i -t 200 &
done
wait
