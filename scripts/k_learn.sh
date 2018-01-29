#!/bin/bash

function join_by { local IFS="$1"; shift; echo "$*"; }

cd ../smtlearn

declare -a options=(1 2 3 4 5)

for i in "${options[@]}"; do
    python api.py generate ../synthetic/kk/1 -n 100 -b 6 -r 2 -k $i -l 3 --h 6
done
wait