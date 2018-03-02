#!/bin/bash

cd ../smtlearn

declare -a options=(1 2 3 4 5 6)

for i in "${options[@]}"; do
    python api.py generate ../synthetic/kk/$i -n 100 -b 6 -r 2 -k $i -l 3 --half_spaces 6
done
wait

scp -r ../synthetic/kk samuelk@himec04.cs.kuleuven.be:/home/samuelk/projects/smtlearn/synthetic/