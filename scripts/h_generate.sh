#!/bin/bash

cd ../smtlearn

declare -a options=(3 4 5 6 7 8 9 10)

for i in "${options[@]}"; do
    python api.py generate ../synthetic/hh/$i -n 100 -b 0 -r 2 -k 2 -l 3 --half_spaces $i
done
wait

scp -r ../synthetic/hh samuelk@himec04.cs.kuleuven.be:/home/samuelk/projects/smtlearn/synthetic/