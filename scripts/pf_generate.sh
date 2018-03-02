#!/bin/bash

cd ../smtlearn

python api.py generate ../synthetic/pf -n 100 -b 6 -r 2 -k 3 -l 3 --half_spaces 6

scp -r ../synthetic/pf samuelk@himec04.cs.kuleuven.be:/home/samuelk/projects/smtlearn/synthetic/