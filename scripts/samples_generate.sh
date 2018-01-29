#!/bin/bash

cd ../smtlearn

python api.py generate ../synthetic/ss/$i -n 100 -b 6 -r 2 -k 2 -l 3 --h 6 -s 10000
scp -R ../synthetic/ss samuelk@himec04.cs.kuleuven.be:/home/samuelk/projects/smtlearn/output/synthetic/