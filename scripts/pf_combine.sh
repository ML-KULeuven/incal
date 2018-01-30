function join_by { local IFS="$1"; shift; echo "$*"; }

cd ../smtlearn
declare -a options=(1 2 3 4 5)

scp -r samuelk@himec04.cs.kuleuven.be:/home/samuelk/projects/smtlearn/output/synthetic/pf ../output/synthetic/pf

python api.py migrate ratio ../output/synthetic/pf -d ../synthetic/pf -s 1000 -f
python api.py migrate accuracy ../output/synthetic/pf -d ../synthetic/pf -s 1000 -f
