function join_by { local IFS="$1"; shift; echo "$*"; }

cd ../smtlearn
declare -a options=(25 50 75 100 250 500 750 1000 2500 5000 7500 10000)

scp -r samuelk@himec04.cs.kuleuven.be:/home/samuelk/projects/smtlearn/output/synthetic/ss/* ../output/synthetic/ss/

python api.py combine ../output/synthetic/ss/summary $(join_by " " "${options[@]}") -p ../output/synthetic/ss/
python api.py migrate ratio ../output/synthetic/ss/summary/ -d ../synthetic/ss/10000 -s 1000 -f
python api.py migrate accuracy ../output/synthetic/ss/summary/ -d ../synthetic/ss/10000 -s 1000 -f
