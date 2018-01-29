function join_by { local IFS="$1"; shift; echo "$*"; }

cd ../smtlearn
declare -a options=(1 2 3 4 5 6)

scp -r samuelk@himec04.cs.kuleuven.be:/home/samuelk/projects/smtlearn/output/synthetic/kk/* ../output/synthetic/kk/

python api.py combine ../output/synthetic/ss/summary $(join_by " " "${options[@]}") -p ../output/synthetic/ss/
python api.py migrate ratio ../output/synthetic/ss/summary/ -d ../synthetic/ss/10000 -s 1000 -f
python api.py migrate accuracy ../output/synthetic/ss/summary/ -d ../synthetic/ss/10000 -s 1000 -f
