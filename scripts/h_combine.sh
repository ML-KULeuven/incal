function join_by { local IFS="$1"; shift; echo "$*"; }

cd ../smtlearn
declare -a options=(3 4 5 6 7 8 9 10)

mkdir ../synthetic/hh/all
for i in "${options[@]}"; do
    cp ../synthetic/hh/$i/* ../synthetic/hh/all/
done

scp -r samuelk@himec04.cs.kuleuven.be:/home/samuelk/projects/smtlearn/output/synthetic/hh ../output/synthetic/hh

python api.py combine ../output/synthetic/hh/summary $(join_by " " "${options[@]}") -p ../output/synthetic/hh/
python api.py migrate ratio ../output/synthetic/hh/summary/ -d ../synthetic/hh/all -s 1000 -f
python api.py migrate accuracy ../output/synthetic/hh/summary/ -d ../synthetic/hh/all -s 1000 -f
