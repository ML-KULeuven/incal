function join_by { local IFS="$1"; shift; echo "$*"; }

cd ../smtlearn
declare -a options=(1 2 3 4 5 6)

mkdir ../synthetic/kk/all
for i in "${options[@]}"; do
    cp ../synthetic/kk/$i/* ../synthetic/kk/all/
done


scp -r samuelk@himec04.cs.kuleuven.be:/home/samuelk/projects/smtlearn/output/synthetic/kk/* ../output/synthetic/kk/

python api.py combine ../output/synthetic/kk/summary $(join_by " " "${options[@]}") -p ../output/synthetic/kk/
python api.py migrate ratio ../output/synthetic/kk/summary/ -d ../synthetic/kk/all -s 1000 -f
python api.py migrate accuracy ../output/synthetic/kk/summary/ -d ../synthetic/kk/all -s 1000 -f
