function join_by { local IFS="$1"; shift; echo "$*"; }

cd ../smtlearn
declare -a options=(1 2 3 4 5)

mkdir ../synthetic/ll/all
for i in "${options[@]}"; do
    cp ../synthetic/ll/$i/* ../synthetic/ll/all/
done

scp -r samuelk@himec04.cs.kuleuven.be:/home/samuelk/projects/smtlearn/output/synthetic/ll ../output/synthetic/ll

python api.py combine ../output/synthetic/ll/summary $(join_by " " "${options[@]}") -p ../output/synthetic/ll/
python api.py migrate ratio ../output/synthetic/ll/summary/ -d ../synthetic/ll/all -s 1000 -f
python api.py migrate accuracy ../output/synthetic/ll/summary/ -d ../synthetic/ll/all -s 1000 -f
