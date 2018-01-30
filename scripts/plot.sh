cd ../smtlearn

# Plot samples
python api.py table id samples acc ../output/synthetic/ss/summary/ ../synthetic/ss/10000/ ../output/synthetic/ss/esummary_1000/ ../synthetic/ss/10000/ plot -a -o ../../ijcai18/figures/s_inc_acc.pdf
python api.py table id samples time ../output/synthetic/ss/summary/ ../synthetic/ss/10000/ ../output/synthetic/ss/esummary_1000/ ../synthetic/ss/10000/ plot -a -o ../../ijcai18/figures/s_inc_time.pdf

# Plot k
python api.py table id k acc ../output/synthetic/kk/summary/ ../synthetic/kk/all/ plot -a -o ../../ijcai18/figures/k_inc_acc.pdf
python api.py table id k time ../output/synthetic/kk/summary/ ../synthetic/kk/all/ plot -a -o ../../ijcai18/figures/k_inc_time.pdf

# Plot l
python api.py table id l acc ../output/synthetic/ll/summary/ ../synthetic/ll/all/ plot -a -o ../../ijcai18/figures/l_inc_acc.pdf
python api.py table id l time ../output/synthetic/ll/summary/ ../synthetic/ll/all/ plot -a -o ../../ijcai18/figures/l_inc_time.pdf

# Plot h
python api.py table id h acc ../output/synthetic/hh/summary/ ../synthetic/hh/all/ plot -a -o ../../ijcai18/figures/h_inc_acc.pdf
python api.py table id h time ../output/synthetic/hh/summary/ ../synthetic/hh/all/ plot -a -o ../../ijcai18/figures/h_inc_time.pdf