cd ../smtlearn

# Plot samples
python api.py table id samples acc ../output/synthetic/ss/summary/ ../synthetic/ss/10000/ INCAL ../output/synthetic/ss/esummary_1000/ ../synthetic/ss/10000/ "non-incremental INCAL" plot -a -o ../../ijcai18/figures/s_inc_acc.pdf
python api.py table id samples time ../output/synthetic/ss/summary/ ../synthetic/ss/10000/ INCAL ../output/synthetic/ss/esummary_1000/ ../synthetic/ss/10000/ "non-incremental INCAL" plot -a -o ../../ijcai18/figures/s_inc_time.pdf
python api.py table id samples active ../output/synthetic/ss/summary/ ../synthetic/ss/all/ INCAL plot -a -o ../../ijcai18/figures/s_inc_active.pdf
python api.py table id samples active_ratio ../output/synthetic/ss/summary/ ../synthetic/ss/all/ INCAL plot -a -o ../../ijcai18/figures/s_inc_active_ratio.pdf

python api.py table id samples time ../output/synthetic/ss/summary ../synthetic/ss/all INCAL print -a
python api.py table id samples time ../output/synthetic/ss/esummary ../synthetic/ss/all INCAL print -a

# Plot k
python api.py table id k acc ../output/synthetic/kk/summary/ ../synthetic/kk/all/ INCAL plot -a -o ../../ijcai18/figures/k_inc_acc.pdf
python api.py table id k time ../output/synthetic/kk/summary/ ../synthetic/kk/all/ INCAL plot -a -o ../../ijcai18/figures/k_inc_time.pdf

# Plot l
python api.py table id l acc ../output/synthetic/ll/summary/ ../synthetic/ll/all/ INCAL plot -a -o ../../ijcai18/figures/l_inc_acc.pdf
python api.py table id l time ../output/synthetic/ll/summary/ ../synthetic/ll/all/ INCAL plot -a -o ../../ijcai18/figures/l_inc_time.pdf

# Plot h
python api.py table id h acc ../output/synthetic/hh/summary/ ../synthetic/hh/all/ INCAL plot -a -o ../../ijcai18/figures/h_inc_acc.pdf
python api.py table id h time ../output/synthetic/hh/summary/ ../synthetic/hh/all/ INCAL plot -a -o ../../ijcai18/figures/h_inc_time.pdf

# Print parameter-free ratio
python api.py table id samples time_ratio ../output/synthetic/pf/ ../synthetic/pf/ INCAL print -a

# Print benchmark
python api.py table id constant full_time ../output/benchmark/pf1000/ ../demo/cache/ INCAL print -a
python api.py table id constant time ../output/benchmark/pf1000/ ../demo/cache/ INCAL print -a
python api.py table id constant time_ratio ../output/benchmark/pf1000/ ../demo/cache/ INCAL print -a
python api.py table id constant acc ../output/benchmark/pf1000/ ../demo/cache/ INCAL print -a
