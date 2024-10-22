#!/usr/bin/env bash

outdir=out/real/
mkdir -p $outdir

for seed in 0 1 2 3 4; do
  python src/script/train_fet.py -d house -m rmse -c 1 -p 2 -s $seed --knn-k 100 -nh 4 -ded 100 -ked 100 -nlb 3 -nab 3 -paf 1 --dropout 0.3 -g 0 > \
    ${outdir}/house_seed${seed}.log
  python src/script/train_fet.py -d taxi -m rmse -c 1 -p 2 -s $seed -e 50 -lr 3e-4 --knn-k 100 -nh 4 -ded 100 -ked 100 -nlb 3 -nab 3 -paf 1 --dropout 0.3 -g 0 > \
    ${outdir}/taxi_seed${seed}.log
  python src/script/train_fet.py -d hdb -m rmse -c 1 -p 2 -s $seed --knn-k 100 -nh 4 -ded 100 -ked 100 -nlb 1 -nab 2 -paf 1 --dropout 0.3 -g 0 > \
    ${outdir}/hdbs_seed${seed}.log
  wait
done
