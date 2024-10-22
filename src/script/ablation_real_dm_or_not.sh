#!/usr/bin/env bash

outdir=out/real/
mkdir -p $outdir

for seed in 0 1 2 3 4; do
  python src/script/train_fet.py -d house -m rmse -c 1 -p 2 -s $seed -v 6 --knn-k 100 -nh 4 -ded 100 -ked 100 -nlb 3 -nab 3 -paf 1 --dropout 0.3 -g 4 --disable-dm > \
    ${outdir}/house_seed${seed}_nodm.log &
  python src/script/train_fet.py -d taxi -m rmse -c 1 -p 2 -s $seed -v 6 -e 50 -lr 3e-4 --knn-k 100 -nh 4 -ded 100 -ked 100 -nlb 3 -nab 3 -paf 1 --dropout 0.3 -g 7 --disable-dm > \
    ${outdir}/taxi_seed${seed}_nodm.log &
  python src/script/train_fet.py -d hdb -m rmse -c 1 -p 2 -s $seed -v 6 --knn-k 100 -nh 4 -ded 100 -ked 100 -nlb 1 -nab 2 -paf 1 --dropout 0.3 -g 5 --disable-dm > \
    ${outdir}/hdbs_seed${seed}_nodm.log &
  wait
done

