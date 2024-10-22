#!/usr/bin/env bash

gpus=(2 3 4 5 6 7)
num_gpus=${#gpus[@]}
cnt=0
k=100

dataset=gisette
folder=out/ablation-pe-avg-freq/$dataset
mkdir -p $folder
for noise in 0.05; do
  for seed in 0 1 2 3 4; do
    for paf in 0 1 2 3 5 10; do
      for np in 2 5 20 50; do
        gpu=${gpus[$cnt]}
        taskset -c 0-55 python src/script/train_fet.py -d $dataset -m acc -c 2 -p $np --key-noise $noise -s $seed -v 6 --knn-k $k \
        -w 100 -nh 1 -ded 200 -ked 200 -nlb 1 -nab 1 -e 50 --dropout 0.0 -paf $paf -g "$gpu" \
        > $folder/${dataset}_fet_p${np}_k${k}_noise${noise}_paf${paf}_s${seed}.log &
        cnt=$((cnt+1))
        if [ $cnt -eq "$num_gpus" ]; then
          wait
          cnt=0
        fi
      done
    done
  done
done


dataset=mnist
folder=out/ablation-pe-avg-freq/$dataset
mkdir -p $folder
for np in 2 5 20 50; do
  for noise in 0.05; do
    for seed in 0 1 2 3 4; do
      for paf in 0 1 2 3 5 10; do
        gpu=${gpus[$cnt]}
        taskset -c 0-55 python src/script/train_fet.py -d $dataset -m acc -c 10 -p $np --key-noise $noise -s $seed -v 6 --knn-k $k \
        -w 100 -nh 1 -ded 200 -ked 200 -nlb 1 -nab 1 -e 30 --dropout 0.0 -paf $paf -g "$gpu" \
        > ${folder}/${dataset}_fet_p${np}_k${k}_noise${noise}_paf${paf}_s${seed}.log &
        cnt=$((cnt+1))
        if [ $cnt -eq "$num_gpus" ]; then
          wait
          cnt=0
        fi
      done
    done
  done
done
