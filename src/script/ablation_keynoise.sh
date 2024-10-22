#!/usr/bin/env bash


gpus=(1 2 3 4 5 6)
num_gpus=${#gpus[@]}
cnt=0
party=50

dataset=gisette
outdir=out/ablation-keynoise/$dataset
mkdir -p $outdir
for seed in 0 1 2 3 4; do
  for noise in 0.02 0.05 0.06 0.07 0.1; do
    for knnk in 100; do
      gpu=${gpus[$cnt]}
      python src/script/train_fet.py -d $dataset -m acc -c 2 -p $party --key-noise $noise -s $seed -g "$gpu" \
      -e 50 -v 6 --knn-k ${knnk} -w 100 -nh 1 -ded 200 -ked 200 -nlb 1 -nab 1 --dropout 0.0 -paf 1 --party-dropout 0.6 \
      > "$outdir"/"$dataset"_fet_p${party}_k${knnk}_noise${noise}_pdo0_s${seed}.log &
      cnt=$((cnt+1))
      if [ $cnt -eq "$num_gpus" ]; then
        wait
        cnt=0
      fi
    done
  done
done


dataset=mnist
outdir=out/ablation-keynoise/$dataset
mkdir -p $outdir
for seed in 0 1 2 3 4; do
  for noise in 0.02 0.05 0.06 0.07 0.1; do
    for knnk in 100; do
      gpu=${gpus[$cnt]}
      python src/script/train_fet.py -d $dataset -m acc -c 10 -p $party --key-noise $noise -s $seed -g "$gpu" \
      -e 30 -v 6 --knn-k ${knnk} -w 100 -nh 4 -ded 200 -ked 200 -nlb 6 -nab 6 --dropout 0.0 -paf 1 --party-dropout 0.6 \
      > "$outdir"/"$dataset"_fet_p${party}_k${knnk}_noise${noise}_pdo0_s${seed}.log &
      cnt=$((cnt+1))
      if [ $cnt -eq "$num_gpus" ]; then
        wait
        cnt=0
      fi
    done
  done
done


