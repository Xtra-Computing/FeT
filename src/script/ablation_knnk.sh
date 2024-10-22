#!/usr/bin/env bash


gpus=(0 1 2 3)
num_gpus=${#gpus[@]}
cnt=0
party=20

dataset=gisette
outdir=out/ablation-knnk/$dataset
mkdir -p $outdir
for seed in 0 1 2 3 4; do
  for noise in 0.05; do
    for knnk in 1 5 10 20 40 60 80 100; do
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
outdir=out/ablation-knnk/$dataset
mkdir -p $outdir
for seed in 0 1 2 3 4; do
  for noise in 0.05; do
    for knnk in 1 5 10 20 40 60 80 100; do
      gpu=${gpus[$cnt]}
      python src/script/train_fet.py -d $dataset -m acc -c 10 -p $party --key-noise $noise -s $seed -g "$gpu" \
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


