#!/usr/bin/env bash


gpus=(2 3 4 5 6 7)
num_gpus=${#gpus[@]}
cnt=0

dataset=gisette
outdir=out/ablation-party-dropout/$dataset
mkdir -p $outdir
for seed in 0 1 2 3 4; do
  for noise in 0.05; do
    for pdo in 0 0.2 0.4 0.6 0.8 1.0; do
      gpu=${gpus[$cnt]}
      taskset -c 0-55 python src/script/train_fet.py -d $dataset -m acc -c 2 -p 50 --key-noise $noise -s $seed -g "$gpu" \
      -e 50 -v 6 --knn-k 100 -w 100 -nh 1 -ded 200 -ked 200 -nlb 1 -nab 1 --dropout 0.0 -paf 2 --party-dropout $pdo \
      > "$outdir"/"$dataset"_fet_p50_k100_noise${noise}_pdo${pdo}_s${seed}.log &
      cnt=$((cnt+1))
      if [ $cnt -eq "$num_gpus" ]; then
        wait
        cnt=0
      fi
    done
  done
done


dataset=mnist
outdir=out/ablation-party-dropout/$dataset
mkdir -p $outdir
for seed in 0 1 2 3 4; do
  for noise in 0.05; do
    for pdo in 0 0.2 0.4 0.6 0.8 1.0; do
      gpu=${gpus[$cnt]}
      taskset -c 0-55 python src/script/train_fet.py -d $dataset -m acc -c 10 -p 50 --key-noise $noise -s $seed -g "$gpu" \
      -e 30 -v 6 --knn-k 100 -w 100 -nh 1 -ded 200 -ked 200 -nlb 1 -nab 1 --dropout 0.0 -paf 2 --party-dropout $pdo \
      > "$outdir"/"$dataset"_fet_p50_k100_noise${noise}_pdo${pdo}_s${seed}.log &
      cnt=$((cnt+1))
      if [ $cnt -eq "$num_gpus" ]; then
        wait
        cnt=0
      fi
    done
  done
done


