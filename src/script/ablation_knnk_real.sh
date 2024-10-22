#!/usr/bin/env bash


gpus=(0 1 2 3 4 5 6 7)
num_gpus=${#gpus[@]}
cnt=0
party=2

dataset=house
outdir=out/ablation-knnk/$dataset
mkdir -p $outdir
for seed in 0 1 2 3 4; do
 for knnk in 1 5 10 20 40 60 80 100; do
   gpu=${gpus[$cnt]}
   python src/script/train_fet.py -d $dataset -m rmse -c 1 -p $party -s $seed -g "$gpu" \
   -e 100 -v 6 --knn-k ${knnk} -nh 1 -ded 100 -ked 100 -nlb 3 -nab 3 --dropout 0.3 -paf 1 \
   > "$outdir"/"$dataset"_fet_p${party}_k${knnk}_s${seed}.log &
   cnt=$((cnt+1))
   if [ $cnt -eq "$num_gpus" ]; then
     wait
     cnt=0
   fi
 done
done


dataset=taxi
outdir=out/ablation-knnk/$dataset
mkdir -p $outdir
for seed in 0 1 2 3 4; do
 for knnk in 1 5 10 20 40 60 80 100; do
   gpu=${gpus[$cnt]}
   python src/script/train_fet.py -d $dataset -m rmse -c 1 -p $party -s $seed -g "$gpu" \
   -e 100 -v 6 --knn-k ${knnk} -nh 1 -ded 100 -ked 100 -nlb 3 -nab 3 --dropout 0.3 -paf 1 \
   > "$outdir"/"$dataset"_fet_p${party}_k${knnk}_s${seed}.log &
   cnt=$((cnt+1))
   if [ $cnt -eq "$num_gpus" ]; then
     wait
     cnt=0
   fi
 done
done

dataset=hdb
outdir=out/ablation-knnk/$dataset
mkdir -p $outdir
for seed in 0 1 2 3 4; do
  for knnk in 1 5 10 20 40 60 80 100; do
    gpu=${gpus[$cnt]}
    python src/script/train_fet.py -d $dataset -m rmse -c 1 -p $party -s $seed -g "$gpu" \
    -e 100 -v 6 --knn-k ${knnk} -nh 1 -ded 100 -ked 100 -nlb 1 -nab 2 --dropout 0.1 -paf 1 \
    > "$outdir"/"$dataset"_fet_p${party}_k${knnk}_noise${noise}_pdo0_s${seed}.log &
    cnt=$((cnt+1))
    if [ $cnt -eq "$num_gpus" ]; then
      wait
      cnt=0
    fi
  done
done

