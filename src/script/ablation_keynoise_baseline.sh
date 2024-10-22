#!/usr/bin/env bash


gpus=(1 2 3 4 5 6 7)
process_per_gpu=2
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
     python src/script/train_solo.py -d $dataset -m acc -c 2 -p $party -sp imp -w 100 -s $seed -g $gpu --key-noise $noise \
     > "$outdir"/"$dataset"_solo_p${party}_k${knnk}_noise${noise}_pdo0_s${seed}.log &
     cnt=$((cnt+1))
     if [ $cnt -eq "$num_gpus" ]; then
       wait
       cnt=0
     fi

     gpu=${gpus[$cnt]}
     python src/script/train_top1.py -d $dataset -m acc -c 2 -p $party -sp imp -w 100 -s $seed -g $gpu -v 6 --knn-k 1 -nh 1 \
     -ded 50 -ked 50 -nlb 1 -nab 1 --dropout 0.0 --key-noise $noise \
     > "$outdir"/"$dataset"_top1_p${party}_k${knnk}_noise${noise}_pdo0_s${seed}.log &
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
     python src/script/train_solo.py -d $dataset -e 50 -m acc -c 10 -p $party -sp imp -w 100 -s $seed -g $gpu --key-noise $noise \
     > "$outdir"/"$dataset"_solo_p${party}_k${knnk}_noise${noise}_pdo0_s${seed}.log &
     cnt=$((cnt+1))
     if [ $cnt -eq "$num_gpus" ]; then
       wait
       cnt=0
     fi
   done
 done
done


batch=0
dataset=mnist
outdir=out/ablation-keynoise/$dataset
mkdir -p $outdir
for seed in 0 1 2 3 4; do
  for noise in 0.02 0.05 0.06 0.07 0.1; do
    for knnk in 100; do
      if [ $fcnt -lt $finish ]; then
        fcnt=$((fcnt+1))    # skip the first 14
        continue
      fi

      gpu=${gpus[$cnt]}
      python src/script/train_top1.py -d $dataset -e 30 -m acc -c 10 -p $party -sp imp -w 100 -s $seed -g $gpu -v 6 --knn-k 1 -nh 1 \
      -ded 50 -ked 50 -nlb 1 -nab 1 --dropout 0.0 --key-noise $noise \
      > "$outdir"/"$dataset"_top1_p${party}_k${knnk}_noise${noise}_pdo0_s${seed}.log &
      cnt=$((cnt+1))
      if [ $cnt -eq "$num_gpus" ]; then
        batch=$((batch+1))
        if [ $batch -eq "$process_per_gpu" ]; then
          wait
          batch=0
        fi
        cnt=0
      fi
    done
  done
done


