#!/usr/bin/env bash

for dataset in gisette mnist; do
  for ns in 0.05; do
    for p in 10 20 30 40 50; do
      python src/preprocess/FuzzySplitter.py -d ${dataset}.libsvm -p $p -kd 4 -ns $ns &
    done
  done
  wait
  echo "Done $dataset"
done
