#!/usr/bin/env bash

# Split VFL dataset using vertibench under different alpha values
for p in 10 20; do
  for ns in 0.05; do
    for a in 0.1 0.5 1.0 5.0 10.0 50.0; do
      python src/preprocess/FuzzySplitter.py -d $1 -p $p -kd 4 -ns $ns -a $a &
    done
  done
done
