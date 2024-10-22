#!/usr/bin/env bash

mkdir -p data/syn/gisette data/syn/mnist
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2 -O data/syn/gisette/gisette.libsvm.bz2
echo "Unziping gisette.libsvm.bz2"
bzip2 -d data/syn/gisette/gisette.libsvm.bz2 

wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 -O data/syn/mnist/mnist.libsvm.bz2
echo "Unziping mnist.libsvm.bz2"
bzip2 -d data/syn/mnist/mnist.libsvm.bz2
