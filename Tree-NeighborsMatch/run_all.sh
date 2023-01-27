#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
model= # specify here the model name (see file conf.py)
dim= # specify here the embedding dim
epsilon= # specify here the value of epsilon
gamma= # specify here the the value of gamma
activ_fun= # specify here the activation function

nohup python3 -u run-adgn-2-8.py --dim $dim --epsilon $epsilon --gamma $gamma --activ_fun $activ_fun >out_$model\_$dim\_$epsilon\_$gamma\_$activ_fun 2>err_$model\_$dim\_$epsilon\_$gamma\_$activ_fun &