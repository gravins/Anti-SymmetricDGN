#!/bin/bash


model= # specify here the model name (see file conf.py)
data= # specify here the dataset name (see file utils/__init__.py)
export CUDA_VISIBLE_DEVICES=0

nohup python3 -u main.py --data_name $data --model_name $model --save_dir saving_dir >out_$model\_$data 2>err_$model\_$data &