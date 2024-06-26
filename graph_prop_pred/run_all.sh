#!/bin/bash


export CUDA_VISIBLE_DEVICES=0

data=GraphProp
model= # specify here the model name

task=dist
nohup python3 -u main.py --data_name $data --task $task --model_name $model --save_dir save_dir_GraphProp >out_$model\_$data\_$task 2>err_$model\_$data\_$task &

task=ecc
nohup python3 -u main.py --data_name $data --task $task --model_name $model --save_dir save_dir_GraphProp >out_$model\_$data\_$task 2>err_$model\_$data\_$task &

task=diam
nohup python3 -u main.py --data_name $data --task $task --model_name $model --save_dir save_dir_GraphProp >out_$model\_$data\_$task 2>err_$model\_$data\_$task &

 
