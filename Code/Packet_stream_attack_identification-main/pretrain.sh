#!/bin/bash

python pretrain.py -s 0 -k 3 --n_streams 10000 -i data/0901_0930_K_3.csv --hidden_dims 20 --epochs 10000 --context_dims 3 --lr 0.001 --batch_size 20 --n_layers 3 -r pre_results --exp_str 0_hid20_context3_k3_s0 --label_config config_label.yaml  