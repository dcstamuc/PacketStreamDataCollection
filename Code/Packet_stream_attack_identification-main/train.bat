python main.py -s 0 -k 20 --n_streams 10000 -i 20200701_0931_K_20_T_1.0s.csv --model_type ann --hidden_dims 20 --epochs 1000 --context_dims 0 --lr 0.001 --batch_size 20 --n_layers 3 -r results --exp_str 0_ann_hid20_k10_s0 --label_config config_label.yaml

python main.py -s 0 -k 20 --n_streams 10000 -i 20200701_0931_K_20_T_1.0s.csv --model_type lstm --hidden_dims 20 --epochs 1000 --context_dims 3 --lr 0.001 --batch_size 20 --n_layers 3 -r results --exp_str 0_lstm_hid20_context3_k10_s0 --label_config config_label.yaml

python main.py -s 0 -k 20 --n_streams 10000 -i 20200701_0931_K_20_T_1.0s.csv --model_type bi_lstm --hidden_dims 20 --epochs 1000 --context_dims 3 --lr 0.001 --batch_size 20 --n_layers 3 -r results --exp_str 0_bi_lstm_hid20_context3_k10_s0 --label_config config_label.yaml

python pretrain.py -s 0 -k 20 --n_streams 10000 -i 20200701_0931_K_20_T_1.0s.csv --hidden_dims 20 --epochs 1000 --context_dims 3 --lr 0.001 --batch_size 20 --n_layers 3 -r pre_results --exp_str 0_hid20_context3_k3_s0 --label_config config_label.yaml  

python main.py -s 0 -k 20 --n_streams 10000 -i 20200701_0931_K_20_T_1.0s.csv --model_type seq2seq --hidden_dims 20 --epochs 1000 --context_dims 3 --lr 0.001 --batch_size 20 --n_layers 3 -r results --exp_str 0_seq2seq_hid20_context3_k10_s0 --pre_model_path pre_results/0_hid20_context3_k3_s0/best_model.pt --label_config config_label.yaml