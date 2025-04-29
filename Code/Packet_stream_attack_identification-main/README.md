# Packet stream attack identification 

This project is related to Deep Sequence Models for Packet Stream Analysis and Early Decisions, LCN 2022.
*Minji Kim, Dongeun Lee, Kookjin Lee, Doowon Kim, Sangman Lee, Jinoh Kim*

There are available 4 different models which are ANN, LSTM, Bi-LSTM, Seq2Seq for packet stream attack identification.


## Pretraining
Before training Seq2Seq, we need to pretrain as following:

```
python pretrain.py -s 0 -k 5 --n_streams 10000 -i INPUT_FILE_PATH --hidden_dims 20 --epochs 10000 --context_dims 3 --lr 0.001 --batch_size 20 --n_layers 3 -r RESULT_PATH --exp_str EXP_STR --label_config config_label.yaml  
```

- `-s`: random seed number e.g., 0
- `-k`: number of k packets e.g., 5
- `--n_streams`: number of streams (data) for training e.g., 10000
- `-i`: path of input dataset e.g., 0901_0930_k_5.csv
- `--hidden_dims`: hidden dimensions e.g., 20
- `--epochs`: number of epochs e.g., 10000
- `--context_dims`: context dimensions e.g., 3
- `--lr`: learning rate e.g., 0.001
- `--batch_size`: batch size e.g., 20
- `--n_layers`: number of layers e.g., 3
- `-r`: path for saving results
- `--exp_str`: experiment folder name e.g., 0_pretrained_seq2seq
- `--label_config`: yaml file for attack label e.g., config_label.yaml

### Pretraining Seq2Seq

*You can find example of below shell script in pretrain.sh*

```
python main.py -s 0 -k 3 --n_streams 10000 -i data/0901_0930_K_3.csv --model_type seq2seq --hidden_dims 20 --epochs 10000 --context_dims 3 --lr 0.001 --batch_size 20 --n_layers 3 -r results --exp_str 0_seq2seq_hid20_context3_k3_s0 --pre_model_path pre_results/0_hid20_context3_k3_s0/best_model.pt --label_config config_label.yaml
```

## Main training

For main training, please use below command (This code includes training and evaluating model) :

```
python main.py -s 0 -k 5 --n_streams 10000 -i INPUT_FILE_PATH --model_type MODEL_TYPE --hidden_dims 20 --epochs 1000 --context_dims 3 --lr 0.001 --batch_size 20 --n_layers 3 -r RESULT_PATH --exp_str EXP_STR --pre_model_path PRETRAINED_MODEL_FILE_PATH --label_config config_label.yaml  
```

- `-s`: random seed number e.g., 0
- `-k`: number of k packets e.g., 5
- `--n_streams`: number of streams (data) for training e.g., 1000
- `-i`: path of input dataset e.g., 0901_0930_k_5.csv
- `--model_type`:model type for training. available choices=['ann', 'lstm', 'bi_lstm', 'seq2seq'] e.g., ann
- `--hidden_dims`: hidden dimensions e.g., 20
- `--epochs`: number of epochs e.g., 10000
- `--context_dims`: context dimensions e.g., 3
- `--lr`: learning rate e.g., 0.001
- `--batch_size`: batch size e.g., 20
- `--n_layers`: number of layers e.g., 3
- `-r`: path for saving results
- `--exp_str`: experiment folder name e.g., 0_pretrained_seq2seq
- `--pre_model_path`: path for pretrained seq2seq model checkpoint for initialization e.g., 0_pretrained_seq2seq/best_model.pt
- `--label_config`: yaml file for attack label e.g., config_label.yaml

### Preview of config_label.yaml

```
# number of label classes
n_classes:
  4
# label lists in order e.g., Unlabeled -> 0, HTTP -> 1, Alpha ->2, Multi ->3
label:
  - Unlabeled
  - HTTP
  - Alpha
  - Multi
# number of each test class e.g., Unlabeled 100, HTTP 100, Alpha 100, Multi 100
num_test:
  100
# number of each validation class e.g., Unlabeled 200, HTTP 200, Alpha 200, Multi 200
num_valid:
  200
```

*You can find examples of below shell scripts in train.sh*

### ANN (MLP)

```
python main.py -s 0 -k 3 --n_streams 10000 -i data/0901_0930_K_3.csv --model_type ann --hidden_dims 20 --epochs 1000 --context_dims 0 --lr 0.001 --batch_size 20 --n_layers 3 -r results --exp_str 0_ann_hid20_k3_s0 --label_config config_label.yaml
```

### LSTM (FWD)

```
python main.py -s 0 -k 3 --n_streams 10000 -i data/0901_0930_K_3.csv --model_type lstm --hidden_dims 20 --epochs 1000 --context_dims 3 --lr 0.001 --batch_size 20 --n_layers 3 -r results --exp_str 0_lstm_hid20_context3_k3_s0 --label_config config_label.yaml
```
### Bidirectional LSTM (BI)

```
python main.py -s 0 -k 3 --n_streams 10000 -i data/0901_0930_K_3.csv --model_type bi_lstm --hidden_dims 20 --epochs 1000 --context_dims 3 --lr 0.001 --batch_size 20 --n_layers 3 -r results --exp_str 0_bi_lstm_hid20_context3_k3_s0 --label_config config_label.yaml
```

### Seq2Seq (SEQ)
```
python main.py -s 0 -k 3 --n_streams 10000 -i data/0901_0930_K_3.csv --model_type seq2seq --hidden_dims 20 --epochs 1000 --context_dims 3 --lr 0.001 --batch_size 20 --n_layers 3 -r results --exp_str 0_seq2seq_hid20_context3_k3_s0 --label_config config_label.yaml
```




## Issues
- Please report all issues on the public forum or contact dcs.tamuc@gmail.com

## License

© This project is licensed under the Apache-2.0 license
