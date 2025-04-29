import torch
import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Linear:
        for name, param in m.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.uniform_(m._parameters[name])
    if type(m) == nn.LSTM:
        # initialize biases and weights
        for name, param in m.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.uniform_(m._parameters[name])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_model_dict(model_dict):
    print("=" * 30)
    for key in sorted(model_dict.keys()):
        parameter = model_dict[key]
        print(key)
        print(parameter.size())
        print(parameter)
    print("=" * 30)
