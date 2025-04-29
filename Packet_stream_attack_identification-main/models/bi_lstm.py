import torch
import torch.nn as nn


class Bi_LSTM(nn.Module):
    def __init__(
        self,
        forward_lstm,
        backward_lstm,
        mlp1,
        mlp2,
        device,
        hid_dim,
        context_dim,
        n_classes,
    ):
        super().__init__()

        self.hidden_dim = hid_dim
        self.forward_lstm = forward_lstm
        self.backward_lstm = backward_lstm
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.fc1 = nn.Linear(context_dim * 2, n_classes)
        self.device = device
        self.context_dim = context_dim

    def forward(self, src, src_backward):

        hidden, cell = self.forward_lstm(src)

        lst_dim = list(hidden.shape)
        cat_h_c = torch.cat((hidden, cell), dim=-1)

        context = self.mlp1(cat_h_c.flatten())

        context = context.view(lst_dim[0], lst_dim[1], self.context_dim * 2)
        (hidden, cell) = torch.split(context, self.context_dim, dim=-1)

        last_hidden = hidden[-1]

        hidden_b, cell_b = self.backward_lstm(src_backward)

        lst_dim_b = list(hidden_b.shape)
        cat_h_c_b = torch.cat((hidden_b, cell_b), dim=-1)

        context_b = self.mlp2(cat_h_c_b.flatten())

        context_b = context_b.view(lst_dim_b[0], lst_dim_b[1], self.context_dim * 2)
        (hidden_b, cell_b) = torch.split(context_b, self.context_dim, dim=-1)

        last_hidden = hidden[-1]
        last_hidden = last_hidden.contiguous().view(-1, self.context_dim)
        last_hidden_b = hidden_b[-1]
        last_hidden_b = last_hidden_b.contiguous().view(-1, self.context_dim)

        fc_input = torch.cat((last_hidden, last_hidden_b), dim=1)

        output = self.fc1(fc_input)

        return output
