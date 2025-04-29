import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers)

    def forward(self, src):

        # src = [K, batch size, feature size] -> [10, 1, 3]

        outputs, (hidden, cell) = self.rnn(src)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim] -> [1, 1, 10]
        # cell = [n layers * n directions, batch size, hid dim] -> [1, 1, 10]

        # outputs are always from the top hidden layer

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(output_dim, hid_dim, n_layers)

        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden, cell):

        # input = [batch size, feature size] -> (1, 3)
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        output, (hidden, cell) = self.rnn(input, (hidden, cell))

        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden, cell


class MLP1(nn.Module):
    def __init__(self, hid_dim, context_dim):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, context_dim)
        self.hid_dim = hid_dim

    def forward(self, cat_h_c):
        # Hidden [n layers, batch, hid dim]
        cat_h_c = cat_h_c.contiguous().view(-1, self.hid_dim)
        cat_h_c = self.fc1(cat_h_c)

        return cat_h_c


class MLP2(nn.Module):
    def __init__(self, hid_dim, context_dim):
        super().__init__()
        self.fc1 = nn.Linear(context_dim, hid_dim)

    def forward(self, cat_h_c):
        # Hidden [n layers, batch, hid dim]
        cat_h_c = self.fc1(cat_h_c)

        return cat_h_c


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, mlp1, mlp2, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        #         self.classifier = classifier
        self.device = device

        assert (
            encoder.hid_dim == decoder.hid_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg):

        # src = [K, batch size, feature size]
        # trg = [batch size, feature size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        lst_dim = list(hidden.shape)

        cat_h_c = torch.cat((hidden, cell), dim=-1)

        context_vec = self.mlp1(cat_h_c.flatten())
        cat_h_c = self.mlp2(context_vec)

        cat_h_c = cat_h_c.view(lst_dim[0], lst_dim[1], lst_dim[2] * 2)
        (hidden, cell) = torch.split(cat_h_c, lst_dim[2], dim=-1)

        # first input (Start token = 0) to the decoder
        input = torch.zeros_like(trg[0])

        for t in range(trg_len):

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states

            prediction, hidden, cell = self.decoder(input, hidden, cell)
            clip_preds = torch.nn.functional.relu(prediction)

            input = clip_preds

            outputs[t] = clip_preds

        return outputs


class LSTM_MLP(nn.Module):
    def __init__(self, encoder, mlp1, device, hid_dim, context_dim, n_classes):
        super().__init__()

        self.hidden_dim = hid_dim
        self.context_dim = context_dim
        self.encoder = encoder
        self.mlp1 = mlp1
        self.fc1 = nn.Linear(context_dim, n_classes)
        self.device = device

    def forward(self, src):
        # Hidden [n layers, batch, hid dim] [2, 20, 20]
        hidden, cell = self.encoder(src)

        lst_dim = list(hidden.shape)

        # MLP1 hid_dim 80
        # HID_DIM=20, N_LAYERS=2, BATCH_SIZE=20, L_R=0.01
        # lst_dim [2, 20, 20]
        # cat_h_c torch.Size([2, 20, 40])
        # AFTER context torch.Size([20, 40])

        cat_h_c = torch.cat((hidden, cell), dim=-1)

        context = self.mlp1(cat_h_c.flatten())

        context = context.view(lst_dim[0], lst_dim[1], self.context_dim * 2)
        (hidden, cell) = torch.split(context, self.context_dim, dim=-1)

        last_hidden = hidden[-1]

        # context.shape
        # torch.Size([2, 20, 20])
        # hidden.shape
        # torch.Size([2, 20, 10])
        # last_hidden.shape
        # torch.Size([20, 10])
        # int(lst_dim[2] / 2)
        # 10
        # BEFORE last_hidden.shape
        # torch.Size([20, 10])
        # output.shape
        # torch.Size([20, 4])

        last_hidden = last_hidden.contiguous().view(-1, self.context_dim)

        output = self.fc1(last_hidden)

        return output
