import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
import itertools


def train(model, model_type, iterator, optimizer, criterion):

    model.train()

    epoch_loss = 0
    y_pred_list, y_list = list(), list()

    for i, data in enumerate(iterator):

        src, label = data

        optimizer.zero_grad()

        if model_type != "ann":
            src = src.permute(1, 0, 2)

        y_pred = model(src)

        loss = criterion(y_pred, label.flatten())

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, y_pred_tags = torch.max(y_pred, dim=1)

        y_pred_list.append(y_pred_tags.detach().cpu().numpy())
        y_list.append(label.flatten().detach().cpu().numpy())

    y_pred_list = [a.flatten().tolist() for a in y_pred_list]
    y_list = [a.flatten().tolist() for a in y_list]

    y_pred_list = list(itertools.chain(*y_pred_list))
    y_list = list(itertools.chain(*y_list))

    acc = accuracy_score(y_list, y_pred_list)

    return epoch_loss / len(iterator), acc


def evaluate(model, model_type, iterator, criterion):

    model.eval()

    epoch_loss = 0
    y_pred_list, y_list = list(), list()

    with torch.no_grad():

        for i, data in enumerate(iterator):

            src, label = data
            if model_type != "ann":
                src = src.permute(1, 0, 2)

            y_pred = model(src)

            loss = criterion(y_pred, label.flatten())
            epoch_loss += loss.item()

            _, y_pred_tags = torch.max(y_pred, dim=1)

            y_pred_list.append(y_pred_tags.detach().cpu().numpy())
            y_list.append(label.flatten().detach().cpu().numpy())

        y_pred_list = [a.flatten().tolist() for a in y_pred_list]
        y_list = [a.flatten().tolist() for a in y_list]

        y_pred_list = list(itertools.chain(*y_pred_list))
        y_list = list(itertools.chain(*y_list))

        acc = accuracy_score(y_list, y_pred_list)

    return epoch_loss / len(iterator), acc


def predict(model, model_type, iterator):

    model.eval()

    y_pred_list, y_list = list(), list()

    with torch.no_grad():

        for i, data in enumerate(iterator):

            X_batch, label = data

            if model_type != "ann":
                X_batch = X_batch.permute(1, 0, 2)

            y_pred = model(X_batch)

            _, y_pred_tags = torch.max(y_pred, 1)

            y_pred_list.append(y_pred_tags.detach().cpu().numpy())
            y_list.append(label.flatten().detach().cpu().numpy())

        y_pred_list = [a.flatten().tolist() for a in y_pred_list]
        y_list = [a.flatten().tolist() for a in y_list]

        y_pred_list = list(itertools.chain(*y_pred_list))
        y_list = list(itertools.chain(*y_list))

    return y_list, y_pred_list


def bi_lstm_train(model, iterator, optimizer, criterion):

    model.train()

    epoch_loss = 0
    y_pred_list, y_list = list(), list()

    for i, data in enumerate(iterator):

        src, label = data

        src_backward = torch.flip(src, (1,))
        src_backward = src_backward.permute(1, 0, 2)
        src = src.permute(1, 0, 2)

        optimizer.zero_grad()

        y_pred = model(src, src_backward)

        loss = criterion(y_pred, label.flatten())

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, y_pred_tags = torch.max(y_pred, dim=1)

        y_pred_list.append(y_pred_tags.detach().cpu().numpy())
        y_list.append(label.flatten().detach().cpu().numpy())

    y_pred_list = [a.flatten().tolist() for a in y_pred_list]
    y_list = [a.flatten().tolist() for a in y_list]

    y_pred_list = list(itertools.chain(*y_pred_list))
    y_list = list(itertools.chain(*y_list))

    acc = accuracy_score(y_list, y_pred_list)

    return epoch_loss / len(iterator), acc


def bi_lstm_evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0
    y_pred_list, y_list = list(), list()

    with torch.no_grad():

        for i, data in enumerate(iterator):

            src, label = data
            src_backward = torch.flip(src, (1,))
            src_backward = src_backward.permute(1, 0, 2)
            src = src.permute(1, 0, 2)

            y_pred = model(src, src_backward)

            loss = criterion(y_pred, label.flatten())
            epoch_loss += loss.item()

            _, y_pred_tags = torch.max(y_pred, dim=1)

            y_pred_list.append(y_pred_tags.detach().cpu().numpy())
            y_list.append(label.flatten().detach().cpu().numpy())

        y_pred_list = [a.flatten().tolist() for a in y_pred_list]
        y_list = [a.flatten().tolist() for a in y_list]

        y_pred_list = list(itertools.chain(*y_pred_list))
        y_list = list(itertools.chain(*y_list))

        acc = accuracy_score(y_list, y_pred_list)

    return epoch_loss / len(iterator), acc


def bi_lstm_predict(model, iterator):

    model.eval()

    y_pred_list, y_list = list(), list()

    with torch.no_grad():

        for i, data in enumerate(iterator):
            X_batch, label = data
            X_batch_backward = torch.flip(X_batch, (1,))
            X_batch_backward = X_batch_backward.permute(1, 0, 2)
            X_batch = X_batch.permute(1, 0, 2)

            y_pred = model(X_batch, X_batch_backward)

            _, y_pred_tags = torch.max(y_pred, 1)

            y_pred_list.append(y_pred_tags.detach().cpu().numpy())
            y_list.append(label.flatten().detach().cpu().numpy())

        y_pred_list = [a.flatten().tolist() for a in y_pred_list]
        y_list = [a.flatten().tolist() for a in y_list]

        y_pred_list = list(itertools.chain(*y_pred_list))
        y_list = list(itertools.chain(*y_list))

    return y_list, y_pred_list


def pre_train(model, iterator, optimizer, criterion):

    model.train()

    epoch_loss = 0

    for i, data in enumerate(iterator):

        src, label = data
        trg, label_ = data

        src = src.permute(1, 0, 2)
        trg = trg.permute(1, 0, 2)

        optimizer.zero_grad()

        output = model(src, trg)

        loss = criterion(output, trg)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def pre_evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0
    enc_outputs = []
    dec_outputs = []

    with torch.no_grad():

        for i, data in enumerate(iterator):

            src, label = data
            trg, label_ = data

            src = src.permute(1, 0, 2)
            trg = trg.permute(1, 0, 2)

            output = model(src, trg)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
