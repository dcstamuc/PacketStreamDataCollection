from utils.utils import *
from utils.data_utils_final_orgin import *
from utils.model_utils import *
from utils.train_utils import *
from datasets.data import NetworkDataset
from models.mlp import ANN
from models.lstm import LSTM
from models.bi_lstm import Bi_LSTM
from models.seq2seq import *
from sklearn.metrics import classification_report
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import yaml


def create_args():
    """
    Parses args.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--seed",
        required=True,
        type=int,
        default=0,
        help="seed number for reproducibility",
    )
    parser.add_argument(
        "-k",
        "--k_packets",
        required=True,
        type=int,
        default=5,
        help="number of K packets",
    )
    parser.add_argument(
        "--n_streams",
        required=True,
        type=int,
        default=10000,
        help="number of streams (data)",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="path of input dataset",
    )
    parser.add_argument(
        "--model_type",
        required=True,
        type=str,
        choices=["ann", "lstm", "bi_lstm", "seq2seq"],
        default="ann",
        help="model type for training. choices=['ann', 'lstm', 'bi_lstm', 'seq2seq'] ",
    )
    parser.add_argument(
        "--hidden_dims",
        required=True,
        type=int,
        default=20,
        help="hidden dimensions [20, 40]",
    )
    parser.add_argument(
        "--epochs",
        required=True,
        type=int,
        default=1000,
        help="number of epochs",
    )
    parser.add_argument(
        "--context_dims",
        required=True,
        type=int,
        default=0,
        help="context dimensions [0, 3, 5, 10], None for ANN",
    )
    parser.add_argument(
        "--lr",
        required=True,
        type=float,
        default=0.001,
        help="learning rate. default=0.001",
    )
    parser.add_argument(
        "--batch_size",
        required=True,
        type=int,
        default=20,
        help="batch size. default=20",
    )
    parser.add_argument(
        "--n_layers",
        required=True,
        type=int,
        default=3,
        help="number of layers. default=3",
    )
    parser.add_argument(
        "-r",
        "--result",
        required=True,
        type=str,
        default="results",
        help="path for results",
    )
    parser.add_argument(
        "--exp_str", required=True, type=str, help="experiment folder name e.g, ANN"
    )
    parser.add_argument(
        "--pre_model_path",
        type=str,
        default=None,
        help="path for pretrained seq2seq model checkpoint for initialization",
    )
    parser.add_argument("--label_config", type=str, help="yaml file for attack label")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = create_args()
    seed_everthing(args.seed)

    label_dict = yaml.safe_load(open(args.label_config, "r"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = {
        "epochs": args.epochs,
        "context_dims": args.context_dims,
        "hidden_dims": args.hidden_dims,
        "k": args.k_packets,
        "n_streams": args.n_streams,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "n_layers": args.n_layers,
    }

    X_train, y_train_cat, X_valid, y_valid_cat, X_test, y_test_cat = define_data(
        args.input, params["k"], params["n_streams"], label_dict, args.model_type
    )

    train_dataset = NetworkDataset(X_train, y_train_cat)
    train_iterator = DataLoader(
        dataset=train_dataset, batch_size=params["batch_size"], shuffle=True
    )

    valid_dataset = NetworkDataset(X_valid, y_valid_cat)
    valid_iterator = DataLoader(dataset=valid_dataset, batch_size=params["batch_size"])

    test_dataset = NetworkDataset(X_test, y_test_cat)
    test_iterator = DataLoader(dataset=test_dataset, batch_size=params["batch_size"])
    input_dim = 2

    if args.model_type == "ann":
        input_dim = int(params["k"]) * 2

        model = ANN(
            device, params["hidden_dims"], input_dim, int(label_dict["n_classes"])
        ).to(device)
    if args.model_type == "lstm":
        enc = Encoder(input_dim, params["hidden_dims"], params["n_layers"])
        model = LSTM(
            enc, device, params["hidden_dims"], int(label_dict["n_classes"])
        ).to(device)
    if args.model_type == "bi_lstm":
        forward_lstm = Encoder(
            input_dim,
            params["hidden_dims"],
            params["n_layers"],
        )
        backward_lstm = Encoder(
            input_dim,
            params["hidden_dims"],
            params["n_layers"],
        )

        # 20 * 2 * 2 = 80
        mlp_hid_dims = params["hidden_dims"] * params["n_layers"] * 2
        mlp_context_dims = params["context_dims"] * params["n_layers"] * 2

        mlp1 = MLP1(mlp_hid_dims, mlp_context_dims)
        mlp2 = MLP1(mlp_hid_dims, mlp_context_dims)

        model = Bi_LSTM(
            forward_lstm,
            backward_lstm,
            mlp1,
            mlp2,
            device,
            params["hidden_dims"],
            params["context_dims"],
            int(label_dict["n_classes"]),
        ).to(device)

    if args.model_type == "seq2seq":
        output_dim = 2
        enc = Encoder(input_dim, params["hidden_dims"], params["n_layers"])

        # 20 * 2 * 2 = 80
        mlp_hid_dims = params["hidden_dims"] * params["n_layers"] * 2
        mlp_context_dims = params["context_dims"] * params["n_layers"] * 2

        mlp1 = MLP1(mlp_hid_dims, mlp_context_dims)

        model = LSTM_MLP(
            enc,
            mlp1,
            device,
            params["hidden_dims"],
            params["context_dims"],
            int(label_dict["n_classes"]),
        ).to(device)

    result_path = Path(f"{args.result}/{args.exp_str}")
    result_path.mkdir(parents=True, exist_ok=True)

    model.apply(init_weights)
    model = model.float()

    if args.pre_model_path and args.model_type == "seq2seq":

        model.load_state_dict(torch.load(args.pre_model_path), strict=False)

    print(f"Model type: {args.model_type}")
    print(f"The model has {count_parameters(model):,} trainable parameters")

    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.CrossEntropyLoss()

    best_valid_acc = 0
    epoch_train_loss = []
    epoch_valid_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for epoch in range(params["epochs"]):

        start_time = time.time()

        if args.model_type != "bi_lstm":

            train_loss, train_acc = train(
                model, args.model_type, train_iterator, optimizer, criterion
            )
            valid_loss, valid_acc = evaluate(
                model, args.model_type, valid_iterator, criterion
            )

        else:
            train_loss, train_acc = bi_lstm_train(
                model, train_iterator, optimizer, criterion
            )
            valid_loss, valid_acc = bi_lstm_evaluate(model, valid_iterator, criterion)

        epoch_train_loss.append(train_loss)
        epoch_valid_loss.append(valid_loss)

        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if best_valid_acc < valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch + 1
            train_loss_ = train_loss
            valid_loss_ = valid_loss
            print(f"Best Epoch: {best_epoch:02}")
            print(f"\tTrain Loss: {train_loss_:.5f} | Train Acc: {train_acc:5f}")
            print(f"\tValid Loss: {valid_loss_:.5f} | Best Valid Acc: {valid_acc:5f}")
            torch.save(model.state_dict(), f"{result_path}/best_model.pt")

        if epoch % 10 == 0:
            print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.5f} | Train Acc: {train_acc:5f}")
            print(f"\tValid Loss: {valid_loss:.5f} | Valid Acc: {valid_acc:5f}")

    plot_loss(
        np.linspace(1, params["epochs"], params["epochs"]).astype(int),
        epoch_train_loss,
        "Train",
        result_path,
    )
    plot_loss(
        np.linspace(1, params["epochs"], params["epochs"]).astype(int),
        epoch_valid_loss,
        "Validation",
        result_path,
    )
    plot_acc(
        np.linspace(1, params["epochs"], params["epochs"]).astype(int),
        epoch_train_acc,
        "Train",
        result_path,
    )
    plot_acc(
        np.linspace(1, params["epochs"], params["epochs"]).astype(int),
        epoch_valid_acc,
        "Validation",
        result_path,
    )

    model.load_state_dict(torch.load(f"{result_path}/best_model.pt"))

    if args.model_type != "bi_lstm":

        # Predict the model with training, validation, and testing dataset
        y_train_list, y_train_pred_list = predict(
            model, args.model_type, train_iterator
        )
        y_valid_list, y_valid_pred_list = predict(
            model, args.model_type, valid_iterator
        )
        y_test_list, y_test_pred_list = predict(model, args.model_type, test_iterator)

    else:
        # Predict the model with training, validation, and testing dataset
        y_train_list, y_train_pred_list = bi_lstm_predict(model, train_iterator)
        y_valid_list, y_valid_pred_list = bi_lstm_predict(model, valid_iterator)
        y_test_list, y_test_pred_list = bi_lstm_predict(model, test_iterator)

    plot_cm(y_train_list, y_train_pred_list, "train", label_dict["label"], result_path)
    plot_cm(y_valid_list, y_valid_pred_list, "valid", label_dict["label"], result_path)
    plot_cm(y_test_list, y_test_pred_list, "test", label_dict["label"], result_path)

    target_names = ["class 0", "class 1", "class 2", "class 3"]

    train_report_dict = classification_report(
        y_train_list,
        y_train_pred_list,
        target_names=target_names,
        output_dict=True,
        digits=5,
    )
    valid_report_dict = classification_report(
        y_valid_list,
        y_valid_pred_list,
        target_names=target_names,
        output_dict=True,
        digits=5,
    )
    test_report_dict = classification_report(
        y_test_list,
        y_test_pred_list,
        target_names=target_names,
        output_dict=True,
        digits=5,
    )

    best_result = {
        "best_train_loss": train_loss_,
        "best_valid_loss": valid_loss_,
        "best_epoch": best_epoch,
    }
    save_classification_report(
        result_path,
        params,
        best_result,
        train_report_dict,
        valid_report_dict,
        test_report_dict,
    )
