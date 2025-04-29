from utils.utils import *
from utils.data_utils import *
from utils.model_utils import *
from utils.train_utils import *
from datasets.data import NetworkDataset
from models.seq2seq import *
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
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
        default=10000,
        help="number of epochs",
    )
    parser.add_argument(
        "--context_dims",
        required=True,
        type=int,
        default=3,
        help="context dimensions [3, 5, 10]",
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
        "--exp_str",
        required=True,
        type=str,
        help="experiment folder name e.g, 0_pretrained_seq2seq",
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
        args.input, params["k"], params["n_streams"], label_dict, "seq2seq"
    )

    train_dataset = NetworkDataset(X_train, y_train_cat)
    train_iterator = DataLoader(
        dataset=train_dataset, batch_size=params["batch_size"], shuffle=True
    )

    valid_dataset = NetworkDataset(X_valid, y_valid_cat)
    valid_iterator = DataLoader(dataset=valid_dataset, batch_size=params["batch_size"])

    test_dataset = NetworkDataset(X_test, y_test_cat)
    test_iterator = DataLoader(dataset=test_dataset, batch_size=params["batch_size"])

    input_dim, output_dim = 2, 2

    enc = Encoder(input_dim, params["hidden_dims"], params["n_layers"])
    dec = Decoder(output_dim, params["hidden_dims"], params["n_layers"])

    # 20 * 2 * 2 = 80
    mlp_hid_dims = params["hidden_dims"] * params["n_layers"] * 2
    mlp_context_dims = params["context_dims"] * params["n_layers"] * 2

    mlp1 = MLP1(mlp_hid_dims, mlp_context_dims)
    mlp2 = MLP2(mlp_hid_dims, mlp_context_dims)

    model = Seq2Seq(enc, dec, mlp1, mlp2, device).to(device)
    model.apply(init_weights)

    model = model.float()

    print(f"The model has {count_parameters(model):,} trainable parameters")

    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.MSELoss()

    best_valid_loss = float("inf")
    epoch_train_loss = []
    epoch_valid_loss = []

    result_path = Path(f"{args.result}/{args.exp_str}")
    result_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(params["epochs"]):

        start_time = time.time()

        train_loss = pre_train(model, train_iterator, optimizer, criterion)
        valid_loss = pre_evaluate(model, valid_iterator, criterion)

        epoch_train_loss.append(train_loss)
        epoch_valid_loss.append(valid_loss)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            train_loss_ = train_loss
            best_epoch = epoch + 1
            print(f"Best Epoch: {best_epoch:02}")
            print(f"\tTrain Loss: {train_loss_:.5f} ")
            print(f"\tBest Valid Loss: {best_valid_loss:.5f} ")
            torch.save(model.state_dict(), f"{result_path}/best_model.pt")

        if epoch % 10 == 0:
            print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.5f}")
            print(f"\tValid Loss: {valid_loss:.5f}")

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

    dic = defaultdict(lambda: defaultdict(dict))
    i = 0
    dic[i]["hidden_dims"] = params["hidden_dims"]
    dic[i]["n_layers"] = params["n_layers"]
    dic[i]["batch_size"] = params["batch_size"]
    dic[i]["learning_rate"] = params["lr"]
    dic[i]["max_epochs"] = params["epochs"]
    dic[i]["Best_epoch"] = best_epoch
    dic[i]["Train_loss"] = train_loss_
    dic[i]["Best_Valid_loss"] = best_valid_loss

    df = pd.DataFrame.from_dict(dic, orient="index")
    df.to_csv(f"{result_path}/pretrain_results.csv", index=False)
