from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import numpy as np
import pandas as pd
import math


def process_data(df, attack_dict, label_dict):
    label_list = df["taxonomy label"].unique().tolist()

    target_group = label_dict["label"]

    for i in label_list:
        if i not in target_group:
            attack_dict[i] = 0

    df = (
        df.groupby("taxonomy label")
        .apply(lambda df: df.sample(attack_dict[df.name], replace=False))
        .reset_index(drop=True)
    )
    return df


def count_num_attacks(df, n_streams):

    attack_dict = {}

    for idx, name in enumerate(df["taxonomy label"].value_counts().index.tolist()):
        count = df["taxonomy label"].value_counts()[idx]
        min_count = min(n_streams, count)
        attack_dict.update({name: min_count})

    return attack_dict


def log_transformation(df):

    numeric_cols = [col for col in df if df[col].dtype.kind != "O"]

    for col in numeric_cols:
        if col.startswith("pktTime"):
            df[col] += 1
            df[col] = np.log2(df.loc[:, col])
        if col.startswith("pktSize"):
            df[col] = np.log2(df.loc[:, col])

    df = shuffle(df)
    df.reset_index(drop=True, inplace=True)

    return df


def process_train_valid_test(df, data, attack_dict, label_dict):
    label_list = df["taxonomy label"].unique().tolist()
    target_group = label_dict["label"]

    if data == "Test":
        num = label_dict["num_test"]

    elif data == "Valid":
        num = label_dict["num_valid"]

    # Number of rows, that we want to be sampled from each category
    attack_dict.update((k, num) for k in attack_dict)

    for i in label_list:
        if i not in target_group:
            attack_dict[i] = 0

    df = df.groupby("taxonomy label").apply(
        lambda df: df.sample(attack_dict[df.name], replace=False)
    )

    return df


# prepare target
def prepare_targets(y, label_dict):
    attack_categories = [label_dict["label"]]
    ordinal_encoder = OrdinalEncoder(categories=attack_categories)
    y_cat = ordinal_encoder.fit_transform(y.reshape(-1, 1))
    return y_cat


def load_data(data, n_streams, label_dict):
    # Read Training csv file
    df = pd.read_csv(data)

    if "c2s" in df.columns:
        df.drop(["c2s"], axis=1, inplace=True)

    df = shuffle(df)
    df.reset_index(drop=True, inplace=True)

    attack_dict = count_num_attacks(df, n_streams)

    print("attack_dict")
    print(attack_dict)
    df = process_data(df, attack_dict, label_dict)

    df = log_transformation(df)

    df_test = process_train_valid_test(df, "Test", attack_dict, label_dict)
    test_idx = pd.Index([x[1] for x in df_test.index])

    df = df.drop(test_idx).reset_index(drop=True)

    df_val = process_train_valid_test(df, "Valid", attack_dict, label_dict)
    val_idx = pd.Index([x[1] for x in df_val.index])

    df = df.drop(val_idx).reset_index(drop=True)

    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    for idx, name in enumerate(df["taxonomy label"].value_counts().index.tolist()):
        print("Train", name)
        print("Counts:", df["taxonomy label"].value_counts()[idx])

    for idx, name in enumerate(df_val["taxonomy label"].value_counts().index.tolist()):
        print("Valid", name)
        print("Counts:", df_val["taxonomy label"].value_counts()[idx])

    for idx, name in enumerate(df_test["taxonomy label"].value_counts().index.tolist()):
        print("Test", name)
        print("Counts:", df_test["taxonomy label"].value_counts()[idx])

    # create Xtrain, ytrain, Xtest and ytest
    X_train = df.iloc[:, :-1].to_numpy()
    y_train = df.iloc[:, -1].to_numpy()

    X_valid = df_val.iloc[:, :-1].to_numpy()
    y_valid = df_val.iloc[:, -1].to_numpy()

    X_test = df_test.iloc[:, :-1].to_numpy()
    y_test = df_test.iloc[:, -1].to_numpy()

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def feature_scaling(X_train, X_valid, X_test, k):
    # feature scalling
    scaler = MinMaxScaler()

    size_max = np.amax(X_train[:, :k])
    time_max = np.amax(X_train[:, k:])

    size_min = np.amin(X_train[:, :k])
    time_min = np.amin(X_train[:, k:])

    size_min_arr = np.full((k), size_min)
    time_min_arr = np.full((k), time_min)

    size_max_arr = np.full((k), size_max)
    time_max_arr = np.full((k), time_max)

    min_arr = np.concatenate((size_min_arr, time_min_arr))
    max_arr = np.concatenate((size_max_arr, time_max_arr))

    min_max_arr = np.concatenate(([min_arr], [max_arr]), axis=0)

    scaler.fit(min_max_arr)

    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    return X_train, X_valid, X_test


def change_outlier(arr):
    if arr[arr > 1].any():
        arr[arr > 1] = 1
        print("Outlier: yes")
    return arr


def reshape_data(X, k):

    size = np.vstack(X[0, :k])
    time = np.vstack(X[0, k:])

    X_convert = [np.concatenate((size, time), axis=1)]

    for i in range(X.shape[0] - 1):

        size = np.vstack(X[i + 1, :k])
        time = np.vstack(X[i + 1, k:])
        X_concat = np.concatenate((size, time), axis=1)
        X_convert = np.append(X_convert, [X_concat], axis=0)
    return np.array(X_convert)


def define_data(df, k_packets, n_streams, label_dict, model_type):

    # load data, feature scaling
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(
        df, n_streams, label_dict
    )
    X_train, X_valid, X_test = feature_scaling(X_train, X_valid, X_test, k_packets)
    X_train = change_outlier(X_train)
    X_valid = change_outlier(X_valid)
    X_test = change_outlier(X_test)

    y_train_cat = prepare_targets(y_train, label_dict)
    y_valid_cat = prepare_targets(y_valid, label_dict)
    y_test_cat = prepare_targets(y_test, label_dict)

    if model_type != "ann":

        X_train = reshape_data(X_train, k_packets)
        X_valid = reshape_data(X_valid, k_packets)
        X_test = reshape_data(X_test, k_packets)

    return X_train, y_train_cat, X_valid, y_valid_cat, X_test, y_test_cat
