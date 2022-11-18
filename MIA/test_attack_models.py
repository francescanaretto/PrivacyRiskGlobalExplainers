from typing import Iterable, List
from pandas import read_pickle
from numpy import concatenate, load
import pandas as pd
from pandas import concat, DataFrame
from sklearn.metrics import classification_report
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
#from create_shadow_models import HOMEDIR
from sklearn.metrics import precision_recall_fscore_support as score

from create_shadow_models import (
    HOMEDIR,
    dir,
    classifier,
    mode,
    kind,
    class_name,
    data_kind,
    N_MODELS,
    N_WORKERS,
    TEST_SIZE,
    BLACK_BOX_TYPE,
    SHADOW_TYPE,
    create_random_forest,
)
from torch import nn
from sklearn.ensemble import RandomForestClassifier
#from classifier_wrapper import pytorch_classifier_wrapper

len_labels = 2
class NetCopy(nn.Module):
    def __init__(self):
        super().__init__()
        ### FOR XAVIER INITIALIZATION
        self.fc1 = nn.Linear(236, 128)  # fc stays for 'fully connected'
        nn.init.xavier_normal_(self.fc1.weight)
        self.drop = nn.Dropout(0.1)
        self.fc4 = nn.Linear(128, 30)
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = torch.tanh(self.fc1(x))
        x = self.fc4(self.drop(x))
        return F.softmax(x, dim=1)
'''class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ### FOR XAVIER INITIALIZATION
        self.fc1 = nn.Linear(30, 128)  # fc stays for 'fully connected'
        nn.init.xavier_normal_(self.fc1.weight)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 128)  # fc stays for 'fully connected'
        nn.init.xavier_normal_(self.fc2.weight)
        self.drop = nn.Dropout(0.1)
        self.fc3 = nn.Linear(128, 128)  # fc stays for 'fully connected'
        nn.init.xavier_normal_(self.fc3.weight)
        self.drop = nn.Dropout(0.1)
        self.fc4 = nn.Linear(128, 15)
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(self.drop(x))
        return F.softmax(x, dim=1)'''


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ### FOR XAVIER INITIALIZATION
        self.fc1 = nn.Linear(236, 128)  # fc stays for 'fully connected'
        nn.init.xavier_normal_(self.fc1.weight)
        self.drop = nn.Dropout(0.3)
        self.fc4 = nn.Linear(128, 30)
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = torch.tanh(self.fc1(x))
        x = self.fc4(self.drop(x))
        return F.softmax(x, dim=1)

def create_dt():
    print("Loading dt")
    model = read_pickle(f"{HOMEDIR}/{dir}/dt_{mode}_{kind}.sav")
    return model

def create_tabnet():
    print("Loading tabnet")
    model = read_pickle(f"{HOMEDIR}/{dir}/tabnet_{mode}_{kind}.sav")
    return model
def create_trepan():
    print("Loading trepan")
    model = read_pickle(f"{HOMEDIR}/{dir}/{classifier}_{mode}_{kind}.sav")
    return model
def create_rf():
    print("Loading rf")
    model = read_pickle(f"{HOMEDIR}/{dir}/rf_{mode}_{kind}.sav")
    return model

def create_attack_dataframe(X_prob, y, target_label):
    df = DataFrame(X_prob)
    df["class_label"] = y
    df["target_label"] = target_label
    return df


def create_inout_dataframe(X_train_prob, y_train, X_test_prob, y_test):
    df_in = create_attack_dataframe(X_train_prob, y_train, 'in')
    df_out = create_attack_dataframe(X_test_prob, y_test, 'out')
    df = concat([df_in, df_out])
    df["target_label"] = df["target_label"].astype("category")
    #if BLACK_BOX_TYPE == 'nn':
        #df = torch.tensor(df.values.astype(np.float64))

    return df


def create_attack_dataset_artificial():
    df_list = []
    for label in range(len_labels):
        try:
            loaded = load(
                f"{HOMEDIR}/attack_{data_kind}_{kind}_{classifier}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_label_{label}/data.npz",
                allow_pickle=True,
            )
            df = DataFrame(loaded["X_test"])
            df["class_label"] = label
            df["target_label"] = loaded["y_test"]
            df_list.append(df)
        except:
            continue
    df = concat(df_list)
    return df


def create_attack_dataset_from_bb(model):
    X_test = pd.read_csv(f"{HOMEDIR}/data/{mode}_original_test_set.csv")
    X_train = pd.read_csv(f"{HOMEDIR}/data/{mode}_original_train_set.csv")
    y_train = pd.read_csv(f"{HOMEDIR}/data/{mode}_original_train_label.csv")
    y_test = pd.read_csv(f"{HOMEDIR}/data/{mode}_original_test_label.csv")
    X_test.pop('Unnamed: 0')
    #X_test.pop('Unnamed: 0.1')
    X_train.pop('Unnamed: 0')
    #X_train.pop('Unnamed: 0.1')
    y_train.pop('Unnamed: 0')
    y_test.pop('Unnamed: 0')
    if BLACK_BOX_TYPE == 'nn':
        X_test_nn = torch.tensor(X_test.values.astype(np.float64))
        X_train_nn = torch.tensor(X_train.values.astype(np.float64))
        y_train_nn = torch.tensor(y_train.values.astype(np.float64))
        y_test_nn = torch.tensor(y_test.values.astype(np.float64))


    #print('prova ', X_train.head(), y_train)
    #print('cfvgbhjkjnlkmò ', X_train.shape, X_test.shape)
    if BLACK_BOX_TYPE == 'rf' or BLACK_BOX_TYPE == 'dt' or BLACK_BOX_TYPE == 'trepan_rf' or BLACK_BOX_TYPE == 'trepan_nn' or BLACK_BOX_TYPE == 'trepan_nn_over':
        X_train_prob = model.predict_proba(X_train)
        X_test_prob = model.predict_proba(X_test)
    else:
        model.eval()
        X_train_prob = model(X_train_nn.float())
        X_test_prob = model(X_test_nn.float())
        X_test_prob = X_test_prob.detach().numpy()
        X_train_prob = X_train_prob.detach().numpy()
    df = create_inout_dataframe(X_train_prob, y_train, X_test_prob, y_test)
    #print('target', df['target_label'].value_counts().sum())
    #print('labels ', df['class_label'].value_counts().sum())
    return df


def test_all(data: DataFrame, attack_models: List) -> str:
    X_test = data.drop(columns=["class_label", "target_label"])
    y_test = data["target_label"]
    predictions = [model.predict(X_test) for model in attack_models]
    max_predictions = [max(p, key=p.count) for p in zip(*predictions)]
    report = classification_report(y_test, max_predictions)
    precision, recall, fscore, support = score(y_test, max_predictions, average=None)
    print('Precision class IN ',precision[0], '\n Recall class IN ',recall[0], '\n F-1 class IN ', fscore[0])
    return report


def test_single(bb_data: DataFrame) -> Iterable:
    for label, label_df in bb_data.groupby("class_label", sort=False):
        #print(f"Label = {label}")
        X = label_df.drop(columns=["class_label", "target_label"])
        y = label_df["target_label"]
        try:
            attack_model = read_pickle(
                f"{HOMEDIR}/attack_{data_kind}_{kind}_{classifier}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_label_{label}/model.pkl.bz2"
            )
            y_pred = attack_model.predict(X)
            #print('y pred')
        except:
            print('entro in except')
            continue
        yield label, y, y_pred


def test_attack_model(dataset, name, n_lab):
    #print("Single")
    y_true_list = []
    y_pred_list = []
    #print('test attack single')
    for label, y_true_label, y_pred_label in test_single(dataset):
        #print('in test attack model')
        report = classification_report(y_true_label, y_pred_label)
        with open(
            f"{HOMEDIR}/attack_{data_kind}_{kind}_{classifier}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_label_{label}/test_single_{name}.txt",
            "w",
        ) as f:
            f.write(report)
        y_true_list.append(y_true_label)
        #print('y true list ', y_true_list)
        y_pred_list.append(y_pred_label)
    y_true = concatenate(y_true_list)
    y_pred = concatenate(y_pred_list)
    report = classification_report(y_true, y_pred)
    with open(
        f"{HOMEDIR}/attack_{data_kind}_{kind}_{classifier}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_test_single_complete_{name}.txt",
        "w",
    ) as f:
        f.write(report)
    #print("All")
    #print("Loading attack models")
    attack_models = [
            read_pickle(
                f"{HOMEDIR}/attack_{data_kind}_{kind}_{classifier}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_label_{label}/model.pkl.bz2"
            )
        for label in range(n_lab)
    ]
    report = test_all(dataset, attack_models)

    with open(
        f"{HOMEDIR}/attack_{data_kind}_{kind}_{classifier}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_test_all_{name}.txt", "w"
    ) as f:
        f.write(report)


def main():
    print(BLACK_BOX_TYPE)
    if BLACK_BOX_TYPE == "nn":
        filename = f"{HOMEDIR}/{dir}/nn_{mode}_{kind}.sav"
        model = torch.load(filename)
    elif BLACK_BOX_TYPE == 'rf':
        model = create_rf()
    elif BLACK_BOX_TYPE == 'trepan_rf' or BLACK_BOX_TYPE == 'trepan_nn' or BLACK_BOX_TYPE == 'trepan_nn_over':
        model = create_trepan()
    elif BLACK_BOX_TYPE == 'dt':
        model = create_dt()
    elif BLACK_BOX_TYPE == 'tabnet':
        model = create_tabnet()

    print("Reconstructing artificial dataset")
    d_artificial = create_attack_dataset_artificial()
    print("Create attack model from black box")
    d_bb = create_attack_dataset_from_bb(model)
    print("Testing attack models on artificial data")
    test_attack_model(d_artificial, "local", len_labels)
    print("Testing attack models on black box data")
    test_attack_model(d_bb, "bb", n_lab=len_labels)


if __name__ == "__main__":
    main()
