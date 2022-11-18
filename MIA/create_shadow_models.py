import bz2
import pickle
import os
from math import ceil
from multiprocessing import Process
from pathlib import Path
import pandas as pd
from numpy import load, savez_compressed, ndarray
from pandas import DataFrame, concat
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

HOMEDIR = "./adult/"
dir = "original"
classifier = "rf"
mode = "adult"
kind = "original"
class_name = "class"
data_kind = 'stat'
SHADOW_TYPE = 'rf'
N_MODELS = 6

N_WORKERS = 2

TEST_SIZE = 0.2

BLACK_BOX_TYPE = "rf"



def create_random_forest(X_train: ndarray, y_train: ndarray):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    return rf


def worker(id: int, X: ndarray, y: ndarray, begin: int, end: int):
    print(f"Starting worker {id}. from {begin} to {end}")
    for i in range(begin, end):
        # Creates the working data
        Path(f"{HOMEDIR}/shadow_{data_kind}_{kind}_{classifier}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_{i}").mkdir(
            exist_ok=True, parents=True
        )
        # Splits train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y
        )
        #oversample = RandomOverSampler(sampling_strategy='minority')
        #X_train, y_train = oversample.fit_resample(X_train, y_train)
        #smote = SMOTE(random_state=42)
        #X_train, y_train = smote.fit_resample(X_train, y_train)
        undersample = RandomUnderSampler(sampling_strategy = 'majority')
        X_train, y_train = undersample.fit_resample(X_train, y_train)
        # Saves data on disk
        pickle.dump(X_train, open(f"{HOMEDIR}/shadow_{data_kind}_{kind}_{classifier}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_{i}/train_set_{BLACK_BOX_TYPE}_{SHADOW_TYPE}_{i}.p", 'wb'))
        pickle.dump(y_train, open(f"{HOMEDIR}/shadow_{data_kind}_{kind}_{classifier}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_{i}/train_label_{BLACK_BOX_TYPE}_{SHADOW_TYPE}_{i}.p", 'wb'))
        pickle.dump(X_test, open(f"{HOMEDIR}/shadow_{data_kind}_{kind}_{classifier}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_{i}/test_set_{BLACK_BOX_TYPE}_{SHADOW_TYPE}_{i}.p", 'wb'))
        pickle.dump(y_test, open(f"{HOMEDIR}/shadow_{data_kind}_{kind}_{classifier}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_{i}/test_label_{BLACK_BOX_TYPE}_{SHADOW_TYPE}_{i}.p", 'wb'))
        # Random forest model
        rf = create_random_forest(X_train, y_train)
        print(f"Process {id} - {i} - Model created")
        # Saves model to disk
        with bz2.open(
            f"{HOMEDIR}/shadow_{data_kind}_{kind}_{classifier}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_{i}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_{i}_model.pkl.bz2", "wb"
        ) as f:
            pickle.dump(rf, f)
        # Prediction probability on the training set
        y_prob_train = rf.predict_proba(X_train)
        # Creates a classification report
        train_report = classification_report(y_train, rf.predict(X_train))
        with open(
            f"{HOMEDIR}/shadow_{data_kind}_{kind}_{classifier}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_{i}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_{i}_train_report.txt", "w"
        ) as f:
            f.write(train_report)
        # "IN" dataset
        df_in = DataFrame(y_prob_train)
        df_in["class_label"] = y_train
        df_in["target_label"] = "in"
        # Prediction probability on the test set
        y_prob_test = rf.predict_proba(X_test)
        test_report = classification_report(y_test, rf.predict(X_test))
        with open(
            f"{HOMEDIR}/shadow_{data_kind}_{kind}_{classifier}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_{i}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_{i}_test_report.txt", "w"
        ) as f:
            f.write(test_report)
        df_out = DataFrame(y_prob_test)
        df_out["class_label"] = y_test
        df_out["target_label"] = "out"
        # Concats the two dataframes
        df = concat([df_in, df_out])
        df["target_label"] = df["target_label"].astype("category")
        df.to_pickle(
            f"{HOMEDIR}/shadow_{data_kind}_{kind}_{classifier}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_{i}/{BLACK_BOX_TYPE}_{SHADOW_TYPE}_{i}_prediction_set.pkl.bz2"
        )

    print(f"Ending worker {id}")


def main():
    # Loads dataset
    filename: str = f"{HOMEDIR}/data/{mode}_{data_kind}_{kind}_{classifier}_shadow_labelled.csv"
    dataset_shadow = pd.read_csv(filename, index_col=0)
    label_shadow = dataset_shadow.pop(class_name)
    print("Data correctly read")
    print('data ', dataset_shadow.shape, label_shadow.shape, dataset_shadow.head())
    # Size of the chunk of models for each worker
    chunk_size: int = ceil(N_MODELS / N_WORKERS)
    # List of worker processes
    processes = []
    begin : int = 0
    end : int = 0
    # Parallelizes shadow model training for each worker
    for i in range(N_WORKERS):
        begin = end
        end = begin + chunk_size
        if end > N_MODELS:
            end = N_MODELS
        process = Process(target=worker, args=(i, dataset_shadow, label_shadow, begin, end))
        processes.append(process)
        process.start()
    # Joins the parallel processes
    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
