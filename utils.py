import configparser
import os.path
import sys
from typing import Dict
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import pandas as pd
from tqdm import tqdm
import logging
import pandas as pd
import pickle as pk


def train_test_split(path, train_division=0.8):
    logger = logging.getLogger("Data_splitting_logs")
    logger.setLevel(logging.INFO)
    format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(format)
    logger.addHandler(handler)

    dataset = pd.read_csv(os.path.join(path, "PTC.csv"))
    if dataset is None:
        raise Exception("The provided path is not valid")

    logger.info("Processing the dataset...")
    user_ids = dataset["ID"].unique()
    train_dataset = []
    test_dataset = []
    for user in tqdm(user_ids):
        user_data = dataset[dataset["ID"] == user]
        len_user = len(user_data)

        train_dataset.append(user_data[:int(len_user * train_division)])
        test_dataset.append(user_data[int(len_user * train_division):])

    train_dataset = pd.concat(train_dataset, axis=0)
    test_dataset = pd.concat(test_dataset, axis=0)

    logger.info("Saving the dataset...")
    train_dataset.to_csv(os.path.join(path, "train.csv"))
    test_dataset.to_csv(os.path.join(path, "test.csv"))


def get_available_home_ids(path):
    dataset = pd.read_csv(os.path.join(path, "PTC.csv"))
    return sorted(dataset["ID"].unique())


def evaluate_predictions(logits, labels):
    predicted_labels = np.argmax(logits, 1)
    accuracy = accuracy_score(labels, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(labels, predicted_labels)
    conf_matrix = confusion_matrix(labels, predicted_labels, labels=[0, 1, 2])
    return accuracy, balanced_accuracy, conf_matrix


def read_source_file(path):
    return pd.read_csv(path)


if __name__ == '__main__':
    train_test_split("data", train_division=0.8)
