import configparser
from typing import Dict
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import pandas as pd


def config(module_name: str):
    """
    Returns the configuration for any specific module
    :param module_name: the name of the module
    :return: dictionary containing the parameters as keys and parameter values as the values
    """
    config = configparser.ConfigParser()
    config.read("config.ini")

    try:
        return config[module_name]
    except KeyError as err:
        print(f"Non existing module name:\n " f"{config.sections()} not {err}")


def convert_str_to_list(string_list):
    return [int(item.strip()) for item in string_list.strip("][").split(",")]


def convert_str_to_bool(string: str):
    if string.lower() == "true" : return True
    elif string.lower() == "false" : return False
    else:raise Exception("Not a valid boolean")


def evaluate_predictions(logits, labels):
    predicted_labels = np.argmax(logits, 1)
    accuracy = accuracy_score(labels, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(labels, predicted_labels)
    conf_matrix = confusion_matrix(labels, predicted_labels, labels=[0, 1, 2])
    return accuracy, balanced_accuracy, conf_matrix


def train_test_split(dataset, train_division):
    data_len = len(dataset)
    train_pivot = int(data_len * train_division)
    indexes = list(range(data_len))
    return indexes[:train_pivot], indexes[train_pivot:]


def read_source_file(path):
    return pd.read_csv(path)
