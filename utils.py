import configparser
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

def config(module_name: str) -> Dict[str, str]:
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


def evaluate_predictions(logits, labels):
    predicted_labels = np.argmax(logits, 1)
    accuracy = accuracy_score(labels, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(labels, predicted_labels)
    conf_matrix = confusion_matrix(labels, predicted_labels)
    return accuracy, balanced_accuracy, conf_matrix


