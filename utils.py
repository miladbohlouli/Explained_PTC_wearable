import os.path
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from tqdm import tqdm
import logging
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import cohen_kappa_score
from Feature_selectors.simple_feature_selector import simple_feature_selector
from Feature_selectors.threshold_feature_selector import threshold_feature_selector
from Feature_selectors.all_features_selector import all_feature_selector
from NAhandlers.averaging import averaging_na_handler
from NAhandlers.dropping import dropping_handler
from NAhandlers.adaptive import  adaptive
from normalizers.guassian_normalizer import guassian_normalizer
from samplers.SMOTE import SMOTE_SAMPLER
from samplers.ADASYN import ADASYN_SAMPLER


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


def evaluate_predictions(logits, labels, train: bool = True):
    probs = softmax(logits, axis=1)
    kohen_kappa = None
    binary_labels = np.zeros((len(labels), 3))
    binary_labels[:, labels] = 1
    predicted_labels = np.argmax(logits, 1)
    accuracy = accuracy_score(labels, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(labels, predicted_labels)
    if not train:
        kohen_kappa = cohen_kappa_score(labels, predicted_labels)
    return accuracy, balanced_accuracy, kohen_kappa


def read_source_file(path):
    return pd.read_csv(path)


if __name__ == '__main__':
    train_test_split("data", train_division=0.8)


def evaluate_model(loss_model, model, test_ds):
    ds, labels = test_ds.get_tensor_data()
    res = model(ds)
    eval_loss = loss_model(res, labels)
    eval_acc, eval_balanced_acc, kohen_kappa = evaluate_predictions(res.detach().numpy(), labels, train=False)
    return eval_acc, eval_balanced_acc, eval_loss, kohen_kappa


def train_step(
        labels,
        loss_model,
        model,
        optimizer,
        samples
):
    res = model(samples.float())
    train_loss = loss_model(res, labels)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Calculate the accuracy
    train_acc, train_balanced_acc, _ = evaluate_predictions(res.detach().numpy(), labels, train=True)
    return train_acc, train_balanced_acc, train_loss.detach().numpy()


def get_preprocessing_tools(config):
    fs = None
    na = None
    nor = None
    res = None

    if config["feature_selector"] == "threshold_feature_selector":
        fs = threshold_feature_selector(threshold=config["threshold"])
    elif config["feature_selector"] == "simple_feature_selector":
        fs = simple_feature_selector()
    elif config["feature_selector"] == "all_feature_selector":
        fs = all_feature_selector()

    if config["na_handler"] == "averaging":
        na = averaging_na_handler()
    elif config["na_handler"] == "dropping":
        na = dropping_handler()
    elif config["na_handler"] == "adaptive":
        na = adaptive()

    if config["normalizer"] == "guassian_normalizer":
        nor = guassian_normalizer()

    if config["resampler"] == "SMOTE":
        res = SMOTE_SAMPLER()
    elif config["resampler"] == "ADASYN":
        res = ADASYN_SAMPLER()
    return fs, na, nor, res