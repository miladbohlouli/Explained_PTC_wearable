from torch.utils.data import DataLoader
from models.mlp import *
from torch.utils.tensorboard import SummaryWriter
from Feature_selectors.simple_feature_selector import simple_feature_selector
from NAhandlers.averaging import averaging_na_handler
from normalizers.guassian_normalizer import guassian_normalizer
from data.mlp_data_loader import mlp_dataset_individual
import numpy as np
import logging
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
from utils import evaluate_predictions
from utils import train_test_split
import warnings
warnings.filterwarnings("ignore")
from ray import tune
from ray.tune import CLIReporter
from functools import partial
from trainers.mlp_individual_trainer import train
import pickle as pk

if __name__ == '__main__':

    config = dict()
    # model parameters
    config["layers"] = tune.grid_search([[128, 3], [256, 3], [128, 128, 3], [128, 256, 3], [256, 256, 3], [128, 128, 128, 3]])
    # config["layers"] = [128, 128, 3]
    config["activation"] = "Relu"
    config["dropout"] = 0.2

    # Data preprocessing parameters
    config["feature_selector"] = "threshold_feature_selector"
    config["na_handler"] = "adaptive"
    config["normalizer"] = "guassian_normalizer"
    config["resampler"] = "None"

    # Training parameters
    config["batch_size"] = 16
    config["num_epochs"] = 30
    config["weight_decay"] = 1e-4
    config["lr"] = 1e-4
    config["validation_not_improved"] = 10
    config["threshold"] = 0.05

    data_dir = os.path.abspath("../../../../data")
    logging_dir = os.path.abspath("")

    feature_selector = simple_feature_selector()
    na_handler = averaging_na_handler()
    normalizer = guassian_normalizer()

    reporter = CLIReporter(
        metric_columns=["accuracy", "training_iteration"]
    )
    func = partial(
        train,
        data_dir=data_dir,
        logging_dir=logging_dir,
        parameter_tuning=True
        )

    result = tune.run(
        func,
        resources_per_trial={"cpu": 8},
        config=config,
        metric="accuracy",
        mode="max",
    )

    best_result = result.get_best_trial("accuracy", "max", "last")

    print(f"accuracy: {best_result.last_result['accuracy']}")

    print(best_result.config)
    with open(logging_dir + "/best_config.pk", "wb") as file:
        pk.dump(best_result.config, file)


