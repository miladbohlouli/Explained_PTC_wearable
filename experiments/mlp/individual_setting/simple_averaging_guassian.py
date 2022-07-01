from torch.utils.data import DataLoader
from models.mlp import *
from torch.utils.tensorboard import SummaryWriter
from Feature_selectors.naive_feature_selector import simple_feature_selector
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
from mlp_individual_trainer import train

if __name__ == '__main__':

    config = dict()
    # model parameters
    config["layers"] = tune.grid_search([[128, 3], [256, 3], [128, 128, 3], [128, 256, 3], [256, 256, 3], [128, 128, 128, 3]])
    # config["layers"] = [128, 3]
    config["activation"] = "Relu"

    # Training parameters
    config["batch_size"] = tune.grid_search([2, 8])
    config["num_epochs"] = 150
    config["weight_decay"] = tune.grid_search([1e-5, 1e-6])
    config["lr"] = tune.grid_search([1e-3, 1e-4, 1e-5])
    config["validation_not_improved"] = 30

    data_dir = os.path.abspath("../../../data")
    logging_dir = os.path.abspath("")

    feature_selector = simple_feature_selector()
    na_handler = averaging_na_handler()
    normalizer = guassian_normalizer()

    reporter = CLIReporter(
        metric_columns=["accuracy", "training_iteration"]
    )
    func = partial(train,
                   data_dir=data_dir,
                   logging_dir=logging_dir,
                   feature_selector=feature_selector,
                   na_handler=na_handler,
                   normalizer=normalizer)
    result = tune.run(
        func,
        resources_per_trial={"cpu": 8},
        config=config,
        metric="accuracy",
        mode="max",
        progress_reporter=reporter,
        log_to_file=True
    )

    best_result = result.get_best_trial("accuracy", "max", "last")

    print(best_result.config)
    with open(logging_dir + "/best_config.txt", "wb") as file:
        file.write(best_result.config)

    print(f"accuracy: {best_result.last_result['accuracy']}")
