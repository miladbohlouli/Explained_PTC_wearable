from torch.utils.data import DataLoader
from models.mlp import *
from torch.utils.tensorboard import SummaryWriter
from Feature_selectors.threshold_feature_selector import threshold_feature_selector
from Feature_selectors.simple_feature_selector import simple_feature_selector
from NAhandlers.adaptive import adaptive
from normalizers.guassian_normalizer import guassian_normalizer
from data.mlp_data_loader import mlp_dataset_individual
import numpy as np
import logging
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
from utils import evaluate_model, train_test_split, train_step
import warnings
warnings.filterwarnings("ignore")
from ray import tune


def train(
        config = None,
        checkpoint_dir = None,
        data_dir = None,
        logging_dir = None,
        parameter_tuning = False,
    ):

    # Setting the logging and the other tools for saving
    train_writer = SummaryWriter(os.path.join(os.path.join(logging_dir, "tensorboard"), "train"))
    eval_writer = SummaryWriter(os.path.join(os.path.join(logging_dir, "tensorboard"), "eval"))

    logger = logging.getLogger("runtime_logs")
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    hdlr1 = logging.FileHandler(os.path.join(logging_dir, "runtime.log"))
    hdlr1.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(hdlr1)

    # load the data
    logging.debug(f"Loading the data")

    if "train.csv" not in os.listdir():
        train_test_split(data_dir)

    home_ids = get_available_home_ids(path=data_dir)
    train_dataset_dir = os.path.join(data_dir, "train.csv")
    test_dataset_dir = os.path.join(data_dir, "test.csv")
    house_eval_accuracies = dict()

    for home_id in home_ids:
        house_eval_accuracies[home_id] = []
        feature_selector, na_handler, normalizer, resampler = get_preprocessing_tools(config)
        logger.info("Processing the train dataset...")
        train_ds = mlp_dataset_individual(
            path=train_dataset_dir,
            shuffle=True,
            feature_selector=feature_selector,
            na_handler=na_handler,
            normalizer=normalizer,
            resampler=resampler,
            home_id=home_id,
            train=True,
            logger=logger
        )

        logger.info("Processing the test dataset...")
        test_ds = mlp_dataset_individual(
                path=test_dataset_dir,
                shuffle=True,
                feature_selector=train_ds.feature_selector,
                na_handler=train_ds.na_handler,
                normalizer=train_ds.normalizer,
                resampler=train_ds.resampler,
                home_id=home_id,
                train=False,
                logger=logger
            )

        feature_size = train_ds.feature_selector.get_feature_size()
        model = build_mlp_model(
            layers=[feature_size] + config["layers"],
            activation=config["activation"],
            dropout=config["dropout"]
        ).float()

        logger.info(model)

        loss_model = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        global_step = 0

        train_loader = DataLoader(
            dataset=train_ds,
            batch_size=config["batch_size"],
        )

        logger.info(f"Training the model for home {home_id} with datasize {len(train_ds)}")

        num_epochs = int(config["num_epochs"])
        num_train_samples = len(train_ds)
        best_eval_accuracy = 0
        validation_accuracy_not_improved_epochs = 0

        # training the model
        for epoch in range(num_epochs):
            model.train()
            train_loss_avr = 0
            train_acc_avr = 0
            train_balanced_acc_avr = 0

            for samples, labels in train_loader:
                train_results = train_step(
                    samples=samples,
                    labels=labels,
                    model=model,
                    loss_model=loss_model,
                    optimizer=optimizer
                )

                train_acc, train_balanced_acc, train_loss = train_results
                num_samples = len(samples)

                train_loss_avr += train_loss
                train_acc_avr += train_acc * num_samples
                train_balanced_acc_avr += train_balanced_acc * num_samples

                train_writer.add_scalar(f"home_id: {home_id}/loss", train_loss, global_step)
                train_writer.add_scalar(f"home_id: {home_id}/accuracy", train_acc, global_step)
                train_writer.add_scalar(f"home_id: {home_id}/balanced accuracy", train_balanced_acc, global_step)

                global_step += 1

            # evaluating the model
            model.eval()
            eval_acc, eval_balanced_acc, eval_loss, kohen_kappa = evaluate_model(
                loss_model=loss_model,
                model=model,
                test_ds=test_ds
            )

            eval_writer.add_scalar(f"home_id: {home_id}/loss", eval_loss, global_step)
            eval_writer.add_scalar(f"home_id: {home_id}/accuracy", eval_acc, global_step)
            eval_writer.add_scalar(f"home_id: {home_id}/balanced accuracy", eval_balanced_acc, global_step)
            eval_writer.add_scalar(f"home_id: {home_id}/kohen_kappa", kohen_kappa, global_step)

            logger.info(f"home_id ({home_id})\t"
                        f"Epoch ({epoch+1:3} / {num_epochs})\t\t "
                        f"train_loss: {train_loss_avr / num_train_samples:.2f}\t"
                        f"eval_loss: {eval_loss:.2f}\t\t"
                        f"train_acc: {train_acc_avr / num_train_samples:.2f}\t\t"
                        f"eval_acc: {eval_acc:.2f}\t\t"
                        f"kohen_kappa: {kohen_kappa:.2f}\t")

            if eval_acc > best_eval_accuracy:
                best_eval_accuracy = eval_acc
                validation_accuracy_not_improved_epochs = 0

            else:
                validation_accuracy_not_improved_epochs += 1

            if validation_accuracy_not_improved_epochs >= config["validation_not_improved"]:
                logger.info("Early stopping the training due to no change in validation accuracies")
                break

        house_eval_accuracies[home_id] = eval_acc

    if parameter_tuning:
        tune.report(accuracy=np.average(list(house_eval_accuracies.values())))
    logger.info(house_eval_accuracies)
    logger.info(f"The average overall accuracy for all the houses: {np.average(list(house_eval_accuracies.values()))}")


if __name__ == '__main__':

    config = dict()
    # model parameters
    config["layers"] = [128, 128, 3]
    config["activation"] = "Relu"
    config["dropout"] = 0.5

    # Data preprocessing parameters
    config["feature_selector"] = "threshold_feature_selector"
    config["na_handler"] = "adaptive"
    config["normalizer"] = "guassian_normalizer"
    config["resampler"] = "None"

    # Training parameters
    config["batch_size"] = 2
    config["num_epochs"] = 30
    config["weight_decay"] = 1e-4
    config["lr"] = 1e-4
    config["validation_not_improved"] = 10
    config["threshold"] = 0.05

    data_dir = os.path.abspath("../data")
    logging_dir = os.path.abspath("")

    train(
        config=config,
        logging_dir=logging_dir,
        data_dir=data_dir
    )

