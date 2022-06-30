from torch.utils.data import SubsetRandomSampler, DataLoader
from mlp import *
from torch.utils.tensorboard import SummaryWriter
from Feature_selectors.naive_feature_selector import simple_feature_selector
from NAhandlers.dropping_method import dropping_handler
from NAhandlers.averaging import averaging_na_handler
from normalizers.guassian_normalizer import guassian_normalizer
from data_loader import mlp_dataset_individual
import numpy as np
import logging
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
from utils import evaluate_predictions
import seaborn as sns
from matplotlib import pyplot as plt
from utils import train_test_split
import warnings
warnings.filterwarnings("ignore")

def train(
        config=None,
        checkpoint_dir=None,
        data_dir=None,
        logging_dir=None
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

    home_ids = get_available_home_ids(path="../../../data")
    train_dataset_dir = os.path.join(data_dir, "train.csv")
    test_dataset_dir = os.path.join(data_dir, "test.csv")
    house_eval_accuracies = dict()
    for home_id in home_ids:

        train_ds = mlp_dataset_individual(
            path=train_dataset_dir,
            shuffle=True,
            feature_selector=simple_feature_selector(),
            na_handler=averaging_na_handler(),
            normalizer=guassian_normalizer(),
            home_id=home_id,
            train=True
        )

        test_ds = mlp_dataset_individual(
                path=test_dataset_dir,
                shuffle=True,
                feature_selector=train_ds.feature_selector,
                na_handler=train_ds.na_handler,
                normalizer=train_ds.normalizer,
                home_id=home_id,
                train=False
            )

        feature_size = train_ds.feature_selector.get_feature_size()
        model = build_mlp_model(
            [feature_size] + config["layers"]
        ).float()

        logger.info(model)

        loss_model = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        global_step = 0

        train_loader = DataLoader(
            dataset=train_ds,
            batch_size=config["batch_size"],
        )

        test_loader = DataLoader(
            dataset=test_ds,
            batch_size=config["batch_size"],
        )

        logger.info(f"Training the model for home {home_id} with datasize {len(train_ds)}")

        num_epochs = int(config["num_epochs"])
        num_train_samples = len(train_ds)
        num_test_samples = len(test_ds)
        best_eval_accuracy = 0

        # training the model
        for epoch in range(num_epochs):
            model.train()
            train_loss_avr = 0
            eval_loss_avr = 0
            train_acc_avr = 0
            train_balanced_acc_avr = 0
            eval_acc_avr = 0
            eval_balanced_acc_avr = 0
            total_conf_matrix = np.zeros((3, 3))

            for samples, labels in train_loader:
                res = model(samples.float())
                train_loss = loss_model(res, labels)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                global_step += 1

                # Calculate the accuracy
                train_acc, train_balanced_acc, _ = evaluate_predictions(res.detach().numpy(), labels)
                train_loss_avr += train_loss.detach().numpy() * len(samples)
                train_acc_avr += train_acc * len(samples)
                train_balanced_acc_avr += train_balanced_acc * len(samples)

                train_writer.add_scalar(f"home_id: {home_id}/loss", train_loss, global_step)
                train_writer.add_scalar(f"home_id: {home_id}/accuracy", train_acc, global_step)
                train_writer.add_scalar(f"home_id: {home_id}/balanced accuracy", train_balanced_acc, global_step)

            # evaluating the model
            model.eval()
            for samples, labels in test_loader:
                res = model(samples.float())

                eval_loss = loss_model(res, labels)

                eval_acc, eval_balanced_acc, eval_conf_matrix = evaluate_predictions(res.detach().numpy(), labels)
                eval_loss_avr += eval_loss.detach().numpy() * len(samples)
                eval_acc_avr += eval_acc * len(samples)
                eval_balanced_acc_avr += eval_balanced_acc * len(samples)
                total_conf_matrix += eval_conf_matrix

                eval_writer.add_scalar(f"home_id: {home_id}/accuracy", eval_acc, global_step)
                eval_writer.add_scalar(f"home_id: {home_id}/balanced accuracy", eval_balanced_acc, global_step)

            logger.info(f"home_id ({home_id})\t"
                  f"Epoch ({epoch+1:3} / {num_epochs})\t train_loss: {train_loss_avr / num_train_samples:.2f}\t"
                  f"train_acc (balanced): {train_balanced_acc_avr / num_train_samples:.2f}\t\t"
                  f"eval_acc (balanced): {eval_balanced_acc_avr / num_test_samples:.2f}\t")

            if eval_balanced_acc_avr / num_test_samples > best_eval_accuracy:
                best_eval_accuracy = eval_balanced_acc_avr / num_test_samples
        house_eval_accuracies[home_id]=best_eval_accuracy

    logger.info(house_eval_accuracies)
    logger.info(f"The average overall accuracy for all the houses: {np.average(list(house_eval_accuracies.values()))}")




if __name__ == '__main__':

    config = dict()
    # model parameters
    config["layers"] = [256, 256, 3]
    config["activation"] = "Relu"

    # Training parameters
    config["batch_size"] = 16
    config["num_epochs"] = 100
    config["weight_decay"] = 1e-5
    config["lr"] = 1e-4

    data_dir = os.path.abspath("../../../data")
    logging_dir = os.path.abspath(".")

    # save
    train(
        config=config,
        data_dir=data_dir,
        logging_dir=logging_dir
    )



