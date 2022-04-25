from torch.utils.data import DataLoader
from experiments.overall_setting.mlp.mlp_model import *
from torch.utils.tensorboard import SummaryWriter
import argparse
from experiments.overall_setting.mlp.data_loader import dataset_loader
import numpy as np
import logging
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
from utils import *
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
from PTC_classifier import Classifier
warnings.simplefilter('ignore', UserWarning)


class mlp_classifier(Classifier):
    def __init__(self, model, data_loader):
        pass


def train(model=None,
          ds_dir="../../../data/PTC.csv",
          saving_dir="save/",
          temp_dir="temp/",
          prefix="",
          verbose=False,
          num_epochs=None
    ):

    mlp_config = config("MLP")
    training_config = config("TRAINING")
    logging.basicConfig(level=logging.DEBUG) if bool(verbose) else None

    # load the data
    logging.debug(f"Loading the data")

    train_writer = SummaryWriter(os.path.join(temp_dir, prefix + "train"))
    eval_writer = SummaryWriter(os.path.join(temp_dir, prefix + "eval"))

    train_dataset = dataset_loader(
        ds_dir,
        train=True,
        na_handling_method=mlp_config["na_handling_method"]
    )

    test_dataset = dataset_loader(
        ds_dir,
        train=False,
        na_handling_method=mlp_config["na_handling_method"]
    )

    logging.debug(f"splitting the train and test data")

    if model is None:
        model = build_mlp_model(
            layers=convert_str_to_list(mlp_config["layers"]),
            activation=mlp_config["activation"],
            batch_norm=convert_str_to_bool(mlp_config["batch_norm"])
        )

    loss_model = CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    if num_epochs is None:
        num_epochs = int(training_config["num_epochs"])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=int(training_config["batch_size"]),
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=int(training_config["batch_size"]),
    )

    # training the model

    global_step = 0
    for i in range(num_epochs):
        model.train()
        train_num_samples = 0
        train_loss_avr = 0
        train_acc_avr = 0
        train_balanced_acc_avr = 0

        for samples, labels in train_loader:
            res = model(samples)
            train_loss = loss_model(res, labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Calculate the accuracy
            train_num_samples += len(samples)
            train_acc, train_balanced_acc, _ = evaluate_predictions(res.detach().numpy(), labels)
            train_loss_avr += train_loss.detach().numpy() * len(samples)
            train_acc_avr += train_acc * len(samples)
            train_balanced_acc_avr += train_balanced_acc * len(samples)

            # Log the evaluation metrics
            train_writer.add_scalar(f"loss", train_loss, global_step)
            train_writer.add_scalar(f"accuracy", train_acc_avr / train_num_samples, global_step)
            train_writer.add_scalar(f"balanced accuracy", train_balanced_acc_avr / train_num_samples, global_step)

            global_step += 1

        # evaluating the model
        eval_loss_avr = 0
        eval_acc_avr = 0
        eval_balanced_acc_avr = 0
        total_conf_matrix = np.zeros([3, 3])

        model.eval()
        test_num_samples = 0
        for samples, labels in test_loader:
            res = model(samples)

            eval_loss = loss_model(res, labels)

            test_num_samples += len(samples)
            eval_acc, eval_balanced_acc, eval_conf_matrix = evaluate_predictions(res.detach().numpy(), labels)
            eval_loss_avr += eval_loss.detach().numpy() * len(samples)
            eval_acc_avr += eval_acc * len(samples)
            eval_balanced_acc_avr += eval_balanced_acc * len(samples)
            total_conf_matrix += eval_conf_matrix

        eval_writer.add_scalar(f"accuracy", eval_acc_avr / test_num_samples, global_step)
        eval_writer.add_scalar(f"balanced accuracy", eval_balanced_acc_avr / test_num_samples, global_step)

        print(f"Epoch ({i+1:3} / {num_epochs})\t train_loss: {train_loss_avr / train_num_samples:.2f}\t"
              f"train_acc (balanced): {train_balanced_acc_avr / train_num_samples:.2f}\t\t"
              f"eval_acc (balanced): {eval_balanced_acc_avr / test_num_samples:.2f}\t")

        fig1 = sns.heatmap(total_conf_matrix, annot=True, cmap=plt.cm.Blues).get_figure()
        plt.ylabel("True label"), plt.xlabel("Predicted labe;")
        eval_writer.add_figure("evaluation confusion matrix", fig1, i)


    # metrics_values = np.asarray([train_acc, train_balanced_acc, eval_acc, eval_balanced_acc])
    # print(metrics_values)
    # metrics_values_pd = pd.DataFrame(metrics_values.T.ravel(), index=["train_acc", "train_acc_balanced", "eval_acc", "eval_acc_balanced"])
    # print(metrics_values_pd)


    # The heatmap of the resampled results to be shown in tensorboard
    # plt.figure(figsize=(15, 5))
    # fig2 = sns.heatmap(metrics_values.T.ravel(),
    #                    annot=True,
    #                    cmap=plt.cm.Blues,
    #                    yticklabels=["values"],
    #                    xticklabels=["train_acc", "train_accuracy_balanced", "eval_acc", "eval_acc_balanced"]).get_figure()
    # plt.xticks(rotation=0)
    # eval_writer.add_figure("The overall evaluation metrics", fig2)


if __name__ == '__main__':
    train()



