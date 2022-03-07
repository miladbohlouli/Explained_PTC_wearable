import pandas as pd
from torch.utils.data import SubsetRandomSampler, DataLoader
from models.mlp_model import *
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataset.mlp_data_loader import mlp_dataset
import numpy as np
import logging
from sklearn.model_selection import KFold
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
from utils import evaluate_predictions
import seaborn as sns
from matplotlib import pyplot as plt

my_parser = argparse.ArgumentParser()
my_parser.add_argument('-ds_dir',
                       default="dataset/raw_data_Liu.csv",
                       type=str)


my_parser.add_argument('-save_dir',
                       help="directory used for saving the models",
                       default="save/",
                       type=str)

my_parser.add_argument('-temp_dir',
                       help="directory used for visualization using the tesnorboard",
                       default="temp/",
                       type=str)

my_parser.add_argument('-prefix',
                       help="prefix for saving and visualization",
                       default="",
                       type=str)

my_parser.add_argument('-verbose',
                       default=False,
                       type=str)

mlp_config = config("mlp")
parser = my_parser.parse_args()
logging.basicConfig(level=logging.DEBUG) if bool(parser.verbose) else None
train_writer = SummaryWriter(os.path.join(parser.temp_dir, "train"))
eval_writer = SummaryWriter(os.path.join(parser.temp_dir, "eval"))


def train():
    # load the dataset
    logging.debug(f"Loading the dataset")

    ds = mlp_dataset(
        parser.ds_dir,
        na_handling_method=mlp_config["na_handling_method"]
    )

    id = np.random.randint(1, ds.num_people + 1)
    logging.info(f"Chosen ID of the households: {id}")
    individual_household = ds[id]

    logging.debug(f"splitting the train and test dataset")
    num_folds = int(mlp_config["k_fold"])
    resampled_evaluated_metrics = dict()
    resampled_evaluated_metrics["train_acc"] = []
    resampled_evaluated_metrics["train_acc_balanced"] = []
    resampled_evaluated_metrics["eval_acc"] = []
    resampled_evaluated_metrics["eval_acc_balanced"] = []

    kfold = KFold(
        n_splits=num_folds,
        shuffle=True
    )

    assert num_folds <= len(list(individual_household))

    for fold, (train_ids, test_ids) in enumerate(kfold.split(individual_household)):
        model = build_mlp_model(
            layers=convert_str_to_list(mlp_config["layers"]),
            activation=mlp_config["activation"],
            batch_norm=bool(mlp_config["batch_norm"])
        )

        loss_model = CrossEntropyLoss()
        optimizer = Adam(model.parameters())
        global_step = 0

        num_epochs = int(mlp_config["num_epochs"])

        logging.info(f"Fold ({fold}/{num_folds})")
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        train_loader = DataLoader(
            dataset=individual_household,
            batch_size=int(mlp_config["batch_size"]),
            sampler=train_subsampler
        )

        test_loader = DataLoader(
            dataset=individual_household,
            batch_size=int(mlp_config["batch_size"]),
            sampler=test_subsampler,
        )

        # training the model
        for i in range(num_epochs):
            model.train()
            train_loss_avr = 0
            eval_loss_avr = 0
            train_acc_avr = 0
            train_balanced_acc_avr = 0
            eval_acc_avr = 0
            eval_balanced_acc_avr = 0
            num_samples = 0
            total_conf_matrix = np.zeros((3, 3))

            for samples, labels in train_loader:
                res = model(samples)
                train_loss = loss_model(res, labels)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                global_step += 1

                train_writer.add_scalar(f"fold: {fold}/loss", train_loss, global_step)

                # Calculate the accuracy
                num_samples += len(samples)
                train_acc, train_balanced_acc, _ = evaluate_predictions(res.detach().numpy(), labels)
                train_loss_avr += train_loss.detach().numpy() * len(samples)
                train_acc_avr += train_acc * len(samples)
                train_balanced_acc_avr += train_balanced_acc * len(samples)

                train_writer.add_scalar(f"fold: {fold}/loss", train_loss, global_step)
                train_writer.add_scalar(f"fold: {fold}/accuracy", train_acc, global_step)
                train_writer.add_scalar(f"fold: {fold}/balanced accuracy", train_balanced_acc, global_step)

            # evaluating the model
            model.eval()
            for samples, labels in test_loader:
                res = model(samples)

                eval_loss = loss_model(res, labels)

                eval_acc, eval_balanced_acc, eval_conf_matrix = evaluate_predictions(res.detach().numpy(), labels)
                eval_loss_avr += eval_loss.detach().numpy() * len(samples)
                eval_acc_avr += eval_acc * len(samples)
                eval_balanced_acc_avr += eval_balanced_acc * len(samples)
                total_conf_matrix += eval_conf_matrix

                eval_writer.add_scalar(f"fold: {fold}/accuracy", eval_acc, global_step)
                eval_writer.add_scalar(f"fold: {fold}/balanced accuracy", eval_balanced_acc, global_step)


            print(f"Fold ({fold+1}/{num_folds})\t"
                  f"Epoch ({i+1} / {num_epochs})\t train_loss: {train_loss_avr / num_samples:.2f}\t"
                  f"train_acc (balanced): {train_balanced_acc_avr / num_samples:.2f}\t\t"
                  f"eval_acc (balanced): {eval_balanced_acc_avr / num_samples:.2f}\t")

            # Logging the cross validated information
            resampled_evaluated_metrics["train_acc"]. append(train_acc_avr / num_samples)
            resampled_evaluated_metrics["train_acc_balanced"].append(train_balanced_acc_avr / num_samples)
            resampled_evaluated_metrics["eval_acc"].append(eval_acc_avr / num_samples)
            resampled_evaluated_metrics["eval_acc_balanced"].append(eval_balanced_acc_avr / num_samples)

            fig1 = sns.heatmap(total_conf_matrix, annot=True, cmap=plt.cm.Blues).get_figure()
            plt.ylabel("True label"), plt.xlabel("Predicted labe;")
            eval_writer.add_figure("evaluation confusion matrix", fig1, i)

    metrics_values = np.asarray([(np.round(np.mean(value), 2), np.round(np.std(value), 2)) for value in resampled_evaluated_metrics.values()])
    metrics_values_pd = pd.DataFrame(metrics_values.T, columns=["train_acc", "train_acc_balanced", "eval_acc", "eval_acc_balanced"],
                        index=["mean", "std"])

    # The heatmap of the resampled results to be shown in tensorboard
    plt.figure(figsize=(15, 5))
    fig2 = sns.heatmap(metrics_values.T,
                    annot=True,
                    cmap=plt.cm.Blues,
                    yticklabels=["mean", "std"],
                    xticklabels=["train_acc", "train_accuracy_balanced", "eval_acc", "eval_acc_balanced"]).get_figure()
    plt.xticks(rotation=0)
    eval_writer.add_figure("The overall evaluation metrics", fig2)


if __name__ == '__main__':
    train()



