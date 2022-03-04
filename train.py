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
writer = SummaryWriter(parser.temp_dir)


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
    kfold = KFold(
        n_splits=num_folds,
        shuffle=True
    )

    assert num_folds <= len(list(individual_household))

    model = build_mlp_model(
        layers=convert_str_to_list(mlp_config["layers"]),
        activation=mlp_config["activation"]
    )

    loss_model = CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    global_step = 0
    num_epochs = int(mlp_config["num_epochs"])

    for fold, (train_ids, test_ids) in enumerate(kfold.split(individual_household)):
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
            sampler=test_subsampler
        )

        # training the model
        for i in range(num_epochs):
            model.train()
            train_loss_list = []
            eval_loss_list = []
            for samples, labels in train_loader:
                res = model(samples)

                train_loss = loss_model(res, labels)
                train_loss_list.append(train_loss.detach().numpy())

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                global_step += 1

                writer.add_scalar("Loss/train_loss", train_loss, global_step)

            # evaluating the model
            model.eval()
            for samples, labels in test_loader:
                res = model(samples)

                eval_loss = loss_model(res, labels)
                eval_loss_list.append(eval_loss.detach().numpy())

                writer.add_scalar("Loss/eval_loss", eval_loss, global_step)

            print(f"Epoch ({i} / {num_epochs})\t train_loss: {np.sum(train_loss_list) / len(train_loss_list):.2f}\t"
                  f" eval_loss: {np.sum(eval_loss_list) / len(eval_loss_list):.2f}")

if __name__ == '__main__':
    train()
    # mnist = MNIST("temp/",
    #       download=True,
    #       train=True)
    #
    # print(next(iter(mnist)))


