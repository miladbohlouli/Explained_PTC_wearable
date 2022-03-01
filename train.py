from models.mlp_model import *
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataset.mlp_data_loader import mlp_dataset
import numpy as np
import logging

writer = SummaryWriter()

my_parser = argparse.ArgumentParser()
my_parser.add_argument('-batch_size',
                       default=64,
                       type=int)

my_parser.add_argument('-ds_path',
                       default="dataset/raw_data_Liu.csv",
                       type=str)


my_parser.add_argument('-save_dir',
                       help="directory used for saving the models",
                       default="save/",
                       type=str)

my_parser.add_argument('-temp',
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

def train():
    # load the dataset
    logging.debug(f"Loading the dataset")

    ds = mlp_dataset(parser.ds_path)
    id = np.random.randint(ds.num_people)
    logging.info(f"Chosen ID of the households: {id}")
    individual_household = ds[id]

    # train and test splitting
    logging.debug(f"splitting the train and test dataset")

    # load if existing a saved model otherwise building the model from scratch
    build_mlp_model(
        layers=convert_str_to_list(mlp_config["layers"]),
        activation=mlp_config["activation"]
    )

    # training the model

    # evaluating the model

    # checking to save the model


if __name__ == '__main__':
    train()



