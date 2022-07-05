import os

import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
import torch
from utils import *
from data.PTC_dataset import PTC_dataset
from Feature_selectors.interface import FeatureSelector
from NAhandlers.interface import NAhandler
from Feature_selectors.simple_feature_selector import simple_feature_selector
from NAhandlers.dropping import dropping_handler
from normalizers.guassian_normalizer import guassian_normalizer
from normalizers.interface import Normalizer


class mlp_dataset_individual(PTC_dataset):
    def __init__(self,
                 path = None,
                 shuffle: bool = True,
                 feature_selector: FeatureSelector = None,
                 na_handler: NAhandler = None,
                 normalizer: Normalizer = None,
                 home_id = None,
                 train = True
         ):
        PTC_dataset.__init__(
            self,
            shuffle=shuffle,
            feature_selector=feature_selector,
            na_handler=na_handler,
            normalizer=normalizer,
            train = train
        )
        self.load_dataset(path)
        if home_id is not None:
            self.make_data_individual(home_id)
        self.pre_process_dataset()


if __name__ == '__main__':
    home_ids = get_available_home_ids(path="")
    train_dataset = mlp_dataset_individual(
        path="train.csv",
        shuffle=True,
        feature_selector=simple_feature_selector(),
        na_handler=dropping_handler(),
        normalizer=guassian_normalizer(),
        home_id=1
    )

    print(train_dataset[0])

