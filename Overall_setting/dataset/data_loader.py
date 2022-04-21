import numpy as np
import pandas as pd
from dataset.data_helpers import *
from torch.utils.data import TensorDataset, Dataset
import torch


class dataset_loader(Dataset):
    ds_path = None
    train_indexes = None
    test_indexes = None
    dataset = None
    labels = None
    mean = None
    std = None
    columns = None

    def __init__(self,
                 path,
                 train: bool = True,
                 train_division: float = 0.7,
                 shuffle: bool = True,
                 normalize: bool = True,
                 na_handling_method: str = "average"):

        """
        A class to construct the train and test datasets as PyTorch Datasets
        :param path: The path for teh source initial file
        :param train: IF we want the train or the test dataset
        :param train_division: The size of the train dataset chosen [0-1]
        :param shuffle: if to shuffle the dataset
        :param normalize: if to normalize the dataset using the standard normalization
        :param na_handling_method: How to handle the missing values
        """
        self.train = train
        if path is not dataset_loader.ds_path:
            ds = pd.read_csv(path)
            dataset_loader.ds_path = path

            # Selecting the useful features
            truncated_dataset = pd.concat([ds.iloc[:, 1:9], ds[
                ["mean.hr_5", "mean.WristT_5", "mean.AnkleT_5", "mean.PantT_5", "mean.act_5"]]], axis=1)

            dataset_loader.labels = ds.loc[:, 'therm_pref']

            train_pivot = int(len(truncated_dataset) * train_division)

            indexes = np.random.permutation(list(range(len(truncated_dataset)))) if shuffle else \
                list(range(len(truncated_dataset)))
            dataset_loader.train_indexes = indexes[:train_pivot]
            dataset_loader.test_indexes = indexes[train_pivot:]

            # correct the missing values
            truncated_dataset = correct_NA_values(
                truncated_dataset,
                method=na_handling_method
            )

            dataset_loader.columns = truncated_dataset.columns

            # remap the categorical values
            sex_dict = {key: value for value, key in enumerate(truncated_dataset.loc[:, "Sex"].unique())}
            truncated_dataset.loc[:, 'Sex'] = truncated_dataset.loc[:, "Sex"].map(sex_dict)
            dataset_loader.labels = dataset_loader.labels.map({key: value for value, key in
                                                               enumerate(sorted(dataset_loader.labels.unique()))})

            # convert the type
            dataset_loader.dataset, dataset_loader.labels = truncated_dataset.to_numpy().astype(np.float32), \
                                                  dataset_loader.labels.to_numpy().astype(np.int64)

            # Normalize the dataset, note that the normalization is done by nomalising using the train section's
            #   mean and std
            if normalize:
                dataset_loader.mean, dataset_loader.std = dataset_loader.dataset[dataset_loader.train_indexes].mean(0) \
                    , dataset_loader.dataset[dataset_loader.train_indexes].std(0)

                dataset_loader.dataset = (dataset_loader.dataset - dataset_loader.mean) / dataset_loader.std

    def __getitem__(self, idx):
        if self.train:
            return dataset_loader.dataset[dataset_loader.train_indexes[idx]], dataset_loader.labels[dataset_loader.train_indexes[idx]]

        elif not self.train:
            return dataset_loader.dataset[dataset_loader.test_indexes[idx]], dataset_loader.labels[dataset_loader.test_indexes[idx]]

        else:
            raise Exception("The entered value for train is not correct")

    def __len__(self):
        if self.train:
            return len(dataset_loader.train_indexes)

        elif not self.train:
            return len(dataset_loader.test_indexes)

        else:
            raise Exception("The entered value for train is not correct")


if __name__ == '__main__':
    train_ds = dataset_loader("raw_data_Liu.csv",
                            train=True
                              )

    test_ds = dataset_loader("raw_data_Liu.csv",
                            train=False
                              )