import numpy as np
import pandas as pd
from dataset.data_helpers import *
from torch.utils.data import TensorDataset
import torch


class mlp_dataset():
    def __init__(self,
                 path,
                 shuffle=True,
                 normlaize=True,
                 na_handling_method="average"):
        """
        :param path: The path to the dataset
        :param shuffle: if the dataset needs to be shuffled
        """

        # Read the initial dataset
        ds = pd.read_csv(path)

        # Some of the features have to be selected
        truncated_dataset = pd.concat([ds.iloc[:, 0:9], ds[
            ["mean.hr_5", "mean.WristT_5", "mean.AnkleT_5", "mean.PantT_5", "mean.act_5"]]], axis=1)
        self.num_people = pd.unique(truncated_dataset.loc[:, "ID"]).__len__()
        self.labels = ds.loc[:, 'therm_pref']
        self.shuffle = shuffle

        # correct the missing values
        truncated_dataset = correct_NA_values(
            truncated_dataset,
            method=na_handling_method
        )

        self.columns = truncated_dataset.columns

        # remap the categorical values
        self.__dict = {key: value for value, key in enumerate(truncated_dataset.loc[:, "Sex"].unique())}
        truncated_dataset.loc[:, 'Sex'] = truncated_dataset.loc[:, "Sex"].map(self.__dict)
        self.labels = self.labels.map({key: value for value, key in enumerate(sorted(self.labels.unique()))})

        print(truncated_dataset.iloc[0, :])

        # convert the type
        self.dataset, self.labels = truncated_dataset.to_numpy().astype(np.float32), self.labels.to_numpy().astype(np.int64)

    def __getitem__(self, idx):
        """
        :param idx:The index of the household
        :return: a torch dataset containing all the samples for each household
        """
        # Check the validity of the entered id and if the id exists
        assert idx in np.unique(self.dataset[:, 0])

        # convert the dataset to tensors
        indexes = np.where(self.dataset[:, 0] == idx)[0]
        np.random.shuffle(indexes) if self.shuffle else None

        personal_data, personal_data_labels = torch.from_numpy(self.dataset[indexes, 4:]), torch.from_numpy(self.labels[indexes])
        return TensorDataset(personal_data, personal_data_labels)

    def __len__(self):
        return len(self.ds)


if __name__ == '__main__':
    mlp_ds = mlp_dataset("raw_data_Liu.csv")
    # print(mlp_ds[1])


