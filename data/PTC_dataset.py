import numpy as np
import torch

from utils import *
from Feature_selectors.interface import *
from Feature_selectors.simple_feature_selector import simple_feature_selector
from NAhandlers.interface import *
from NAhandlers.dropping import dropping_handler
from normalizers.interface import *
from normalizers.guassian_normalizer import guassian_normalizer
from samplers.interface import SAMPLING_INTERFACE
from torch.utils.data import Dataset


class PTC_dataset(Dataset):
    def __init__(self,
                 shuffle: bool = True,
                 feature_selector: FeatureSelector = None,
                 na_handler: NA_HANDLER_INTERFACE = None,
                 normalizer: NORMALIZATION_INTERFACE = None,
                 resampler: SAMPLING_INTERFACE = None,
                 train: bool = True):
        """
        A class to construct the train and test datasets as PyTorch Datasets
        :param path: the path for teh source initial file
        :param dataset: the data given as the input
        :param sampler: the indexes that the data will be chosen, if None no sampling will be done
        :param shuffle: if to shuffle the data
        :param feature_selector: feature selection object used for feature selection
        :param na_handler: missing values' fixer class
        :param normalizer: object for normalization
        """

        # One of the path or data should be not None
        self.dataset = None
        self.labels = None
        self.columns = None
        self.path = None
        self.train = train
        self.feature_selector = feature_selector
        self.na_handler = na_handler
        self.normalizer = normalizer
        self.resampler = resampler
        self.shuffle = shuffle

    def load_dataset(self, path):
        if path is not None:
            self.dataset = PTC_dataset.read_source_file(path)
            self.path = path
        else:
            raise Exception("Path need to be provided for loading the dataset")

    def pre_process_dataset(self, logger):
        assert self.dataset is not None

        logger.info(f"\n{30*'*'}\n"
                    f"Preprocessing the {'train' if self.train else 'test'} dataset with the following config:\n"
                    f"feature selector: {self.feature_selector.__class__.__name__}\n"
                    f"missing value handler: {self.na_handler.__class__.__name__}\n"
                    f"shuffle: {self.shuffle}\n"
                    f"over sampling: {self.resampler.__class__.__name__}\n"
                    f"normalization: {self.normalizer.__class__.__name__}\n"
                    f"{30*'*'}")

        if self.train:
            self.feature_selector.fit(self.dataset)
        logger.info(
            f"Selecting the features (number of the features {len(self.feature_selector.selected_features_list)})...")
        self.dataset, self.labels = self.feature_selector.select_features(self.dataset)
        self.columns = self.feature_selector.selected_features_list

        self.convert_nominal_numerical()
        self.convert_labels_numerical()

        logger.info(f"Handling the missing values...")
        if self.train:
            self.na_handler.fit(self.dataset)
        self.dataset = self.na_handler.fix(self.dataset)

        if self.shuffle:
            sampler = list(range(len(self.dataset)))
            np.random.shuffle(sampler)
            self.dataset = self.dataset.iloc[sampler, :]

        logger.info(f"Resampling the dataset to have a balanced dataset (size of dataset: {len(self.dataset)})...")
        if self.train and self.resampler is not None:
            self.dataset, self.labels = self.resampler.fit_resample(self.dataset, self.labels)
            logger.info(f"Dataset size after resampling : {len(self.dataset)}...")

        logger.info(f"Normalizing the dataset...")
        if self.normalizer is not None:
            if self.train:
                self.normalizer.fit(self.dataset)
            self.dataset = self.normalizer.normalize(self.dataset)

        self.dataset = self.dataset.to_numpy()
        self.labels = self.labels.to_numpy()

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]

    def __len__(self):
        return len(self.dataset)

    def convert_nominal_numerical(self):
        """
        Detect the nominal features in self.labels and self.data and convert them to numerical
        """
        object_columns = [column for column in self.dataset.columns if self.dataset[column].dtype == 'O']
        if len(object_columns) == 0: return
        dummies = []
        for column in [object_columns]:
            dummies.append(pd.get_dummies(self.dataset[column]))
            self.dataset = self.dataset.drop(column, axis='columns')
        self.dataset = pd.concat([self.dataset, *dummies], axis=1)

    def make_data_individual(self, home_id):
        self.dataset = self.dataset[self.dataset["ID"] == home_id]

    def convert_labels_numerical(self):
        self.labels = self.labels.map({key: value for value, key in enumerate(sorted(self.labels.unique()))})

    def get_tensor_data(self):
        assert self.dataset is not None
        return torch.from_numpy(self.dataset).float(), torch.from_numpy(self.labels)

    @staticmethod
    def cal_mean_std(dataset):
        return dataset.mean(0), dataset.std(0)

    @staticmethod
    def read_source_file(path):
        dataset = pd.read_csv(path)
        if dataset is None:
            raise Exception("The provided path is not valid")
        return dataset


if __name__ == '__main__':
    pass



