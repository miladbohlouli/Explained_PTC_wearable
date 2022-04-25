import numpy as np
from data.utils import *


class PTC_dataset:
    def __init__(self,
                 path: str = None,
                 dataset=None,
                 sampler: list = None,
                 shuffle: bool = True,
                 feature_selector: FeatureSelector = None,
                 na_handler: NAhandler = None,
                 normalizer: Normalizer = None):
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
        assert (path is not None) or (dataset is not None)
        self.dataset = None
        self.labels = None
        self.columns = None
        self.path = None

        if path is not None:
            ds = PTC_dataset.read_source_file(path)
            self.path = path

        elif dataset is not None:
            if type(dataset) is not pd.DataFrame:
                ds = pd.DataFrame(dataset)
            else:
                ds = dataset

        # subsampling the original data according to the provided sampler
        sampler = sampler if sampler is not None else list(range(len(ds)))
        if shuffle:
            np.random.shuffle(sampler)
        self.dataset = ds.iloc[sampler, :]
        self.labels = ds.iloc[sampler, :].loc[:, "therm_pref"]

        # Feature selections
        self.dataset = feature_selector.select_features(self.dataset)
        self.columns = self.dataset.columns

        # remap the categorical values
        self.convert_nominal_numerical()

        # correct the missing values
        self.dataset = na_handler.fix(self.dataset)

        # normalize the data using the normalizer class
        if normalizer is not None:
            self.dataset = normalizer.normalize(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]

    def __len__(self):
        return len(self.dataset)

    def get_data(self):
        """
        For getting the whole data as whole and not batch-based
        :return: The whole data as numpy arrays
        """
        return self.dataset, self.labels

    def convert_nominal_numerical(self):
        # Todo: Make this part more smart (to detect the nominal features automatically)
        """
        Detect the nominal features in self.labels and self.data and convert them to numerical
        """
        sex_dict = {key: value for value, key in enumerate(sorted(self.dataset.loc[:, "Sex"].unique()))}
        self.dataset.loc[:, 'Sex'] = self.dataset.loc[:, "Sex"].map(sex_dict)
        self.labels = self.labels.map({key: value for value, key in enumerate(sorted(self.labels.unique()))})

        # Convert the datatype of the features
        self.dataset, self.labels = self.dataset.to_numpy().astype(np.float32), self.labels.to_numpy().astype(np.int64)

    @staticmethod
    def cal_mean_std(dataset):
        return dataset.mean(0), dataset.std(0)

    @staticmethod
    def read_source_file(path):
        dataset = pd.read_csv(path)
        if dataset is None:
            raise Exception("The provided path is not valid")
        return dataset





