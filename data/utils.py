import pandas as pd


class NAhandler:
    def __init__(self, method: str = 'average'):
        self.mean = None
        self.std = None

        if method == 'average' or method == "drop":
            self.method = method
        else:
            raise Exception("The entered format for the method is not valid")

    def fit(self, dataset):
        self.mean, self.std = dataset.mean(), dataset.std()

    def fix(self, dataset):
        """
        Just to correct the missing values (Could also be overridden by another type of correction)
        :param method: The method that will be used, possible values include,
            "drop": Drops the raws containing the missing values in the data
            "average": Replace the available values with average
        :return: The result without any missing values
        """
        if self.mean is None or self.std is None:
            raise Exception("The NAhandler needs to be trained at first")

        if type(dataset) is not pd.DataFrame:
            dataset = pd.DataFrame(dataset)

        if self.method == "drop":
            return dataset.dropna()

        elif self.method == "average":
            return dataset.fillna(self.mean)


class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, dataset):
        self.mean = dataset.mean(0)
        self.std = dataset.std(0)

    def normalize(self, dataset):
        if self.mean is None or self.std is None:
            raise Exception("The Normalizer needs to be trained at first")

        return (dataset - self.mean) / self.std


class FeatureSelector:
    def __init__(self):
        self.selected_features_list = None

    def fit(self, dataset, labels):
        if type(dataset) is not pd.DataFrame:
            dataset = pd.DataFrame(dataset)

        self.selected_features_list = dataset.columns[1:9] + ["mean.hr_5", "mean.WristT_5", "mean.AnkleT_5",
                                                              "mean.PantT_5", "mean.act_5"]

    def select_features(self, dataset):
        if self.selected_features_list is None:
            raise Exception("The feature selector needs to be trained at first")

        if type(dataset) is not pd.DataFrame:
            dataset = pd.DataFrame(dataset)

        return dataset[:, self.selected_features_list]
