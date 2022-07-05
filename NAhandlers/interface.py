import pandas as pd


class NA_HANDLER_INTERFACE:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, dataset):
        raise Exception("not implemented")

    def fix(self, dataset):
        """
        Just to correct the missing values (Could also be overridden by another type of correction)
        :param method: The method that will be used, possible values include,
            "drop": Drops the raws containing the missing values in the data
            "average": Replace the available values with average
        :return: The result without any missing values
        """
        raise Exception("not implemented")