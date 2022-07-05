import numpy as np

from NAhandlers.interface import NA_HANDLER_INTERFACE
import pandas as pd


class dropping_handler(NA_HANDLER_INTERFACE):
    def fit(self, dataset):
        pass

    def fix(self, dataset):
        """
        Just to correct the missing values (Could also be overridden by another type of correction)
        :param method: The method that will be used, possible values include,
            "drop": Drops the raws containing the missing values in the data
            "average": Replace the available values with average
        :return: The result without any missing values
        """
        if type(dataset) is not pd.DataFrame:
            dataset = pd.DataFrame(dataset)
        return dataset.dropna()

