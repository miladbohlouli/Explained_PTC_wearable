import pandas as pd
from NAhandlers.interface import NAhandler


class averaging_na_handler(NAhandler):
    def fit(self, dataset):
        self.mean, self.std = dataset.mean(), dataset.std()

    def fix(self, dataset):
        if self.mean is None or self.std is None:
            raise Exception("The NAhandler needs to be trained at first")

        return dataset.fillna(self.mean)
