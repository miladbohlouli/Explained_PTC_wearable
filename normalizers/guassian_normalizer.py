import pandas as pd
from normalizers.interface import Normalizer

class guassian_normalizer(Normalizer):
    def fit(self, dataset):
        self.mean = dataset.mean(0)
        self.std = dataset.std(0)

    def normalize(self, dataset):
        if self.mean is None or self.std is None:
            raise Exception("The Normalizer needs to be trained at first")

        return (dataset - self.mean) / self.std

