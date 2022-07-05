import pandas as pd

class NORMALIZATION_INTERFACE:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, dataset):
        raise Exception("not implemented")

    def normalize(self, dataset):
        raise Exception("not implemented")

