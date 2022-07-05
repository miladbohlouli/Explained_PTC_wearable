import pandas as pd


class SAMPLING_INTERFACE:
    def __init__(self):
        self.model = None

    def fit_resample(self, dataset, labels):
        raise Exception("not implemented")