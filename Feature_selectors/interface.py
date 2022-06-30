import pandas as pd

class FeatureSelector:
    def __init__(self):
        self.label_title = None
        self.selected_features_list = None

    def fit(self, dataset):
        """
        The default function that will be used for feature selection (The function may be overriden for other feature
        selection methods)
        :param dataset: The source dataset
        """
        raise Exception("not implemented")

    def select_features(self, dataset):
        """"
        The function for getting selecting the features from a dataset and returning the dataset with selected features
        """
        raise Exception("not implemented")

    def get_feature_size(self):
        if self.selected_features_list is None:
            raise Exception("The model has not been trained yet")
        else:
            return len(self.selected_features_list)