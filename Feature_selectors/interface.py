import pandas as pd

class FeatureSelector:
    def __init__(self, include_changing_parameters: bool = False):
        self.label_title = None
        self.selected_features_list = []
        self.include_changing_parameters = include_changing_parameters


        if self.include_changing_parameters:
            print("Adding the unchangings")
            self.add_unchanging_parameters()


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
        if self.selected_features_list is []:
            raise Exception("The model has not been trained yet")
        else:
            return len(self.selected_features_list)

    def add_unchanging_parameters(self):
        self.selected_features_list += ['Sex', 'Age', 'Height', 'Weight', 'ColdSens', 'ColdExp', 'Workhr',
                                       'Coffeeintake', 'location']