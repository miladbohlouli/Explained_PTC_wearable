import pandas as pd


class FeatureSelector:
    def __init__(self, include_changing_parameters: bool = False):
        self.label_title = "therm_pref"
        self.selected_features_list = []
        self.include_changing_parameters = include_changing_parameters

        # Add the unchanging parameters for houses
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
        if self.selected_features_list is []:
            raise Exception("The feature selector needs to be trained at first")

        if type(dataset) is not pd.DataFrame:
            dataset = pd.DataFrame(dataset)
        return dataset[self.selected_features_list], dataset[self.label_title]

    def get_feature_size(self):
        if self.selected_features_list is []:
            raise Exception("The model has not been trained yet")
        else:
            return len(self.selected_features_list)

    def add_unchanging_parameters(self):
        self.selected_features_list += ['Sex', 'Age', 'Height', 'Weight', 'ColdSens', 'ColdExp', 'Workhr',
                                       'Coffeeintake', 'location']

    def __remove_unchanging(self):
        ls = ['Sex', 'Age', 'Height', 'Weight', 'ColdSens', 'ColdExp', 'Workhr',
                                       'Coffeeintake', 'location']
        for item in ls:
            if item in self.selected_features_list:
                self.selected_features_list.remove(item)

    def __remove_uniques(self):
        for item in ["ID", "Vote_time"]:
            if item in self.selected_features_list:
                self.selected_features_list.remove(item)

    def remove_redundant(self):
        self.__remove_uniques()
        self.__remove_unchanging()
        if self.label_title in self.selected_features_list:
            self.selected_features_list.remove(self.label_title)
