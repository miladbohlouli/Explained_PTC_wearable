from Feature_selectors.interface import FeatureSelector
import pandas as pd

class simple_feature_selector(FeatureSelector):
    def __init__(self, include_changing_parameters: bool = False):
        FeatureSelector.__init__(self, include_changing_parameters=include_changing_parameters)

        self.selected_features_list += ["mean.hr_5", "mean.WristT_5", "mean.AnkleT_5",
                                       "mean.PantT_5"]

        self.remove_redundant()

        self.label_title = 'therm_pref'

    def fit(self, dataset):
        pass

    def select_features(self, dataset):
        if self.selected_features_list is []:
            raise Exception("The feature selector needs to be trained at first")

        if type(dataset) is not pd.DataFrame:
            dataset = pd.DataFrame(dataset)
        return dataset[self.selected_features_list], dataset[self.label_title]
