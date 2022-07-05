from Feature_selectors.interface import FeatureSelector
import pandas as pd


class threshold_feature_selector(FeatureSelector):
    def __init__(self, include_changing_parameters: bool = False, threshold: float = 0.2):
        FeatureSelector.__init__(self, include_changing_parameters=include_changing_parameters)

        self.label_title = 'therm_pref'

        self.threshold = threshold

    def fit(self, dataset):
        dataset_columns = (dataset.isnull().sum() < len(dataset) * self.threshold)
        self.selected_features_list += list(dataset_columns.where(dataset_columns == True).dropna().index)
        self.remove_redundant()
