from Feature_selectors.interface import FeatureSelector
import pandas as pd

class all_feature_selector(FeatureSelector):
    def __init__(self, include_changing_parameters: bool = False):
        FeatureSelector.__init__(self, include_changing_parameters)

        self.selected_features_list += ['mean.Temperature_60', 'grad.Temperature_60', 'sd.Temperature_60',
                                        'mean.Temperature_480', 'grad.Temperature_480', 'sd.Temperature_480',
                                        'mean.Humidity_60', 'grad.Humidity_60', 'sd.Humidity_60',
                                        'mean.Humidity_480', 'grad.Humidity_480', 'sd.Humidity_480',
                                        'mean.Winvel_60', 'grad.Winvel_60', 'sd.Winvel_60', 'mean.Winvel_480',
                                        'grad.Winvel_480', 'sd.Winvel_480', 'mean.Solar_60', 'grad.Solar_60',
                                        'sd.Solar_60', 'mean.Solar_480', 'grad.Solar_480', 'sd.Solar_480',
                                        'mean.hr_5', 'grad.hr_5', 'sd.hr_5', 'mean.hr_15', 'grad.hr_15',
                                        'sd.hr_15', 'mean.hr_60', 'grad.hr_60', 'sd.hr_60', 'mean.WristT_5',
                                        'grad.WristT_5', 'sd.WristT_5', 'mean.WristT_15', 'grad.WristT_15',
                                        'sd.WristT_15', 'mean.WristT_60', 'grad.WristT_60', 'sd.WristT_60',
                                        'mean.AnkleT_5', 'grad.AnkleT_5', 'sd.AnkleT_5', 'mean.AnkleT_15',
                                        'grad.AnkleT_15', 'sd.AnkleT_15', 'mean.AnkleT_60', 'grad.AnkleT_60',
                                        'sd.AnkleT_60', 'mean.PantT_5', 'grad.PantT_5', 'sd.PantT_5',
                                        'mean.PantT_15', 'grad.PantT_15', 'sd.PantT_15', 'mean.PantT_60',
                                        'grad.PantT_60', 'sd.PantT_60', 'mean.act_5', 'grad.act_5', 'sd.act_5',
                                        'mean.act_15', 'grad.act_15', 'sd.act_15', 'mean.act_60', 'grad.act_60',
                                        'sd.act_60']

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
