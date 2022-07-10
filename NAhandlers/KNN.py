from sklearn.impute import KNNImputer
from NAhandlers.interface import NA_HANDLER_INTERFACE
import pandas as pd


class KNN_IMPUTER(NA_HANDLER_INTERFACE):
    def __init__(self):
        NA_HANDLER_INTERFACE.__init__(self)
        self.imputer = KNNImputer()

    def fit(self, dataset):
        self.imputer.fit(dataset)

    def fix(self, dataset: pd.DataFrame) -> pd.DataFrame:
        data_set_columns = dataset.columns
        ds = self.imputer.transform(dataset)
        return pd.DataFrame(ds, columns=data_set_columns)
