import pandas as pd
from NAhandlers.interface import NA_HANDLER_INTERFACE
from sklearn.impute import IterativeImputer


class ITERATIVE_IMPUTER(NA_HANDLER_INTERFACE):
    def __init__(self):
        NA_HANDLER_INTERFACE.__init__(self)
        self.imputer = IterativeImputer()

    def fit(self, dataset):
        self.imputer.fit(dataset)

    def fix(self, dataset: pd.DataFrame) -> pd.DataFrame:
        data_set_columns = dataset.columns
        ds = self.imputer.transform(dataset)
        return pd.DataFrame(ds, columns=data_set_columns)
