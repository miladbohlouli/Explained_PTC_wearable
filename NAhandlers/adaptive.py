import pandas as pd
from NAhandlers.interface import NAhandler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class adaptive(NAhandler):
    def __init__(self):
        NAhandler.__init__(self)
        self.imputer = IterativeImputer()

    def fit(self, dataset):
        self.imputer.fit(dataset)

    def fix(self, dataset: pd.DataFrame) -> pd.DataFrame:
        data_set_columns = dataset.columns
        ds = self.imputer.transform(dataset)
        return pd.DataFrame(ds, columns=data_set_columns)
