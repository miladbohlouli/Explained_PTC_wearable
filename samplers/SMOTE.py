import pandas as pd
from imblearn.over_sampling import SMOTE
from samplers.interface import SAMPLING_INTERFACE


class SMOTE_SAMPLER(SAMPLING_INTERFACE):
    def __init__(self):
        SAMPLING_INTERFACE.__init__(self)
        self.model = SMOTE()

    def fit_resample(self, dataset, labels):
        return self.model.fit_resample(dataset, labels)


