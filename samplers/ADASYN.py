import pandas as pd
from imblearn.over_sampling import ADASYN
from samplers.interface import SAMPLING_INTERFACE


class ADASYN_SAMPLER(SAMPLING_INTERFACE):
    def __init__(self):
        SAMPLING_INTERFACE.__init__(self)
        self.model = ADASYN()

    def fit_resample(self, dataset, labels):
        return self.model.fit_resample(dataset, labels)


