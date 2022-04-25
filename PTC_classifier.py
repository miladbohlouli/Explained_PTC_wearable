from data import PTC_dataset


class Classifier:
    """
    The above class is the general structure of the classifiers for the PTC models
    """
    def __init__(self,
                 model=None,
                 train_data_loader: PTC_dataset = None,
                 num_inputs: int = 13,
                 num_classes: int = 3
                 ):
        # You may define all the processes that are common between all the trainings
        # every classifer should have a data_loader and a model
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.model = model
        self.train_data_loader = train_data_loader

    def fit(self):
        """
        Will be used for training the provided model (is specific for each model e.g. for decision tree and mlps)
            based on the train_data_loader
        """
        raise Exception("Training function not yet implemented")

    def predict(self, X):
        """
        Will be used for
        :param X:
        :return:
        """
        raise Exception("Predicting the model not yet implemented")
