def correct_NA_values(dataset,
                      method: str = "average"):
    """
    Just to correct the missing values
    :param dataset: The dataframe of the dataset containing the missing values
    :param method: The method that will be used, possible values include,
        "drop": Drops the raws containing the missing values in the dataset
        "average": Replace the available values with average
    :return: The result without any missing values
    """
    if method == "drop":
        return dataset.dropna()

    elif method == "average":
        return dataset.fillna(dataset.mean())

    else:
        raise Exception("The entered format for the method is not valid")


def normalize(dataset):
    return ((dataset - dataset.mean(0)) / dataset.std(0))