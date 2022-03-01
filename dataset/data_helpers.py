def correct_NA_values(dataset,
                      method: str = "drop"):
    """
    Just to correct the missing values
    :param dataset: The dataframe of the dataset containing the missing values
    :param method: The method that will be used, possible values include,
        "drop": Drops the raws containing the missing values in the dataset
    :return: The result without any missing values
    """
    if method == "drop":
        return dataset.dropna()

    else:
        # Todo: convert to debugging message using the Debugginh package
        print("The entered format for the method is not valid")