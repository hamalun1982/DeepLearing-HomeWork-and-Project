import numpy as np
def split_data(data, train_frac, val_frac, test_frac, seed):

    """

    Split a numpy array into three parts for training, validation, and testing.



    Args:

        - data: numpy array, to be split along the first axis

        - train_frac, fraction of data to be used for training

        - val_frac, fraction of data to be used for validation

        - test_frac, fraction of data to be used for testing

        - seed, random seed for reproducibility
    Returns:

        - Training Set

        - Validation Set

        - Testing Set
    
    
    """
    if ((train_frac+val_frac+test_frac) !=1):
        print("ERROR: Train, validation, and test fractions must sum to one.")
    else:
            
        np.random.seed(seed)

        size = data.shape[0]

        split_train = int(train_frac * size)

        split_val = int(val_frac * size)+split_train
       
        np.random.shuffle(data)

        return data[:split_train], data[split_train:split_val], data[split_val:]

