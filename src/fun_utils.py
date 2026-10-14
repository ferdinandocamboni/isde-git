from pandas import read_csv
import numpy as np

import pandas as pd
import numpy as np


def load_mnist(csv_filename):
    """Load the MNIST dataset."""
    # Carica i dati dal file CSV
    data = pd.read_csv(csv_filename)
    z = np.array(data)

    # Estrae le etichette (prima colonna) e le immagini (restanti colonne)
    y = z[:, 0]
    X = z[:, 1:]

    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two random subsets

    """
    pass
