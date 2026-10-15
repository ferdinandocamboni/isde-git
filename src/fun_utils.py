from pandas import read_csv
import numpy as np

import pandas as pd
import numpy as np


def load_data(filename):
    """Load the MNIST dataset."""
    # Carica i dati dal file CSV
    data = pd.read_csv(filename)
    z = np.array(data)

    # Estrae le etichette (prima colonna) e le immagini (restanti colonne)
    y = z[:, 0]
    X = z[:, 1:]

    return X, y


def split_data(x, y, tr_fraction=0.5):
    #Split the data x, y into two random subsets

    idx = np.array(range(x.shape[0]))
    np.random.shuffle(idx)
    ntr = int(x.shape[0]*tr_fraction)

    xtr = x[idx[0:ntr], :]
    ytr = y[idx[0:ntr]]
    xts = x[idx[ntr:], :]
    yts = y[idx[ntr:]]

    return xtr, ytr, xts, yts