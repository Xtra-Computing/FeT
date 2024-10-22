"""
Loader for gisette dataset. Adaption for FeT
"""

import pandas as pd

from src.utils.BasicUtils import move_item_to_start_


def load_both(primary_path, secondary_path):
    print(f'Loading primary from {primary_path}')
    primary = pd.read_csv(primary_path, index_col=False)

    print(f'Loading secondary from {secondary_path}')
    secondary = pd.read_csv(secondary_path, index_col=False)

    labels = primary['y'].to_numpy()
    labels[labels == -1] = 0

    primary.drop(columns=['y'], inplace=True)
    secondary.drop(columns=['y'], inplace=True)

    data1 = primary.to_numpy()
    data2 = secondary.to_numpy()

    return [data1, data2], labels



