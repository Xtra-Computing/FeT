"""
Loader for HDB dataset
"""

import pandas as pd

from src.utils.BasicUtils import move_item_to_start_


def load_hdb(hdb_path):
    print(f'Loading hdb from {hdb_path}')
    hdb_data = pd.read_csv(hdb_path)

    hdb_data.drop(columns=['lon', 'lat'])

    hdb_data.info(verbose=True)

    labels = hdb_data['resale_price'].to_numpy()
    hdb_data = hdb_data.drop(columns=['resale_price']).to_numpy()

    return hdb_data, labels


def load_both(hdb_path, airbnb_path, active_party='hdb'):
    print(f'Loading house from {hdb_path}')
    hdb_data = pd.read_csv(hdb_path)
    print(f'Loading airbnb from {airbnb_path}')
    school_data = pd.read_csv(airbnb_path)

    if active_party == 'hdb':
        labels = hdb_data['resale_price'].to_numpy()
        hdb_data.drop(columns=['resale_price'], inplace=True)

        # move lon and lat to end
        hdb_cols = list(hdb_data.columns)
        move_item_to_start_(hdb_cols, ['lon', 'lat'])
        hdb_data = hdb_data[hdb_cols]
        print(f'Current hdb columns {hdb_data.columns}')

        school_data.drop(columns=['school_name'], inplace=True)

        # move lon and lat to start
        school_cols = list(school_data.columns)
        move_item_to_start_(school_cols, ['lon', 'lat'])
        school_data = school_data[school_cols]
        print(f'Current airbnb columns {school_data.columns}')

        data1 = hdb_data.to_numpy()
        data2 = school_data.to_numpy()
    else:
        raise NotImplementedError

    return [data1, data2], labels



