import pandas as pd
from src.utils.BasicUtils import move_item_to_start_, move_item_to_end_


def load_house(house_path):
    print("Loading house from {}".format(house_path))
    house_data = pd.read_csv(house_path)

    house_data.drop(columns=['lon', 'lat'], inplace=True)

    house_data.info(verbose=True)

    labels = house_data['totalPrice'].to_numpy()
    house_data = house_data.drop(columns=['totalPrice']).to_numpy()

    return house_data, labels


def load_both(house_path, airbnb_path, active_party='house'):
    print("Loading house from {}".format(house_path))
    house_data = pd.read_csv(house_path)
    print("Loading airbnb from {}".format(airbnb_path))
    airbnb_data = pd.read_csv(airbnb_path)

    if active_party == 'house':
        labels = house_data['totalPrice'].to_numpy()
        house_data.drop(columns=['totalPrice'], inplace=True)

        # move lon and lat to end
        house_cols = list(house_data.columns)
        move_item_to_start_(house_cols, ['lon', 'lat'])
        house_data = house_data[house_cols]
        print("Current house columns {}".format(house_data.columns))

        # move lon and lat to start
        airbnb_cols = list(airbnb_data.columns)
        move_item_to_start_(airbnb_cols, ['lon', 'lat'])
        airbnb_data = airbnb_data[airbnb_cols]
        print("Current airbnb columns {}".format(airbnb_data.columns))

        data1 = house_data.to_numpy()
        data2 = airbnb_data.to_numpy()
    else:
        raise NotImplementedError

    return [data1, data2], labels



