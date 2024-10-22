import pandas as pd

from src.utils.BasicUtils import move_item_to_start_, move_item_to_end_


class NYAirbnbTaxiLoader:
    def __init__(self, airbnb_path, taxi_path=None, link=False):
        print("Loading airbnb from {}".format(airbnb_path))
        self.airbnb_data = pd.read_csv(airbnb_path)
        print("Loaded.")
        if taxi_path is not None:
            print("Loading taxi from {}".format(taxi_path))
            self.taxi_data = pd.read_csv(taxi_path)
            print("Loaded.")

        if link:
            self.labels = self.airbnb_data['price'].to_numpy()
            self.airbnb_data.drop(columns=['price'], inplace=True)

            # move lon and lat to end of airbnb
            ab_cols = list(self.airbnb_data)
            ab_cols.insert(len(ab_cols), ab_cols.pop(ab_cols.index('longitude')))
            ab_cols.insert(len(ab_cols), ab_cols.pop(ab_cols.index('latitude')))
            self.airbnb_data = self.airbnb_data[ab_cols]
            print("Current airbnb columns: " + str(list(self.airbnb_data)))
            self.airbnb_data = self.airbnb_data.to_numpy()

            # move lon and lat to the front of taxi
            tx_cols = list(self.taxi_data)
            tx_cols.insert(0, tx_cols.pop(tx_cols.index('lat')))
            tx_cols.insert(0, tx_cols.pop(tx_cols.index('lon')))
            self.taxi_data = self.taxi_data[tx_cols]
            print("Current taxi columns: " + str(list(self.taxi_data)))
            self.taxi_data = self.taxi_data.to_numpy()
        else:
            self.airbnb_data.drop(columns=['longitude', 'latitude'], inplace=True)
            self.labels = self.airbnb_data['price'].to_numpy()
            self.airbnb_data = self.airbnb_data.drop(columns=['price']).to_numpy()

    def load_single(self):
        return self.airbnb_data, self.labels

    def load_parties(self):
        return [self.airbnb_data, self.taxi_data], self.labels


class NYBikeTaxiLoader:
    def __init__(self, bike_path, taxi_path=None, link=False):
        print("Loading bike from {}".format(bike_path))
        self.bike_data = pd.read_pickle(bike_path)
        # self.bike_data = self.bike_data.head(10000)
        # print("Remove N/A from bike")
        # self.bike_data.dropna()
        print("Loaded.")
        if taxi_path is not None:
            print("Loading taxi from {}".format(taxi_path))
            self.taxi_data = pd.read_pickle(taxi_path)
            print("Loaded.")

        if link:
            self.labels = self.bike_data['tripduration'].to_numpy()
            self.bike_data.drop(columns=['tripduration'], inplace=True)

            # move lon and lat to end of airbnb
            bike_cols = list(self.bike_data)
            move_item_to_start_(bike_cols, ['start_lon', 'start_lat', 'end_lon', 'end_lat',
                                          'start_hour', 'end_hour'])
            self.bike_data = self.bike_data[bike_cols]
            self.bike_data.drop(columns=['start_hour', 'end_hour'], inplace=True)
            print("Current bike columns: " + str(list(self.bike_data)))
            self.bike_data = self.bike_data.to_numpy()

            # move lon and lat to the front of taxi
            tx_cols = list(self.taxi_data)
            move_item_to_start_(tx_cols, ['start_lon', 'start_lat', 'end_lon', 'end_lat',
                                          'start_hour', 'end_hour'])
            self.taxi_data = self.taxi_data[tx_cols]
            self.taxi_data.drop(columns=['start_hour', 'end_hour'], inplace=True)
            print("Current taxi columns: " + str(list(self.taxi_data)))
            self.taxi_data = self.taxi_data.to_numpy()
        else:
            print("Remove columns that are used for linkage")
            self.bike_data.drop(columns=['start_lon', 'start_lat', 'end_lon', 'end_lat',
                                         'start_hour', 'end_hour'], inplace=True)
            print('Extract labels')
            self.labels = self.bike_data['tripduration'].to_numpy()
            print("Extract data")
            self.bike_data = self.bike_data.drop(columns=['tripduration']).to_numpy()

    def load_single(self):
        return self.bike_data, self.labels

    def load_parties(self):
        return [self.bike_data, self.taxi_data], self.labels
