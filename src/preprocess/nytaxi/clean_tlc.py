import os
import sys

import pandas as pd


def clean_tlc_for_airbnb(tlc_ori_data_path, out_tlc_data_path, sample_n=None, keep_col=None):
    print("Reading from {}".format(tlc_ori_data_path))
    tlc_ori_data = pd.read_csv(tlc_ori_data_path, parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

    print("get pick-up and drop-off hour")
    tlc_ori_data.drop(columns=['store_and_fwd_flag'], inplace=True)

    print("get pick-up and drop-off hour")
    tlc_ori_data['pickup_hour'] = tlc_ori_data['tpep_pickup_datetime'].dt.hour
    tlc_ori_data['dropoff_hour'] = tlc_ori_data['tpep_dropoff_datetime'].dt.hour

    print("drop specific time information")
    tlc_ori_data.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], inplace=True)

    print("divide pickup and dropoff dataset")
    tlc_ori_data_pickup = tlc_ori_data.drop(columns=['dropoff_hour', 'dropoff_longitude', 'dropoff_latitude'])
    tlc_ori_data_pickup['is_pickup'] = 1
    tlc_ori_data_pickup.rename(columns={'pickup_hour': 'hour',
                                        'pickup_longitude': 'lon',
                                        'pickup_latitude': 'lat'}, inplace=True)
    tlc_ori_data_dropoff = tlc_ori_data.drop(columns=['pickup_hour', 'pickup_longitude', 'pickup_latitude'])
    tlc_ori_data_dropoff.rename(columns={'dropoff_hour': 'hour',
                                         'dropoff_longitude': 'lon',
                                         'dropoff_latitude': 'lat'}, inplace=True)
    tlc_ori_data_dropoff['is_pickup'] = 0

    print("concat pickup and dropoff dataset by rows")
    out_tlc_data = pd.concat([tlc_ori_data_pickup, tlc_ori_data_dropoff])
    print("Finished, print all the columns:")
    print(out_tlc_data.dtypes)

    if keep_col is None:
        print("make categorical features one-hot")
        out_tlc_data = pd.get_dummies(out_tlc_data,
                                      columns=['hour', 'VendorID', 'RatecodeID', 'payment_type'],
                                      prefix=['hr', 'vid', 'rid', 'pt'], drop_first=True)
    else:
        print("Filter columns {}".format(keep_col))
        out_tlc_data = out_tlc_data[keep_col + ['lon', 'lat']]
        print("make categorical features one-hot")
        dummy_col, dummy_prefix = [], []
        col_prefix = {
            'hour': 'hr',
            'VendorID': 'vid',
            'RatecodeID': 'rid',
            'payment_type': 'pt'
        }
        for col, prefix in col_prefix.items():
            if col in out_tlc_data.columns:
                dummy_col.append(col)
                dummy_prefix.append(prefix)
        out_tlc_data = pd.get_dummies(out_tlc_data, columns=dummy_col, prefix=dummy_prefix, drop_first=True)

    print("sampling from dataset")
    if sample_n is not None:
        out_tlc_data = out_tlc_data.sample(n=sample_n, random_state=0)

    print("Saving cleaned dataset to {}".format(out_tlc_data_path))
    out_tlc_data.to_csv(out_tlc_data_path, index=False)
    print("Saved {} samples to file".format(len(out_tlc_data.index)))


def clean_tlc_for_bike(tlc_ori_data_path, out_tlc_data_path, sample_n=None):
    print("Reading from {}".format(tlc_ori_data_path))
    tlc_ori_data = pd.read_csv(tlc_ori_data_path, parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

    print("Drop values that are not reasonable")
    tlc_ori_data.dropna(inplace=True)
    tlc_ori_data = tlc_ori_data[tlc_ori_data['trip_distance'] > 0]
    tlc_ori_data = tlc_ori_data[tlc_ori_data['trip_distance'] < 10]

    print("get duration of the trip")
    tlc_ori_data['taxi_duration'] = (tlc_ori_data['tpep_dropoff_datetime']
                                     - tlc_ori_data['tpep_pickup_datetime']).astype('timedelta64[s]')
    tlc_ori_data = tlc_ori_data[tlc_ori_data['taxi_duration'] > 0]
    tlc_ori_data = tlc_ori_data[tlc_ori_data['taxi_duration'] < 10000]

    print("get pick-up and drop-off hour")
    tlc_ori_data['start_hour'] = tlc_ori_data['tpep_pickup_datetime'].dt.hour
    tlc_ori_data['end_hour'] = tlc_ori_data['tpep_dropoff_datetime'].dt.hour

    print("drop specific time information")
    tlc_ori_data.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], inplace=True)

    print("divide pickup and dropoff dataset")
    tlc_ori_data.rename(columns={'pickup_longitude': 'start_lon',
                                 'pickup_latitude': 'start_lat',
                                 'dropoff_longitude': 'end_lon',
                                 'dropoff_latitude': 'end_lat'}, inplace=True)

    print("Drop useless features")
    out_tlc_data = tlc_ori_data[['start_lon', 'start_lat', 'end_lon', 'end_lat',
                                 'start_hour', 'end_hour', 'trip_distance', 'taxi_duration']]

    print("sampling from dataset")
    if sample_n is not None:
        out_tlc_data = out_tlc_data.sample(n=sample_n, random_state=0)

    print("Saving cleaned dataset to {}".format(out_tlc_data_path))
    out_tlc_data.to_pickle(out_tlc_data_path)
    print("Saved {} samples to file".format(len(out_tlc_data.index)))


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/nytaxi")  # change working directory
    # clean_tlc("yellow_tripdata_2016-06.csv", "taxi_201606_clean.csv", sample_n=None)
    # clean_tlc_for_airbnb("yellow_tripdata_2016-06.csv", "taxi_201606_clean_sample_1e6.csv",
    #                      sample_n=1000000, keep_col=['RatecodeID', 'tip_amount'])
    clean_tlc_for_bike("yellow_tripdata_2016-06.csv", "taxi_201606_clean_sample_1e5.pkl",
                       sample_n=100000)
