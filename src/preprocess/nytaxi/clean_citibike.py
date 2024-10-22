import os
import sys
from datetime import datetime

import pandas as pd


def fill_zero_padding(s: str):
    """
    Fill a datetime string with zero paddings in days and months
    :param s: original datetime string with format '%-m/%-d/%Y %H:%M:%S'
    :return:
    """
    month, day, other = s.split('/')
    month = "0" + month if len(month) == 1 else month
    day = "0" + day if len(day) == 1 else day
    return month + "/" + day + "/" + other


def clean_bike(bike_ori_data_path, out_bike_data_path, sample_n=None):
    print("Reading from {}".format(bike_ori_data_path))
    date_parser = lambda x: datetime.strptime(fill_zero_padding(x), '%m/%d/%Y %H:%M:%S')
    bike_ori_data = pd.read_csv(bike_ori_data_path, parse_dates=['starttime', 'stoptime'],
                                date_parser=date_parser)

    print("Remove all nonsense data")
    bike_ori_data.dropna(inplace=True)
    bike_ori_data = bike_ori_data[bike_ori_data['tripduration'] < 2000]

    print("Remove useless features from dataset")
    bike_ori_data.drop(columns=['bikeid', 'usertype', 'start station name', 'end station name'], inplace=True)

    print("Get pick-up and drop-off hour")
    bike_ori_data['start_hour'] = bike_ori_data['starttime'].dt.hour
    bike_ori_data['end_hour'] = bike_ori_data['stoptime'].dt.hour

    print("Drop specific time information")
    bike_ori_data.drop(columns=['starttime', 'stoptime'], inplace=True)

    print("Rename columns")
    bike_ori_data.rename(columns={'start station id': 'start_id',
                                  'end station id': 'end_id',
                                  'start station longitude': 'start_lon',
                                  'start station latitude': 'start_lat',
                                  'end station longitude': 'end_lon',
                                  'end station latitude': 'end_lat'}, inplace=True)

    print("Change birth year to age")
    bike_ori_data['age'] = bike_ori_data['birth year'].apply(lambda x: 2016 - x)
    bike_ori_data.drop(columns=['birth year'], inplace=True)

    print("Columns: " + str(bike_ori_data.columns))

    out_bike_data = pd.get_dummies(bike_ori_data,
                                   columns=['gender', 'start_id', 'end_id'],
                                   prefix=['gender', 'sid', 'eid'], drop_first=True)

    print("sampling from dataset")
    if sample_n is not None:
        out_bike_data = out_bike_data.sample(n=sample_n, random_state=0)

    print("Saving cleaned dataset to {}".format(out_bike_data_path))
    out_bike_data.to_pickle(out_bike_data_path)
    print("Saved {} samples to file".format(len(out_bike_data.index)))


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/nytaxi")  # change working directory
    clean_bike("201606-citibike-tripdata.csv", "bike_201606_clean_sample_2e5.pkl", sample_n=200000)
