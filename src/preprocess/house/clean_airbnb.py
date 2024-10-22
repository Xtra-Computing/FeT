import pandas as pd
import os
import sys


def clean_airbnb(airbnb_path, out_airbnb_path):
    airbnb_data = pd.read_csv(airbnb_path)

    # remove useless columns and NA
    airbnb_data.drop(columns=['id', 'name', 'host_id', 'host_name', 'last_review'], inplace=True)
    airbnb_data.dropna(inplace=True)

    # remove extreme high prices
    airbnb_data = airbnb_data[airbnb_data['price'] < 3000]

    airbnb_data.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)

    airbnb_data = pd.get_dummies(airbnb_data,
                                 columns=['neighbourhood', 'room_type'],
                                 prefix=['nbr', 'rt'], drop_first=True)

    print("Got columns " + str(airbnb_data.columns))
    print("Got {} lines".format(len(airbnb_data.index)))

    airbnb_data.to_csv(out_airbnb_path, index=False)


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/beijing")  # change working directory
    clean_airbnb("airbnb.csv", "airbnb_clean.csv")
