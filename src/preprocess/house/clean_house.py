import pandas as pd
import os
import sys


def clean_house(house_path, out_house_path, include_cid=False):
    house_data = pd.read_csv(house_path, encoding="iso-8859-1", parse_dates=['tradeTime'],
                             dtype={'Cid': 'category'})

    house_data.dropna(inplace=True)

    house_data['buildingType'] = house_data['buildingType'].astype('int')

    # remove the houses sold before 2013
    house_data = house_data[house_data['tradeTime'].dt.year > 2012]

    house_data['trade_year'] = house_data['tradeTime'].dt.year
    house_data['trade_month'] = house_data['tradeTime'].dt.month

    # rename longitude and latitude
    house_data.rename(columns={'Lng': 'lon', 'Lat': 'lat', 'Cid': 'cid'}, inplace=True)

    # # filter too large data
    # house_data = house_data[house_data['DOM'] < 365]

    # remove non-numeric values in constructionTime
    house_data['constructionTime'] = house_data['constructionTime'].str.extract('(\d+)', expand=False)

    # remove non-numeric values in floor
    house_data['floor'] = house_data['floor'].str.extract('(\d+)', expand=False)

    # remove houses with prices extremely large or small [10w, 2000w)
    house_data = house_data[house_data['totalPrice'] >= 10]
    house_data = house_data[house_data['totalPrice'] < 1000]

    # one-hot categorical features
    if include_cid:
        house_data = pd.get_dummies(house_data,
                                    columns=['cid', 'district', 'buildingType', 'renovationCondition',
                                             'buildingStructure', 'trade_year', 'trade_month'],
                                    prefix=['cid', 'did', 'bt', 'rc', 'bs', 'ty', 'tm'], drop_first=True)
    else:
        house_data = pd.get_dummies(house_data,
                                    columns=['district', 'buildingType', 'renovationCondition', 'buildingStructure',
                                             'trade_year', 'trade_month'],
                                    prefix=['did', 'bt', 'rc', 'bs', 'ty', 'tm'], drop_first=True)

    # price is not needed to predict totalPrice, otherwise totalPrice = price * squares
    house_data.drop(columns=['url', 'id', 'communityAverage', 'price', 'tradeTime'], inplace=True)

    print("Got columns " + str(house_data.columns))
    print("Got {} lines".format(len(house_data.index)))

    house_data.dropna(inplace=True)

    house_data.to_csv(out_house_path, index=False)


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/beijing")  # change working directory
    clean_house("house.csv", "house_clean.csv")

