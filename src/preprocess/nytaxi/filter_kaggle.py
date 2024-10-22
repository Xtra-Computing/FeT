import os
import sys

import pandas as pd


def filter_kaggle(kaggle_train_path, kaggle_out_path):
    print("Start filtering")

    print("Loading training data from csv files")
    kaggle_train = pd.read_csv(kaggle_train_path, index_col=0, parse_dates=['key'])

    print("Filtering data")
    filtered_train = kaggle_train.loc['2009-01-01': '2009-01-31']
    print("Finished filtering training set, got {} samples".format(len(filtered_train.index)))

    print("Saving the filtered data")
    filtered_train.to_csv(kaggle_out_path)
    print("Done")


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/nytaxi")  # change working directory
    filter_kaggle("kaggle_train_ori.csv", "kaggle_data.csv")
