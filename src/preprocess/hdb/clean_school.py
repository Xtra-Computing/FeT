import os
import sys
import re
import requests
import json

from tqdm import tqdm
import pandas as pd


def clean_school(rank_path_list, out_path):
    school_df_list = []
    for i, rank_path in enumerate(rank_path_list):
        school_name_list = []
        n_places_after_2b_list = []
        vacancy_rate_list = []
        with open(rank_path, 'r') as f:
            for line in f:
                out_params = re.split('–', line)  # This '–' is not minus symbol '-' in the keyboard
                school_name = out_params[0].strip().lower()
                in_params = re.split('[(),]', out_params[1])
                n_places_after_2b = int(in_params[0])
                vacancy_rate = eval(in_params[2].strip())
                school_name_list.append(school_name)
                n_places_after_2b_list.append(n_places_after_2b)
                vacancy_rate_list.append(vacancy_rate)
        school_df_i = pd.DataFrame({
            'school_name': school_name_list,
            'n_places_{}'.format(i): n_places_after_2b,
            'vacancy_rate_{}'.format(i): vacancy_rate_list
        })
        school_df_i.set_index('school_name', inplace=True)
        school_df_list.append(school_df_i)

    all_school_df = school_df_list[0].join(school_df_list[1:])
    all_school_df.to_csv(out_path, index=True)


def get_school_loc(school_summary_path, out_path):
    school_df = pd.read_csv(school_summary_path)
    school_list = school_df['school_name'].to_list()
    names = []
    lats = []
    lons = []
    for name in tqdm(school_list):
        query_str = "https://developers.onemap.sg/commonapi/search?searchVal=" + str(
            name) + "&returnGeom=Y&getAddrDetails=Y"
        resp = requests.get(query_str)

        # Convert JSON into Python Object
        try:
            data_geo_location = json.loads(resp.content)
        except json.decoder.JSONDecodeError:
            print("Failed to retrieve result")
            continue
        if data_geo_location['found'] != 0:
            lats.append(data_geo_location['results'][0]['LATITUDE'])
            lons.append(data_geo_location['results'][0]['LONGITUDE'])
            names.append(name)
        else:
            print("No Results")

    school_loc_df = pd.DataFrame({'school_name': names,
                                  'lat': lats,
                                  'lon': lons}).set_index('school_name')
    school_df = school_df.set_index('school_name').join(school_loc_df)
    school_df.dropna(inplace=True)
    school_df.to_csv(out_path)


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/hdb")  # change working directory
    # clean_school(["primary_school_rank_2015.txt",
    #               "primary_school_rank_2016.txt",
    #               "primary_school_rank_2017.txt",
    #               "primary_school_rank_2018.txt"],
    #              "school_summary.csv")
    get_school_loc("school_summary.csv", "school_clean.csv")
