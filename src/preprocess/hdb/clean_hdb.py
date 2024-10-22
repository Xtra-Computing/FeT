import os
import sys
import requests
import json

from tqdm import tqdm
import pandas as pd


def get_blk_loc(hdb_path, out_hdb_w_blk_loc_path):
    hdb_df = pd.read_csv(hdb_path, parse_dates=['month'])

    hdb_df['address'] = hdb_df['block'] + " " + hdb_df['street_name']
    addrs_unique = hdb_df['address'].drop_duplicates(keep='first')

    # get the address of blocks by OneMap API
    latitude = []
    longitude = []
    blk_no = []
    road_name = []
    postal_code = []
    address = []
    for addr in tqdm(addrs_unique):
        query_string = 'https://developers.onemap.sg/commonapi/search?searchVal=' + str(
            addr) + '&returnGeom=Y&getAddrDetails=Y'
        resp = requests.get(query_string)

        # Convert JSON into Python Object
        data_geo_location = json.loads(resp.content)
        if data_geo_location['found'] != 0:
            latitude.append(data_geo_location['results'][0]['LATITUDE'])
            longitude.append(data_geo_location['results'][0]['LONGITUDE'])
            blk_no.append(data_geo_location['results'][0]['BLK_NO'])    # this one is a unique block No.
            road_name.append(data_geo_location['results'][0]['ROAD_NAME'])
            postal_code.append(data_geo_location['results'][0]['POSTAL'])
            address.append(addr)
            # print(str(addr) + " ,Lat: " + data_geo_location['results'][0]['LATITUDE'] + " Long: " +
            #       data_geo_location['results'][0]['LONGITUDE'])
        else:
            print("No Results")

    print("Converting to dataframe")
    block_loc_df = pd.DataFrame({'address': address, 'lat': latitude, 'lon': longitude})
    print("Joining with HDB data")
    hdb_df = hdb_df.merge(block_loc_df, on='address', how='left')
    print("Saving to {}".format(out_hdb_w_blk_loc_path))
    hdb_df.to_csv(out_hdb_w_blk_loc_path, index=False)


def clean_hdb(hdb_path, out_hdb_path):
    hdb_df = pd.read_csv(hdb_path, parse_dates=['month'])
    hdb_df.dropna(inplace=True)
    hdb_df.drop(columns=['month', 'block', 'street_name', 'address'], inplace=True)

    hdb_df['lease_commence_year_before_2020'] = 2020 - hdb_df['lease_commence_date']
    hdb_df.drop(columns=['lease_commence_date', 'remaining_lease'], inplace=True)

    hdb_df = pd.get_dummies(hdb_df,
                            columns=['town', 'flat_type', 'storey_range', 'flat_model'],
                            prefix=['tn', 'ft', 'sr', 'fm'], drop_first=True)

    hdb_df['resale_price'] = hdb_df['resale_price'] / 1000  # change to kS$

    hdb_df.to_csv(out_hdb_path, index=False)


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../../data/hdb")  # change working directory
    # get_blk_loc("resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv",
    #             "hdb_2017_onwards_w_blk_loc.csv")
    clean_hdb("hdb_2017_onwards_w_blk_loc.csv",
              "hdb_clean.csv")