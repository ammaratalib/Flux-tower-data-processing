# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:30:27 2025

@author: ammara
"""

import cdsapi
import datetime
import os

# Set the API credentials as environment variables
os.environ['CDSAPI_URL'] = 'https://cds.climate.copernicus.eu/api'
os.environ['CDSAPI_KEY'] = '0a2b0eec-ed62-4a42-8fb8-83ca726ef382' # get that key from your CDS account 

download_dir = "\\corellia.environment.yale.edu\MaloneLab\Research\era5_data"  # Change this to your desired directory

# Ensure the directory exists
os.makedirs(download_dir, exist_ok=True)

# Define the output file path
output_file = os.path.join(download_dir, "US-Skr.nc")  # Change the filename if needed

# Initialize CDS API client
client = cdsapi.Client()

# Ensure the directory exists
os.makedirs(download_dir, exist_ok=True)

# Define the output file path
output_file = os.path.join(download_dir, "US-skr.nc")  # Change the filename if needed


dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "2m_dewpoint_temperature",
        "2m_temperature",
        "surface_solar_radiation_downwards"
    ],
    "year": [
        "2004", "2005", "2006",
        "2007"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [25.36, -81.1, 25.35, -81.01]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download(target=output_file)

##########################################################################################################




















