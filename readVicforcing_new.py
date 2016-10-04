# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:06:08 2016

@author: Michael Ou
"""

import pandas as pd
import numpy as np
import os

os.chdir(r'D:\Cloud\Dropbox\postdoc\summa\summaData\blivneh')

#%%
num_skip_line = 5
num_total_hru = 11723

df = pd.read_table(
    r'D:\Cloud\Dropbox\postdoc\summa\summaData\blivneh\full_data_41.21875_-116.21875', skiprows=num_skip_line)


def remove_pound(x):
    if type(x) is str:
        return float(x.lstrip().lstrip('#'))
    else:
        return x

df.iloc[:, 4:] = df.iloc[:, 4:].applymap(remove_pound)


from struct import *

import numpy as np
import pandas as pd

# define column names
col_names = 'year', 'month', 'day', 'hour', 'pptrate', 'airtemp', 'windspd', 'spechum', 'SWRadAtm', 'LWRadAtm', 'airpres'  # , 'asat'
# define column formats
col_format = 'i4',   'i4',    'i4',   'i4',  'f8',      'f8',      'f8',      'f8',      'f8',       'f8',       'f8'  # ,      'f4'
endian = 'little'  # could be big-endian

# read binary output file from VIC forcing generator
vic_file = "D:\\Cloud\\Dropbox\\postdoc\\summa\\summaData\\blivneh\\full_data_41.21875_-116.21875_bin"


def read_vic_bin_forcing(vic_file, col_names, col_format, endian='little'):
    # open data file
    with open(vic_file, "rb") as f:
        # read 4 header identifiers
        id_header = [int.from_bytes(
            f.read(2), byteorder=endian, signed=False) for i in range(4)]
        # get the number of header bytes
        num_header_byte = int.from_bytes(
            f.read(2), byteorder=endian, signed=False)
        # read data in the binary file and output as a pandas DataFrame
        dt = np.dtype({'names': col_names, 'formats': col_format}, align=True)
        f.seek(num_header_byte,  0)
        df = pd.DataFrame(np.fromfile(f, dt))
        return df

#%%
# create ncdf file
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime, timedelta

n = df.shape[0]

# each nc file contain one HRU data
rootgrp = Dataset("D:\\Cloud\\Dropbox\\postdoc\\summa\\summaData\\blivneh\\test.nc",
                  "w", format="NETCDF4_CLASSIC")
dim_time = rootgrp.createDimension("time", n)  # the time variable
# unlimited dim for concatanation
dim_hru = rootgrp.createDimension("hru", None)

# define variables
time = rootgrp.createVariable("time", "f4", ("time",), zlib=True)
time.units = 'hours since 1990-01-01 00:00:00'

# method 1
df_start_date = df.ix[0, :4].to_dict()
df_start_date = dict((k, int(v)) for k, v in df_start_date.items())
dates = [datetime(**df_start_date) + n * timedelta(hours=1) for n in range(n)]
time[:] = date2num(dates, units=time.units)

# method 2
df['time'] = pd.to_datetime(df.iloc[:, :4])
time[:] = date2num(df['time'].astype(datetime), units=time.units)

data_step = rootgrp.createVariable("data_step", "f4", zlib=True)  # scalar
data_step.units = 'seconds'
data_step[:] = np.asarray(3600.0)

hruId = rootgrp.createVariable("hruId", "i4", ("hru",), zlib=True)
hruId[:] = [1]

pptrate = rootgrp.createVariable("pptrate", "f4", ("hru", "time",), zlib=True)
pptrate[:] = df['pptrate'].values.reshape([1, n])

airtemp = rootgrp.createVariable("airtemp", "f4", ("hru", "time",), zlib=True)
airtemp[:] = df['airtemp'].values.reshape([1, n])

windspd = rootgrp.createVariable("windspd", "f4", ("hru", "time",), zlib=True)
windspd[:] = df['windspd'].values.reshape([1, n])

spechum = rootgrp.createVariable("spechum", "f4", ("hru", "time",), zlib=True)
spechum[:] = df['spechum'].values.reshape([1, n])

airpres = rootgrp.createVariable("airpres", "f4", ("hru", "time",), zlib=True)
airpres[:] = df['airpres'].values.reshape([1, n])

LWRadAtm = rootgrp.createVariable(
    "LWRadAtm", "f4", ("hru", "time",), zlib=True)
LWRadAtm[:] = df['LWRadAtm'].values.reshape([1, n])

SWRadAtm = rootgrp.createVariable(
    "SWRadAtm", "f4", ("hru", "time",), zlib=True)
SWRadAtm[:] = df['SWRadAtm'].values.reshape([1, n])

rootgrp.close()

#%% loop all the hru
import geopandas as gd
df = gd.read_file('grid_hru_intersec.dbf')
grouped = df.groupby('hru_id2')
hru_area = grouped.frac_area.sum()
df['frac_frac'] = df.frac_area / hru_area[df.hru_id2].values
df['grid16th'] = 'full_data_' + \
    ((df.ymax + df.ymin) * 0.5).astype(str) + \
    '_' + ((df.xmax + df.xmin) * 0.5).astype(str)

df_forcing_sum = []
for i, row in grouped.get_group(17000001)[['grid16th', 'frac_frac']].iterrows():
    vic_file = row['grid16th']
    df_forcing = read_vic_bin_forcing(vic_file, col_names, col_format)
    df_forcing_sum = df_forcing_sum + df_forcing * row[1][1]
