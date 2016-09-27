# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 21:29:06 2016

@author: Michael Ou
"""

import pandas as pd
import numpy as np
from netCDF4 import Dataset, date2num
from datetime import datetime
import argparse

#%% constants
seconds_per_hour = 3600.0
timeunit = 'hours since 1990-01-01 00:00:00'
# define column names for gridded dataframe
col_names  = 'year', 'month', 'day', 'hour', 'pptrate', 'airtemp', 'windspd', 'spechum', 'SWRadAtm', 'LWRadAtm', 'airpres'
# define column formats for gridded dataframe
col_format = 'i4',   'i4',    'i4',   'i4',  'f4',      'f4',      'f4',      'f4',      'f4',       'f4',       'f4'
# define endian (system-dependent)
endian = 'little' # could be big-endian 
    
#%% set arguments
parser = argparse.ArgumentParser(description = 'Convert gridded forcing data for SUMMA HRU')
parser.add_argument('-f', '--fracfile', help = 'file describing hru and grid intersection', required = True)
parser.add_argument('-d', '--datadir',  help = 'directory of gridded forcing data', required = True)
parser.add_argument('-o', '--outdir',   help = 'directory of output files', required = True)

# assign argument values
args      = parser.parse_args()
frac_file = args.fracfile
data_dir  = args.datadir
out_dir   = args.outdir

#%% function read gridded forcing data
def read_vic_bin_forcing(vic_file, col_names, col_format, endian = 'little'):
    # open data file
    with open(vic_file, "rb") as f:
        # read 4 header identifiers
        id_header = [int.from_bytes(f.read(2), byteorder = endian, signed = False) for i in range(4)] 
        # get the number of header bytes
        num_header_byte = int.from_bytes(f.read(2), byteorder = endian, signed = False)        
        # read data in the binary file and output as a pandas DataFrame
        dt = np.dtype({'names': col_names, 'formats': col_format}, align=True)
        f.seek(num_header_byte,  0)
        df = pd.DataFrame(np.fromfile(f, dt))
        return df

      
#%% create ncdf file     
def to_nc(n, times, df, ihru, nc_name):
    '''
    Inputs:
        n:       number of rows
        time:    time stamps as hours since yyyy-mm-dd HH:MM:SS
        df:      dataframe including pptrate, airtemp, windspd, spechum, airpres, LWRadAtm, SWRadAtm
        ihru:    index of HRU
        nc_name: name of the output netcdf file
    Outputs:
        write a netcdf file named nc_name
    '''

    # each nc file contain one HRU data
    rootgrp = Dataset(nc_name, "w", format="NETCDF4_CLASSIC")
    
    # define dimensions
    dim_hru  = rootgrp.createDimension("hru",  None)     # unlimited dim for concatanation
    dim_time = rootgrp.createDimension("time", n)       # the time dimension
    
    # define variables
    time = rootgrp.createVariable("time","f4",("time",),zlib=True)
    time.units = timeunit
    time[:] = times
        
    data_step = rootgrp.createVariable("data_step","f4",zlib=True) 
    data_step.units = 'seconds'
    data_step.assignValue(seconds_per_hour)
    
    hruId = rootgrp.createVariable("hruId","i4",("hru",),zlib=True)
    hruId[:] = (np.array(ihru))
    
    pptrate = rootgrp.createVariable("pptrate","f4",("hru","time",),zlib=True)
    pptrate[:] = df['pptrate'].values.reshape([1,n])
    
    airtemp = rootgrp.createVariable("airtemp","f4",("hru","time",),zlib=True)
    airtemp[:] = df['airtemp'].values.reshape([1,n])
    
    windspd = rootgrp.createVariable("windspd","f4",("hru","time",),zlib=True)
    windspd[:] = df['windspd'].values.reshape([1,n])
    
    spechum = rootgrp.createVariable("spechum","f4",("hru","time",),zlib=True)
    spechum[:] = df['spechum'].values.reshape([1,n])
    
    airpres = rootgrp.createVariable("airpres","f4",("hru","time",),zlib=True)
    airpres[:] = df['airpres'].values.reshape([1,n])
    
    LWRadAtm = rootgrp.createVariable("LWRadAtm","f4",("hru","time",),zlib=True)
    LWRadAtm[:] = df['LWRadAtm'].values.reshape([1,n])
    
    SWRadAtm = rootgrp.createVariable("SWRadAtm","f4",("hru","time",),zlib=True)
    SWRadAtm[:] = df['SWRadAtm'].values.reshape([1,n])
    
    return rootgrp.close()
        
#%% calculate the areal average of forcing data
if not data_dir.endswith('/'):
    data_dir = data_dir + '/'
    
if not out_dir.endswith('/'):
    out_dir = out_dir + '/'
    
# open fraction file
df_frac = pd.read_csv(frac_file)
df_frac.lat = df_frac.lat.astype(str)
df_frac.lon = df_frac.lon.astype(str)
df_frac['latlon'] =  df_frac.lat + '_' + df_frac.lon

# read a sample and extract time infomation
df_sample = read_vic_bin_forcing(data_dir + 'full_data_' + df_frac.latlon[0], col_names, col_format, endian)
nrow = df_sample.shape[0]

# present time in netcdf preferred unit
df_times = pd.to_datetime(df_sample.iloc[:,:4])
times = date2num(df_times.astype(datetime), units=timeunit)
# read a sample

# group by hru
group_hru = df_frac.groupby('hruID')
hru_index = group_hru.hruID.first().values

for hru in hru_index:
    hru_panel = pd.Panel({ 
                    'frac' + str(row.Index) : read_vic_bin_forcing(data_dir + 'full_data_' + row.latlon, col_names, col_format, endian).iloc[:,4:] * row.frac 
                    for row in group_hru.get_group(hru).itertuples() })
    df_hru = hru_panel.sum(axis = 0)
    to_nc(nrow, times, df_hru, hru, out_dir + str(hru) + '.nc')
    #break

print('cong, finish !!!')