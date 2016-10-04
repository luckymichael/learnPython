#!/usr/lusers/mgou/.conda/envs/py35/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:49:48 2016

@author: Michael Ou

Read SUMMA netCDF files and output monthly mean 

"""
import argparse
import ntpath
import xarray as xr
#import pandas as pd
#from datetime import datetime
import numpy as np

#dt = datetime.utcnow()
#dt
#dt64 = np.datetime64(dt)
#dt64

parser = argparse.ArgumentParser(description='Write Montly Stactistics for SUMMA output.')
parser.add_argument('-i', '--fin',       help='file path of SUMMA output', required = True)
parser.add_argument('-o', '--fout',      help='file path of statistics output')
parser.add_argument('-v', '--variable',  help='variable name for calculation', required = True)
parser.add_argument('-s', '--starttime', help='starting time', )
parser.add_argument('-e', '--endtime',   help='endding time')
parser.add_argument('--hru',       help='starting HRU index')
#parser.parse_args(['-h'])

#args = parser.parse_args('-i D:\Cloud\Dropbox\postdoc\summa\summaData\columbia_full_run\columbia2013-2015_2014-2015_G00001-00100_24.nc --hru 1 -v scalarSWE'.split())
args = parser.parse_args()
print(args)

fin_name = ntpath.basename(args.fin).rstrip(".nc")
# create the time series until the time dimension variable is written correctly
#times = np.arange('2014-10-01T00:00:00', '2015-10-01T00:00:00', step = np.timedelta64(1,'D'), dtype='datetime64[ns]')

if args.starttime == None:
    stime = fin_name[fin_name.index("_") + 1 : fin_name.index("_") + 5] + "-10-01"
else:
    stime = args.starttime

if args.endtime == None:
    etime = fin_name[fin_name.index("_") + 6 : fin_name.index("_") + 10] + "-10-01"
else:
    etime = args.endtime

if args.fout == None: 
    args.fout = args.fin[:args.fin.rindex("_")] + "-Month-" + str(args.variable) + ".nc"
    
times = np.arange(stime, etime, dtype='datetime64[D]')

var_name = args.variable

ds = xr.open_dataset(args.fin)
ds['time'] = times
hru_month = ds[var_name + '_mean'].groupby('time.month').mean('time').to_dataset()  # mean('time') only takes mean on time otherwise on time and hru
hru_month['hru'] = int(args.hru) + hru_month['hru']

hru_month.to_netcdf(args.fout)

