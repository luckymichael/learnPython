# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 01:27:46 2016

@author: mgou

convert summa netcdf inputs to text format to be used for the master branch
"""

#%%
import xarray as xr
import numpy as np
import pandas as pd
import platform
import os

sys = platform.uname()
if sys[0] == 'Linux':
    wdpath = r'/media/mgou/Elements/dnr/RRCA/RRCA_Directory_Structure/1050run/condor/results_linux/RRCA'

if sys[0] == 'Darwin':
    wdpath = r'/Users/mgou/uwhydro/summaProj/summaData/columbia_full_run/problematicHRU/17000084'

if sys[0] == 'Windows':
    wdpath = r'C:\cygwin64\home\mgou\uwhydro\summaProj\summaData\problematicHRU\17000084'

os.chdir(wdpath)
#%%
ncfile_locatt = 'LocalAttributes_17000084.nc'
ds_locatt = xr.open_dataset(ncfile_locatt)

# text file creater
headers = ['hruId', 'HRUarea', 'latitude', 'longitude', 'elevation', 'tan_slope', 'contourLength', 'mHeight', 'vegTypeIndex', 'soilTypeIndex', 'slopeTypeIndex', 'downHRUindex']
line0 = ''
line1 = ''
for header in headers:
    varstr = '{}'.format(ds_locatt[header].values[0])
    maxwidth = max(len(header), len(varstr)) + 1
    line0 = line0 + header.ljust(maxwidth)
    line1 = line1 + varstr.ljust(maxwidth)

f = open('LocalAttributes.txt','w')
f.writelines(line0 + '\n')
f.writelines(line1)
f.close()

#%%
file_forcing_list = 'forcingFileList_17000084.txt'

f_list = open(file_forcing_list,'r')

for line in f_list:
    if line[0] == '!': continue
    ds= xr.open_dataset(line.rstrip()); break



# text file creater
headers_time = ['ascii_year','ascii_month','ascii_day','ascii_hour','ascii_min','ascii_sec']
headers_forcing = ['pptrate', 'SWRadAtm', 'LWRadAtm', 'airtemp', 'windspd', 'airpres','spechum' ]
#%%
%time complaints = pd.read_csv(r'E:\Downloads\311_Service_Requests_from_2010.csv')
%time complaints = pd.read_csv(r'/media/mgou/DATA/Downloads/311_Service_Requests_from_2010.csv')
# windwos 10: 121 ms
complaints['Complaint Type']
complaints.head(5)
complaints[:5]['Complaint Type']
complaints[['Complaint Type', 'Borough']]
complaints[['Complaint Type', 'Borough']][:10]
complaints['Complaint Type'].value_counts()
