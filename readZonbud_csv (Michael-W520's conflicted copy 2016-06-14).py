# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import platform
import os
import gdal
#%% get the file path
sys = platform.uname()
if sys[0] == 'Linux':
    wdpath=r'/media/mgou/Elements/dnr/RRCA/RRCA_Directory_Structure/1050run/condor/results_linux/RRCA'
    
if sys[0] == 'Darwin':
    wdpath=r'/Volumes/Elements/dnr/RRCA/RRCA_Directory_Structure/1050run/condor/results_linux/RRCA'

if sys[0] == 'Windows':
    wdpath=r'G:\dnr\RRCA\RRCA_Directory_Structure\1050run\condor\results_linux\RRCA'
    
os.chdir(wdpath)

model_nrow = 100
model_ncol = 326
#%% read one bal file
def read_bal(file_path):
    col_widths = [20,2,17,2,20,2,17]
    skiprows = np.array([1,2,3,4,5,6,7,8,9,17,18,19,20,21,29,30,31,32,33,34,35,36,37])-1
    bal = pd.read_fwf(file_path,widths = col_widths,skiprows=skiprows,header=None)
    net_flow = bal[[0,2]][0:7]
    net_flow = pd.DataFrame( bal[2][0:7].values - bal[2][7:14].values )
    net_flow.columns = ['net_flow']
    net_flow.index = bal[0][0:7].values
    return net_flow
#%% read a list: Wall time: 49.1 s
def read_scen(scen):
    # read the baseline    
    flow = read_bal('baseline/{0}.R001.C001.bal'.format(scen))
    flow.columns = ['baseline']
    
    for irow in range(model_nrow):
        for icol in range(model_ncol):
            cellfile = 'result-{0}/{0}.R{1:03d}.C{2:03d}.bal'.format(scen,irow+1,icol+1) 
            if os.path.isfile(cellfile): 
                #print(cellfile)
                bal = read_bal(cellfile)
                flow['R{:03d}.C{:03d}'.format(irow+1,icol+1)] = bal.values
                #break
            else:
                continue
    
    return depletion
#%% read a list: Wall time: 51.1 s
def read_scen1(scen):
    # read the baseline
    tif = np.empty((model_nrow,model_ncol,6),np.float16)
    tif.fill(np.NAN)
    flow = read_bal('baseline/{0}.R001.C001.bal'.format(scen))
    flow.columns = ['baseline']
    flow = flow.transpose()
    for irow in range(model_nrow):
        for icol in range(model_ncol):
            cellname = 'R{:03d}.C{:03d}'.format(irow+1,icol+1) 
            cellfile = 'result-{0}/{0}.'.format(scen) + cellname + '.bal'
            if os.path.isfile(cellfile): 
                #print(cellfile)
                bal = read_bal(cellfile)
                cellname
                bal.columns = ['R{:03d}.C{:03d}'.format(irow+1,icol+1)]
                wel_diff = flow['WELLS']['baseline']
                flow = flow.append(bal.transpose())
            else:
                continue
    
    return depletion
        
#%% calculate depletion and export the raster
def cal_depletion(csv_file):
    n_flow_term = 6
    # set up depletion raster
    tif = np.empty((model_nrow,model_ncol,n_flow_term),np.float16)
    tifna = tif.fill(np.NAN)
    df = pd.read_csv(csv_file,index_col=0)       
    df['row'] = map(lambda x: int(x),[x[1:4] for x in df.index.values])
        
#%%
scen = 'NPNM'
%time d = read_scen(scen).transpose()
csv_file = scen + '.flow.csv'
d.to_csv(scen + '.flow.csv')
#%% https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html

def array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array):

    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    #outRasterSRS = osr.SpatialReference()
    #outRasterSRS.ImportFromEPSG(4326)
    #outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

#%%
%time c = read_scen1('NPNM')
tif_driver = gdal.GetDriverByName('GTiff')
if outDs is None:
    print 'Could not create reclass_40.tif'
    #exit(1)
tif = gdal.GetDriverByName('GTiff').Create(scen + '.tif', model_ncol, model_nrow, 1 ,gdal.GDT_Float32)  # Open the file

#%%
import numpy as np
import matplotlib.pyplot as plt

H = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])  # added some commas and array creation code

fig = plt.figure(figsize=(6, 3.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(H)
ax.set_aspect('equal')

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()