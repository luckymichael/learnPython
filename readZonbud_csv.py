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
import osr
import ogr
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
#%% get the file path
sys = platform.uname()
if sys[0] == 'Linux':
    wdpath = r'/media/mgou/Elements/dnr/RRCA/RRCA_Directory_Structure/1050run/condor/results_linux/RRCA'

if sys[0] == 'Darwin':
    wdpath = r'/Volumes/Elements/dnr/RRCA/RRCA_Directory_Structure/1050run/condor/results_linux/RRCA'

if sys[0] == 'Windows':
    wdpath = r'G:\dnr\RRCA\RRCA_Directory_Structure\1050run\condor\results_linux\RRCA'

os.chdir(wdpath)

model_nrow = 90
model_ncol = 326
flow_terms = ['STORAGE', 'CONSTANT HEAD', 'DRAINS', 'ET', 'STREAM LEAKAGE']
rasterOrigin = (266023, 14092806)
pixelWidth = 5280.0
pixelHeight = 5280.0

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
    return flow

#%% https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html

def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array):
    """
    convert a 2d array to a tiff file
    https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
    """
    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(102704)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


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
                # print(cellfile)
                bal = read_bal(cellfile)
                cellname
                bal.columns = ['R{:03d}.C{:03d}'.format(irow+1,icol+1)]
                flow = flow.append(bal.transpose())
            else:
                continue

    return flow

#%% calculate depletion and export the raster
def cal_depletion(df):
    n_flow_term = len(flow_terms)
    # set up depletion raster
    tif = np.empty((model_nrow,model_ncol,n_flow_term),np.float16)
    tif.fill(np.NAN)
    #df = pd.read_csv(csv_file,index_col=0)
    pump0 = df['WELLS']['baseline']
    row_col = df.index.values
    irow = [int(x[1:4])-1 for x in row_col[1:]]
    icol = [int(x[6:9])-1 for x in row_col[1:]]
    for i in range(n_flow_term):
        flow_term = flow_terms[i]        
        tif[irow,icol,i] = - 100. * (df[flow_term][1:].values - df[flow_term]['baseline']) / (df['WELLS'][1:].values - pump0)
    return tif


#%%
def plot_depletion(scen,tif):
    
    fig, axes = plt.subplots(nrows=3, ncols=2,figsize=(20,13)) 
    # set up the color bar
    #cmap = plt.get_cmap('gist_earth')
    #cmap = norm_cmap(values, cmap, Normalize, cm, vmin=vmin, vmax=vmax)
    ax = axes[2,1]
    ax.set_visible(False)
    fig.set_label(scen)
    for i in range(len(flow_terms)):
        row = i // 2
        col = i % 2
        ax = axes[row,col]
        ax.set_title(flow_terms[i])
        ax.set_xlim((95, model_ncol-1))
        ax.set_xlabel('Column number')
        ax.set_ylabel('Row number')
        img = ax.imshow(tif[:,:,i], cmap=plt.get_cmap('gist_earth'),vmin = 0.0, vmax = 100.0, aspect = 'equal')
    # add color bar
    #cax = fig.add_axes([0.03, 0.2, 0.05, 0.5])
    #cax.tick_params(labelsize=14)
    #cbar_abs = matplotlib.colorbar.ColorbarBase(cax, cmap=swe_cmap,norm=norm).set_label(label='Snow Water Equivalent (mm)', size=16, labelpad=-85)
    
    cax = fig.add_axes([0.03, 0.2, 0.03, 0.5])
    #cax.get_xaxis().set_visible(False)
    #cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    cbar = plt.colorbar(img, orientation='vertical', cax = cax)
    cbar.set_label('Flow change percentage (based on pumping change) %', size=13, labelpad=-85)
    plt.savefig(scen + '.flow.pdf', dpi = 300)
    return fig


#%% do all together
for scen in ['NPNM','NPWM','WPNM','WPWM']:
    csv_file = scen + '.flow.csv'
    df_flow = read_scen(scen)
    df_flow = df_flow.transpose()
    df_flow.to_csv(csv_file)    
    tif = cal_depletion(df_flow)
    
    # save as tiff and text file
    i = 0
    for flow_term in flow_terms:
        np.savetxt(scen + '.' + flow_term + '.txt', tif[:,:,i], fmt = '%10.6f')
        array2raster(scen + '.' + flow_term + '.tif', rasterOrigin, pixelWidth, pixelHeight, np.flipud(tif[:,:,i]))
        i += 1    
    
    # plot it
    plot_depletion(scen,tif)
