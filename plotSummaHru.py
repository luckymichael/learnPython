# -*- coding: utf-8 -*-
"""
Created on Thu May 12 01:34:06 2016

@author: Michael
"""
## %matplotlib inline
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geopandas as gp
import time
import numpy as np
import platform
import os
import xarray as xr

## set working directory
if platform.uname()[0] == 'Linux':
    wdpath=r'/home/mgou/uwhydro/summaProj/summaData/columbia_full_run'
    bsnshp=r'/home/mgou/Dropbox/postdoc/summa/columbia/data/ColumbiaBasin.shp'
    hrushp=r'/home/mgou/Dropbox/postdoc/summa/columbia/data/columbia_hru_output.shp'
   
if platform.uname()[0] == 'Windows':
    wdpath=r'D:\Cloud\Dropbox\postdoc\summa\summaData\columbia_full_run'
    bsnshp=r'D:\Cloud\Dropbox\postdoc\summa\columbia\data\ColumbiaBasin.shp'
    hrushp=r'D:\Cloud\Dropbox\postdoc\summa\columbia\data\columbia_hru_output.shp'
    
if platform.uname()[0] == 'Darwin':
    wdpath=r'/Users/mgou/uwhydro/summaProj/summaData/columbia_full_run'
    bsnshp=r'/Users/mgou/Dropbox/postdoc/summa/columbia/data/ColumbiaBasin.shp'
    hrushp=r'/Users/mgou/Dropbox/postdoc/summa/columbia/data/columbia_hru_output.shp'
    
    
outfolder=wdpath
os.chdir(outfolder)

#%% 

## parameters
mons = ["Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep"]
correct_month_order = [9,10,11,0,1,2,3,4,5,6,7,8]
rows = [0,0,0,0,1,1,1,1,2,2,2,2]
cols = [0,1,2,3,0,1,2,3,0,1,2,3]

vmin=0.01
vmax=1000.0
swe_cmap=plt.get_cmap('Blues')
swe_cmap.set_under(color='tan')
#swe_cmap.set_bad('red',1.) # nan color
#swe_cmap.set_over(color='black')
norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)

# define projection
crs = ccrs.Mercator(central_longitude=-120, min_latitude=40, max_latitude=53, globe=None)

# read the shapefiles
gdf_basin = gp.GeoDataFrame.from_file(bsnshp)
gdf_basin.to_crs(crs=crs.proj4_params,inplace=True)
    
gdf_hru = gp.GeoDataFrame.from_file(hrushp)
gdf_hru.to_crs(crs=crs.proj4_params,inplace=True)

ds_hru = xr.open_dataset("monthlySWE.nc")

# ds_hru.scalarSWE_mean.values[ds_hru.scalarSWE_mean.values > 1e10] = None
# ds_hru.scalarSWE_mean = ds_hru.scalarSWE_mean.where(cond, other=None, drop=False)
swe01 = ds_hru.isel(month=0).scalarSWE_mean.values
idxmax = np.argmax(swe01)
swe01[idxmax]
#%% some function to define basemap


def add_gridlines(axis,labelsize=15):
    gl=axis.gridlines(draw_labels=True, 
                    xlocs = [-100, -110, -115, -120, -125], 
                    ylocs = [40, 42, 44, 46, 48, 50, 52, 54],
                    linewidth=1, color='gray', alpha=0.5, linestyle='--')
                    
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': labelsize}
    gl.ylabel_style = {'size': labelsize}
    return gl

# see this how to change backgroud color: http://stackoverflow.com/questions/32200438/change-the-background-colour-of-a-projected-matplotlib-axis
def add_map_features(ax, states_provinces=True, country_borders=True, land=True, ocean=True,lake=False):
    if states_provinces==True:
        states_provinces = cfeature.NaturalEarthFeature(
                category='cultural',
                name='admin_1_states_provinces_lines',
                scale='50m',
                facecolor='none')
        ax.add_feature(states_provinces, edgecolor='black', zorder = 2) #linewidth = 2

    if country_borders==True:
        country_borders = cfeature.NaturalEarthFeature(
                category='cultural',
                name='admin_0_boundary_lines_land',
                scale='50m',
                facecolor='none')
        ax.add_feature(country_borders, edgecolor='black', zorder = 2, linewidth = 1)

    if land==True:
        land = cfeature.NaturalEarthFeature(
            category='physical',
            name='land',
            scale='50m',
            facecolor='gray')
        ax.add_feature(land,facecolor='lightgray', zorder = 0)

    if ocean==True:
        ocean = cfeature.NaturalEarthFeature(
            category='physical',
            name='ocean',
            scale='50m',
            facecolor='blue')
        ax.add_feature(ocean,facecolor='lightblue', zorder = 1)
        
    if lake==True:
        rivers_lakes = cfeature.NaturalEarthFeature(
            category='physical',
            name='rivers_lake_centerlines',
            scale='50m',
            facecolor='none')
        ax.add_feature(rivers_lakes,facecolor='lightblue', zorder = 2)
    

#%% exercise plotting basin

def gpd_Basin1(gdf = gdf_basin, i = 6):
    # define the projection that the figure is plotted to 
    #ccrs.Mercator?
    plt.style.use('bmh')
    
    #fig, axes = plt.subplots(nrows=3, ncols=4,figsize=(24,20), subplot_kw=dict(projection=crs))
    fig = plt.figure(figsize=(14,14))
    ax  = plt.axes(projection=crs) 
    ax.set_extent([-125.01, -109.5, 40, 53], ccrs.Geodetic())
    ax.set_axis_bgcolor('None')
    ax.set_title(mons[i], fontsize=16)
    gl = add_gridlines(ax)
    add_map_features(ax, land = True, states_provinces = True, country_borders = True) #this will create the problem that no color for geopandas 
    gdf['var'] = i*2150.0
    gdf.plot(ax=ax,column='var',cmap=swe_cmap,linewidth=0.5,vmin=vmin,vmax=vmax,alpha=1.0,zorder=999)
    

        

%time gpd_Basin1(i = 9)


#%% exercise plotting basin for 12 month by geopandas
def gpd_Basin12(gdf = gdf_basin):
    
    gdf['var'] = 0.0    
    
    fig, axes = plt.subplots(nrows=3, ncols=4,figsize=(24,20), subplot_kw=dict(projection=crs))
    
    for i in range(12):
        ax=axes[rows[i],cols[i]]
        ax.set_extent([-125.01, -109.5, 40, 53], ccrs.Geodetic())
        ax.set_axis_bgcolor('lightgrey')
        ax.set_title(mons[i], fontsize=16)
        gl = add_gridlines(ax)
        add_map_features(ax)
        
        gdf['var'] = i*2150.0
        gdf.plot(ax=ax,column='var',cmap=swe_cmap,linewidth=0.5,vmin=vmin,vmax=vmax,alpha=1.0,zorder=999)


        # add 
        cbar_ax_abs = fig.add_axes([0.03, 0.2, 0.015, 0.6])
        cbar_ax_abs.tick_params(labelsize=14)
        cbar_abs = matplotlib.colorbar.ColorbarBase(cbar_ax_abs, cmap=swe_cmap,norm=norm).set_label(label='Snow Water Equivalent (mm)', size=16, labelpad=-85)
        
        fig.savefig('test_basin_plot.png'.format(i),dpi=300,bbox_inches='tight')

%time gpd_Basin12()


#%% plot the actual single HRU output 2016/08/24
def plot_HRU1(gdf = gdf_hru, i = 3):
    
    shpname = hrushp
    
    # plot single one
    fig = plt.figure(figsize=(14,14))
    
    ax  = plt.axes(projection=crs) 
    ## define the coordinate limits
    ax.set_extent([-125.01, -109.5, 41, 53], ccrs.Geodetic())    
    ## set title
    ax.set_title(mons[i], fontsize=16)    
    ## add grid line
    gl = add_gridlines(ax)
    ## add base map features
    add_map_features(ax)    
    
    column='swe{}'.format(i)
    gdf['values'] = ds_hru.isel(month=correct_month_order[i]).scalarSWE_mean.values
    # drop water (nan values)
    gdf = gdf[gdf['values'] < 1e10]
    gdf.plot(ax=ax,column='values',cmap=swe_cmap,vmin=vmin,vmax=vmax,linewidth=0,alpha=1.0,zorder=999)
    
    #cbar = plt.colorbar(img, orientation='vertical', cax = cax)
    cbar_ax_abs = fig.add_axes([0.03, 0.2, 0.015, 0.6])
    cbar_ax_abs.tick_params(labelsize=14)
    cbar_abs = matplotlib.colorbar.ColorbarBase(cbar_ax_abs, cmap=swe_cmap, norm=norm, extend='both').set_label(label='Snow Water Equivalent (mm)', size=16, labelpad=-85)
    fig.savefig('test_plot_Single_HRU{}.png'.format(i), dpi=300,bbox_inches='tight')

%time plot_HRU1()

#%%
%time gdf.plot(column='values',cmap=swe_cmap,vmin=vmin,vmax=vmax,linewidth=0,alpha=1.0)

#%%
for i in range(12):
    plot_HRU1(i = i)
    
# plot colorbar
def plot_colorbar():
    fig = plt.figure(figsize=(1,10))
    cbar_ax_abs = fig.add_axes([0.03, 0.2, 0.4, 0.6])
    cbar_ax_abs.tick_params(labelsize=14)
    cbar_abs = matplotlib.colorbar.ColorbarBase(cbar_ax_abs, cmap=swe_cmap, norm=norm, extend='both').set_label(label='Snow Water Equivalent (mm)', size=16, labelpad=-85)
    fig.savefig('cbar.png',dpi=300,bbox_inches='tight')

%time plot_colorbar()

#%% plot the actual HRU output by geopandas 2016/08/24

def gpd_HRU12(gdf = gdf_hru):
    
    gdf['var'] = 0.0    
    
    fig, axes = plt.subplots(nrows=3, ncols=4,figsize=(24,20), subplot_kw=dict(projection=crs))
    
    for i in range(12):
        ax=axes[rows[i],cols[i]]
        ax.set_extent([-125.01, -109.5, 40, 53], ccrs.Geodetic())
        ax.set_axis_bgcolor('lightgrey')
        ax.set_title(mons[i], fontsize=16)
        gl = add_gridlines(ax)
        add_map_features(ax)
        
        gdf['var'] = ds_hru.isel(month=correct_month_order[i]).scalarSWE_mean.values
        gdf.plot(ax=ax,column='var',cmap=swe_cmap,linewidth=0,vmin=vmin,vmax=vmax,alpha=1.0,zorder=999)


        # add 
        cbar_ax_abs = fig.add_axes([0.03, 0.2, 0.015, 0.6])
        cbar_ax_abs.tick_params(labelsize=14)
        cbar_abs = matplotlib.colorbar.ColorbarBase(cbar_ax_abs, cmap=swe_cmap,norm=norm).set_label(label='Snow Water Equivalent (mm)', size=16, labelpad=-85)
        
        fig.savefig('test_hru_plot.png'.format(i),dpi=300,bbox_inches='tight')

%time gpd_HRU12()
#%% plot old
def gpd_HRU12(gdf = gdf_hru):

plt.style.available
plt.style.use('seaborn-paper')

gdf.describe()

figsize=(10,10)


for i in range(12):
    #ax=axes[rows[i],cols[i]]
    #plt.plot([1,2,3,4])
    gdf.plot(ax=ax,column='swe{}'.format(i+1),cmap=swe_cmap,linewidth=0.0,vmin=vmin,vmax=vmax,alpha=1.0)
    #colname = 'swe{}'.format(i+1)
    #hruvars = gdf[colname]
    #plot_hru(ax, gdfnew, 11800, swe_cmap, norm, hruvars, mons[i], alpha=1.0)
    plt.title(mons[i])
    plt.savefig('test.separate.plot.{:02d}.png'.format(i+1),dpi=300,bbox_inches='tight')
    #fig.clear()

end[2] = time.time()
#f, axes = plt.subplots(nrows=3, ncols=4,figsize=(20,18), subplot_kw=dict(projection=crs))

    #gdf['val'] = i
    #plot = gp.plot(gdf,column='val',ax=ax,cmap=swe_cmap,linewidth=0.0)
    #
#f.subplots_adjust(left=0.1,bottom=0.05,right=0.95,top=0.95)

## Fine-tune figure; make subplots close to each other 
#f.subplots_adjust(hspace=0)

## # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
#plt.setp([a.get_xticklabels() for a in axes[0, :]], visible=False)
#plt.setp([a.get_yticklabels() for a in axes[:, 1]], visible=False)

 #cbar = plt.colorbar(img, orientation='vertical', cax = cax)
    cbar_ax_abs = fig.add_axes([0.03, 0.2, 0.015, 0.6])
    cbar_ax_abs.tick_params(labelsize=14)
    cbar_abs = matplotlib.colorbar.ColorbarBase(cbar_ax_abs, cmap=swe_cmap, norm=norm).set_label(label='Snow Water Equivalent (mm)', size=16, labelpad=-85)
    fig.savefig('test_plot_Single_HRU.png', dpi=300,bbox_inches='tight')


#fig.savefig('test300.png', dpi=300,bbox_inches='tight')
#fig.savefig('test600.png', dpi=600,bbox_inches='tight')

#%% plot
# start to plot
# shpname = 'D:\\@Workspace\\SUMMA\\outputs\\columbia_hru_output.shp'
#shpname = r'/home/mgou/Dropbox/postdoc/summa/columbia/data/columbia_hru_output.shp'

lh = gdf['LtHtTtl']
lhmax = lh.max()
lhmin = lh.min()
lhnew = (lh-lhmin)/(lhmax-lhmin)
#cmap = sns.cubehelix_palette(light=1, as_cmap=True)
cmap = plt.get_cmap('YlOrBr')
colors = cmap(lhnew)

plt.figure(figsize=(10,10))

ax = plt.axes(projection=ccrs.Mercator(central_longitude=-120, min_latitude=41, max_latitude=53, globe=None)) 
ax.set_extent([-109.5, -125.01, 40, 53], ccrs.Geodetic())

gl=ax.gridlines(draw_labels=True, xlocs = [-100, -110, -115, -120, -125], ylocs = [40, 42, 44, 46 ,48, 50, 52])
gl.xlabels_top = False
gl.ylabels_right = False

gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

country_borders = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_boundary_lines_land',
        scale='50m',
        facecolor='none')

land = cfeature.NaturalEarthFeature(
    category='physical',
    name='land',
    scale='50m',
    facecolor='gray')

ocean = cfeature.NaturalEarthFeature(
    category='physical',
    name='ocean',
    scale='50m',
    facecolor='blue')

rivers_lakes = cfeature.NaturalEarthFeature(
    category='physical',
    name='rivers_lake_centerlines',
    scale='50m',
    facecolor='none')

ax.add_feature(land,facecolor='lightgray', zorder = 1)
ax.add_feature(ocean,facecolor='lightblue', zorder = 1)
ax.add_feature(states_provinces, edgecolor='black', zorder = 2) #linewidth = 2
ax.add_feature(country_borders, edgecolor='black', zorder = 2)

# read shape file
#shphru = cartopy.io.shapereader.Reader('D:\\@Workspace\\SUMMA\\shapefiles\\ColumbiaBasin_Proj.shp')
shphru = cartopy.io.shapereader.Reader('D:\\@Workspace\\SUMMA\\outputs\\columbia_hru_output.shp')
hrus = shphru.records()

ihru = 0
for hru in hrus:
    ax.add_geometries(hru.geometry, crs=ccrs.epsg(5070), facecolor=cmap(lhnew[ihru]), zorder=3)
    ihru += 1
    if ihru > 5000:
        break

#%%

import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import numpy as np

NUM_COLORS = 20

cm = plt.get_cmap('gist_rainbow')
cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
fig = plt.figure()
ax = fig.add_subplot(111)
# old way:
#ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
# new way:
ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
for i in range(NUM_COLORS):
    ax.plot(np.arange(10)*(i+1))

fig.savefig('moreColors.png')
plt.show()