# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:37:43 2016

@author: Michael Ou
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geopandas as gp
import xarray as xr
import argparse

## set arguments
parser = argparse.ArgumentParser(description='Plot Montly Stactistics for SUMMA output.')
#parser.add_argument('-b', '--bsnshp',   help='file path of SUMMA output', required = True)
parser.add_argument('ncname',   help='netcdf file name', )
parser.add_argument('hrushp',   help='file path of hru shapefile, e.g. columbia_hru_output_geo.shp')
parser.add_argument('varname',  help='output variable name')
parser.add_argument('output',   help='output file name of the plot')
parser.add_argument('title',    help='legend title')
parser.add_argument('cmap',     help='colorbar name')
parser.add_argument('vmin',     help='minimum value of color bar')
parser.add_argument('vmax',     help='maximum value of color bar')
parser.add_argument('under',    help='Set color to be used for low out-of-range values')
parser.add_argument('over',     help='Set color to be used for high out-of-range values.')
parser.add_argument('extend',   help='max|min|both|neither')




#%% 
def set_cbar(cbar, vmin, vmax, under=None, over=None):
    cmap=plt.get_cmap(cbar)
    if under != 'None': cmap.set_under(color=under)
    if over  != 'None': cmap.set_over(color=over)    
    norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    return cmap, norm

#%% define projection
def set_proj():
    return ccrs.Mercator(central_longitude=-120, min_latitude=41, max_latitude=53, globe=None)



#%% some function to define basemap


def add_gridlines(axis,labelsize=15):
    gl=axis.gridlines(draw_labels=True, 
                    xlocs = [-100, -110, -115, -120, -125], 
                    ylocs = [42, 44, 46, 48, 50, 52],
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
        
#%%
def gpd_HRU12(gdf, ds, varname='scalarSWE_mean', cmap=None, norm=None, extend=None, legend_title=None, crs=None, vmin=None, vmax=None, output_name=None):
    
    gdf['var'] = 0.0    
    ## parameters
    mons = ["Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep"]
    #correct_month_order = [9,10,11,0,1,2,3,4,5,6,7,8]
    rows = [0,0,0,0,1,1,1,1,2,2,2,2]
    cols = [0,1,2,3,0,1,2,3,0,1,2,3]
    fig, axes = plt.subplots(nrows=3, ncols=4,figsize=(24,20), subplot_kw=dict(projection=crs))
    
    for i in range(12):
        ax=axes[rows[i],cols[i]]
        ax.set_extent([-125.01, -109.5, 41, 53], ccrs.Geodetic())
        ax.set_axis_bgcolor('lightgrey')
        ax.set_title(mons[i], fontsize=16)
        gl = add_gridlines(ax)
        add_map_features(ax)
        
        gdf['var'] = ds[varname].isel(time=i).values
        gdf.plot(ax=ax,column='var',cmap=cmap,linewidth=0,vmin=vmin,vmax=vmax,alpha=1.0,zorder=999)

  
    # add 
    cbar_ax_abs = fig.add_axes([0.03, 0.2, 0.015, 0.6])
    cbar_ax_abs.tick_params(labelsize=14)
    cbar_abs = matplotlib.colorbar.ColorbarBase(cbar_ax_abs, cmap=cmap,norm=norm,extend=extend).set_label(label=legend_title, size=16, labelpad=-85)
    if output_name == None:
        output_name = varname + '.png'
    fig.savefig(output_name.format(i),dpi=300,bbox_inches='tight')
#%%

def main():
    #args = parser.parse_args('-i D:\Cloud\Dropbox\postdoc\summa\summaData\columbia_full_run\columbia2013-2015_2014-2015_G00001-00100_24.nc --hru 1 -v scalarSWE'.split())
    args = parser.parse_args()
    print(args)
    
    ncname        = args.ncname
    hrushp        = args.hrushp
    varname       = args.varname
    plotname      = args.output
    legend_title  = args.title
    cbar          = args.cmap
    vmin          = float(args.vmin)
    vmax          = float(args.vmax)
    under         = args.under
    over          = args.over
    extend        = args.extend

    #ncname   = r'D:\Cloud\Dropbox\postdoc\summa\summaData\columbia_full_run\output\columbia-2014-monthly.nc'
    #hrushp   = r'D:\Cloud\Dropbox\postdoc\summa\columbia\data\columbia_hru_output_geo.shp'
    #var_name = 'scalarSWE_mean'
    #legend_title= 'Snow water equivalent (mm)'
    #plotname = 'D:\\Cloud\\Dropbox\\postdoc\\summa\\summaData\\columbia_full_run\\output\\' + var_name + '.png'

    crs=set_proj()
    cmap, norm = set_cbar(cbar, vmin, vmax, under=under, over=over)
    # read netcdf
    ds_hru = xr.open_dataset(ncname)
    # read the shapefiles    
    gdf_hru = gp.GeoDataFrame.from_file(hrushp)
    gdf_hru.to_crs(crs=crs.proj4_params,inplace=True)
    
    gpd_HRU12(gdf=gdf_hru, ds=ds_hru, varname=varname, cmap=cmap, norm=norm,legend_title=legend_title, crs=crs,vmin=vmin,vmax=vmax, extend=extend, output_name=plotname)
    