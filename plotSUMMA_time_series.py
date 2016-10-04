# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 12:58:40 2016

@author: Michael Ou
"""

import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from matplotlib import gridspec
from matplotlib.collections import PolyCollection
import matplotlib.dates as dates
from datetime import datetime
import pandas as pd        

#%%
tfreeze = 273.16
tzero   = 273.15

ds_all = xr.open_dataset(r"D:\Cloud\Dropbox\postdoc\summa\summaData\problematicHRU\17000025\out.nc")
ds_all.airtemp.values = ds_all.airtemp.values - tzero
ds_all.mLayerTemp.values = ds_all.mLayerTemp.values - tzero
ds_all.iLayerHeight.values = -ds_all.iLayerHeight.values

#%%

ds = ds_all.isel(time=range(100))
ds_all.time
ds.pptrate.values
ds.airtemp.plot(color='purple', marker='d')
ds.nSnow
ds.midTotoStartIndex
ds.ifcTotoStartIndex

ds.mLayerTemp
ds.mLayerVolFracWat
ds.iLayerHeight
ds.mLayerDepth

ds.mLayerDepth.isel(midTotoAndTime = range(ds.nSnow.values[0],ds.nLayers.values[0])).values


#%% plot func
def plot1d(axes, var):
    
#%% plot 1 column
def plotCol(ds, ):
for i in range(ds.time.count().values):
    print(i)
    dt = 
    if i >  5:
        break
#%%

#from pandas.tseries import converter as pdtc
#import matplotlib.units as munits
#import numpy as np

#munits.registry[np.datetime64] = pdtc.DatetimeConverter()

y = np.random.rand(11)
fig, ax = plt.subplots()
plt.plot(times, y)
ax.xaxis_date()
fig.autofmt_xdate()
#%%
# nt = 300  + 1
ds = ds_all #.isel(time=range(nt))
stime = None #dates.date2num(pd.datetime(2013, 1, 1, 0))
etime = None #dates.date2num(pd.datetime(2015, 12, 31, 23))
# create the datetime series that recognized by matplotlib
times = dates.date2num(pd.date_range(start='2013-01-01 00:00:00', end='2016-01-01 00:00:00', periods=None, freq='D').to_pydatetime())
nt = len(times) - 1
vals  = np.empty(0)
verts = np.empty([0, 4, 2])
ihru = 0
for i in range(nt):    
    if (stime != None and times[i] < stime):
        si = i + 1
        continue     
    if (etime != None and times[i] >= etime):
        break
    ei = i    
    idx_time = i #* 24
    nLayers   = ds.nLayers.values[idx_time,ihru]
    ifcStarat = ds.ifcTotoStartIndex.values[idx_time, ihru]                                 # interface starting index
    midStarat = ds.midTotoStartIndex.values[idx_time, ihru]                                 # layer starting index
    heights   = ds.iLayerHeight.values[(ifcStarat - 1):(nLayers+ifcStarat), ihru]           # layer heights
    val_col   = ds.mLayerTemp.values[(midStarat - 1):(nLayers+midStarat-1), ihru]
    vals      = np.append(vals, val_col)     # values
    verts     = addColumn(nLayers, times[i], times[i+1], heights, verts)      # vertices

# clean up null values
mask  = vals < 1e10
vals  = vals[mask]
verts = verts[mask,:,:]
#%% plot
# fig, [axppt, axtemp, axsoil] = plt.subplots(nrows=3, figsize=(20,20), sharex=True, gridspec_kw = {'height_ratios':[1, 1, 3]})
# fig.subplots_adjust(hspace=0.05)
# plt.tight_layout()
fig, axsoil = plt.subplots(figsize=(20,20))
coll = PolyCollection(verts, array = vals, cmap=matplotlib.cm.jet, edgecolors='none')
axsoil.add_collection(coll)
axsoil.autoscale_view()
axsoil.yaxis.set_label_text('depth (m)')
axsoil.xaxis_date()
fig.autofmt_xdate()
#plt.colorbar(coll, label='Temeprature (C)') #, orientation = 'horizontal', anchor = (0.8, 0.5))
# add 
#cbar_ax_abs = fig.add_axes([0.03, 0.2, 0.015, 0.6])
#cbar_ax_abs.tick_params(labelsize=14)
#cbar_abs = matplotlib.colorbar.ColorbarBase(cbar_ax_abs, cmap=swe_cmap,norm=norm).set_label(label='Snow Water Equivalent (mm)', size=16, labelpad=-85)
      
#%% plot curve
ds.pptrate.plot(ax = axppt, color='green')#, marker='d')
axppt.set_title("") #axppt.set_title(ds.pptrate.long_name + " (" + ds.pptrate.units + ")")
ds.airtemp.plot(ax = axtemp, color='purple')#, marker='o')
axtemp.set_title("") #axtemp.set_title(ds.airtemp.long_name + " (C)")
    
#fig.savefig('summa_profiles.png',dpi=300,bbox_inches='tight')

#%%

def addBlock(x1, x2, y1, y2, verts):
    """
    add the verts (four points) of a layer of a column
    """
    block_pts = np.empty([1, 4, 2])
    block_pts[0, 0, 0] = x1; block_pts[0, 0, 1] = y1 # point 1
    block_pts[0, 1, 0] = x2; block_pts[0, 1, 1] = y1 # point 2
    block_pts[0, 2, 0] = x2; block_pts[0, 2, 1] = y2 # point 3
    block_pts[0, 3, 0] = x1; block_pts[0, 3, 1] = y2 # point 4
    verts = np.append(verts ,block_pts, axis = 0)
    return verts
    

def addColumn(n, time1, time2, heights, verts):
    """
    add the vertices of a column
    """
    t1 = time1 #dates.date2num(time1) # time1.astype(datetime) 
    t2 = time2 #dates.date2num(time2) # time2.astype(datetime)
    for x1, x2, y1, y2 in zip(np.repeat(t1, n), np.repeat(t2, n), heights[:-1], heights[1:]):
        #print(x1, x2, y1, y2)
        verts = addBlock(x1, x2, y1, y2, verts)
    return verts
   
    

#%% plot
fig, [axppt, axtemp, axsnow] = plt.subplots(nrows=3, figsize=(10,10), sharex=True, gridspec_kw = {'height_ratios':[1, 1, 3]})
fig.subplots_adjust(hspace=0.05)

plt.tight_layout()
# fig.suptitle('HRU', fontsize=14, fontweight='bold')
plt.plot(ds.time.values, ds.airtemp.values[:,0])
ds.pptrate.plot(ax = axppt, color='green')#, marker='d')
axppt.set_title("") #axppt.set_title(ds.pptrate.long_name + " (" + ds.pptrate.units + ")")
ds.airtemp.plot(ax = axtemp, color='purple')#, marker='o')
axtemp.set_title("") #axtemp.set_title(ds.airtemp.long_name + " (C)")


#%%

fig, axes = plt.subplots(ncols=2)
fig = plt.figure(figsize=(10, 10)); gs = gridspec.GridSpec(nrows = 3, ncols = 1, height_ratios= [1, 1, 3]) 
axppt = plt.subplot(gs[0]); ds.pptrate.plot(ax = axppt)
axtemp = plt.subplot(gs[1]); ds.airtemp.plot(ax = axtemp)
axs = plt.subplot(gs[2]); ds.pptrate.plot(ax = axs)



grid = np.random.rand(4, 3)
fig, axes = plt.subplots(figsize=(12, 6))
axes.imshow(grid, aspect = 'auto')
axes.matshow?






plt.savefig('grid_figure.pdf')









#

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib as mpl

# Generate data. In this case, we'll make a bunch of center-points and generate
# verticies by subtracting random offsets from those center-points
numpoly, numverts = 100, 4
centers = 100 * (np.random.random((numpoly,2)) - 0.5)
offsets = 10 * (np.random.random((numverts,numpoly,2)) - 0.5)
verts = centers + offsets
verts = np.swapaxes(verts, 0, 1)

# In your case, "verts" might be something like:
# verts = zip(zip(lon1, lat1), zip(lon2, lat2), ...)
# If "data" in your case is a numpy array, there are cleaner ways to reorder
# things to suit.

# Color scalar...
# If you have rgb values in your "colorval" array, you could just pass them
# in as "facecolors=colorval" when you create the PolyCollection
z = np.random.random(numpoly) * 500

fig, ax = plt.subplots()

# Make the collection and add it to the plot.
coll = PolyCollection(verts, array=z, cmap=mpl.cm.jet, edgecolors='none')
#coll = PolyCollection(verts)
ax.add_collection(coll)
ax.autoscale_view()

# Add a colorbar for the PolyCollection
fig.colorbar(coll, ax=ax)
plt.show()