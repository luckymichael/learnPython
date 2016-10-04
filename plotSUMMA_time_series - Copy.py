# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 12:58:40 2016

@author: Michael Ou
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import PolyCollection
import matplotlib.dates as dates
from datetime import datetime

tfreeze = 273.16
tzero   = 273.15

ds_all = xr.open_dataset(r"D:\Cloud\Dropbox\postdoc\summa\summaData\problematicHRU\17000025\out.nc")
ds_all.airtemp.values = ds_all.airtemp.values - tzero
ds_all.mLayerTemp.values = ds_all.mLayerTemp.values - tzero
ds_all.iLayerHeight.values = -ds_all.iLayerHeight.values

times = np.arange('2013-01-01T00:00:00', '2016-01-01T00:00:00', step = np.timedelta64(1,'h'), dtype='datetime64[ns]')

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
import pandas as pd        
ds = ds_all.isel(time=range(3))
nLayers = ds.nLayers.values[0,0]
heights = ds.iLayerHeight.values[0:(nLayers+1),0]
verts = np.empty([0, 4, 2])


vals  = ds.mLayerTemp.values[:nLayers]
verts = addColumn(nLayers, ds.time.values[0], ds.time.values[1], heights, verts)
fig, ax = plt.subplots()
coll = PolyCollection(verts, array = vals[:,0])
ax.add_collection(coll)

ax.autoscale_view()

ax.xaxis_date()
fig.autofmt_xdate()

plt.xlim(verts[0,0,0],verts[-1,1,0])

x1 = ds.time.values[0]
x2 = ds.time.values[1]
y1 = heights[0]
y2 = heights[1]
def addBlock(x1, x2, y1, y2, verts):
    block_pts = np.empty([1, 4, 2])
    block_pts[0, 0, 0] = x1; block_pts[0, 0, 1] = y1 # point 1
    block_pts[0, 1, 0] = x2; block_pts[0, 1, 1] = y1 # point 2
    block_pts[0, 2, 0] = x2; block_pts[0, 2, 1] = y2 # point 3
    block_pts[0, 3, 0] = x1; block_pts[0, 3, 1] = y2 # point 4
    verts = np.append(verts ,block_pts, axis = 0)
    return verts
    
n = nLayers
time1 = ds.time.values[0]
time2 = ds.time.values[1]

def addColumn(n, time1, time2, heights, verts):
    t1 = time1 #dates.date2num(time1) # time1.astype(datetime) 
    t2 = time2 #dates.date2num(time2) # time2.astype(datetime)
    for x1, x2, y1, y2 in zip(np.repeat(t1, n), np.repeat(t2, n), heights[:-1], heights[1:]):
        print(x1, x2, y1, y2)
        verts = addBlock(x1, x2, y1, y2, verts)
    return verts
   
    

#%% plot
fig, [axppt, axtemp, axsnow] = plt.subplots(nrows=3, figsize=(10,10), sharex=True, gridspec_kw = {'height_ratios':[1, 1, 3]})
fig.subplots_adjust(hspace=0.05)
plt.tight_layout()
# fig.suptitle('HRU', fontsize=14, fontweight='bold')

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