#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot-prelim-maps.py
#------------------------------------------------------------------------------
# Version 0.1
# 16 July, 2020
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.cm as cm
from matplotlib import colors as mcol
from matplotlib.cm import ScalarMappable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.ticker as mticker
# Mapping libraries:
import cartopy
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cf
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

use_minmax = False
use_horizontal_colorbar = True
projection = 'equalearth'
#cmap = 'viridis'
cmap = 'coolwarm'
fontsize = 12

#------------------------------------------------------------------------------
# I/O: GloSAT.prelim01_reqSD_alternativegrid-178101-201912.nc
#------------------------------------------------------------------------------

# load .nc file (netcdf4) into xarray

filename_nc = 'GloSAT.prelim01_reqSD_alternativegrid-178101-201912.nc'

ds = xr.open_dataset(filename_nc, decode_cf=True) 
lat = np.array(ds.latitude)
lon = np.array(ds.longitude)
time = np.array(ds.time)
ds.coords['month'] = ds.time.dt.month 
ds.coords['year'] = ds.time.dt.year
ds_global_timeseries = ds.groupby('year').mean() # global timeseries
ds_monthly_climatology = ds.groupby('month').mean(dim='time') # monthly climatology
ds_annual_mean = ds.groupby('year').mean(dim='time') # annual mean anomaly

# PLOT: monthly climaatology

monthstr = ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for i in range(len(ds_monthly_climatology['month'])):

    v = np.array(ds_monthly_climatology['temperature_anomaly'])[i,:,:]
    x, y = np.meshgrid(lon, lat)
    
    if use_minmax == True:    
        vmin = np.min(v)
        vmax = np.max(v)
    else:
        vmin = -2.0
        vmax = +2.0

    filestr = "temperature_anomaly_monthly_climatology" + "_" + str(i) + ".png"
    titlestr = 'Monthly climatology: ' + monthstr[i]
    colorbarstr = 'Anomaly (from 1961-1990)'

    fig  = plt.figure(figsize=(15,10))

    if projection == 'platecarree': p = ccrs.PlateCarree(central_longitude=0); threshold = 0
    if projection == 'mollweide': p = ccrs.Mollweide(central_longitude=0); threshold = 1e6
    if projection == 'robinson': p = ccrs.Robinson(central_longitude=0); threshold = 0
    if projection == 'equalearth': p = ccrs.EqualEarth(central_longitude=0); threshold = 0
    if projection == 'geostationary': p = ccrs.Geostationary(central_longitude=0); threshold = 0
    if projection == 'goodehomolosine': p = ccrs.InterruptedGoodeHomolosine(central_longitude=0); threshold = 0
    if projection == 'europp': p = ccrs.EuroPP(); threshold = 0
    if projection == 'northpolarstereo': p = ccrs.NorthPolarStereo(); threshold = 0
    if projection == 'southpolarstereo': p = ccrs.SouthPolarStereo(); threshold = 0
    if projection == 'lambertconformal': p = ccrs.LambertConformal(central_longitude=0); threshold = 0

    ax = plt.axes(projection=p)
    ax.stock_img()
    ax.coastlines()
    #ax.coastlines(resolution='50m')
    #ax.add_feature(cf.RIVERS.with_scale('50m'))
    #ax.add_feature(cf.BORDERS.with_scale('50m'))
    #ax.add_feature(cf.LAKES.with_scale('50m'))
    ax.gridlines()
        
    g = ccrs.Geodetic()
    trans = ax.projection.transform_points(g, x, y)
    x0 = trans[:,:,0]
    x1 = trans[:,:,1]
    
    if projection == 'platecarree':
        ax.set_extent([-180, 180, -90, 90], crs=p)    
        gl = ax.gridlines(crs=p, draw_labels=True, linewidth=1, color='black', alpha=0.5, linestyle='-')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-180,-120,-60,0,60,120,180])
        gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        for mask in (x0>threshold,x0<=threshold):            
            im = ax.pcolor(ma.masked_where(mask, x), ma.masked_where(mask, y), ma.masked_where(mask, v), vmin=vmin, vmax=vmax, transform=ax.projection, cmap=cmap)
    else:
        for mask in (x0>threshold,x0<=threshold):
            im = ax.pcolor(ma.masked_where(mask, x0), ma.masked_where(mask, x1), ma.masked_where(mask, v), vmin=vmin, vmax=vmax, transform=ax.projection, cmap=cmap) 
    im.set_clim(vmin,vmax)
    
    if use_horizontal_colorbar == True:
        cb = plt.colorbar(im, orientation="horizontal", shrink=0.5, extend='both')
        cb.set_label(colorbarstr, labelpad=25, fontsize=fontsize)
    else:
        cb = plt.colorbar(im, shrink=0.5, extend='both')
        cb.set_label(colorbarstr, rotation=270, labelpad=25, fontsize=fontsize)

    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(filestr)
    plt.close('all')
    
# PLOT: annual mean

for i in range(198,len(ds_annual_mean['year'])): # plot 1979-present (ERA5 range)

    v = np.array(ds_annual_mean['temperature_anomaly'])[i,:,:]    
    x, y = np.meshgrid(lon, lat)
    
    if use_minmax == True:    
        vmin = np.min(v)
        vmax = np.max(v)
    else:
        vmin = -2.0
        vmax = +2.0

    filestr = "temperature_anomaly_annual_mean" + "_" + str(i) + ".png"
    titlestr = 'Mean annual anomaly: ' + str(ds_annual_mean['year'][i].values)
    colorbarstr = 'Anomaly (from 1961-1990)'

    fig  = plt.figure(figsize=(15,10))

    if projection == 'platecarree': p = ccrs.PlateCarree(central_longitude=0); threshold = 0
    if projection == 'mollweide': p = ccrs.Mollweide(central_longitude=0); threshold = 1e6
    if projection == 'robinson': p = ccrs.Robinson(central_longitude=0); threshold = 0
    if projection == 'equalearth': p = ccrs.EqualEarth(central_longitude=0); threshold = 0
    if projection == 'geostationary': p = ccrs.Geostationary(central_longitude=0); threshold = 0
    if projection == 'goodehomolosine': p = ccrs.InterruptedGoodeHomolosine(central_longitude=0); threshold = 0
    if projection == 'europp': p = ccrs.EuroPP(); threshold = 0
    if projection == 'northpolarstereo': p = ccrs.NorthPolarStereo(); threshold = 0
    if projection == 'southpolarstereo': p = ccrs.SouthPolarStereo(); threshold = 0
    if projection == 'lambertconformal': p = ccrs.LambertConformal(central_longitude=0); threshold = 0

    ax = plt.axes(projection=p)
    ax.stock_img()
    ax.coastlines()
    #ax.coastlines(resolution='50m')
    #ax.add_feature(cf.RIVERS.with_scale('50m'))
    #ax.add_feature(cf.BORDERS.with_scale('50m'))
    #ax.add_feature(cf.LAKES.with_scale('50m'))
    ax.gridlines()
        
    g = ccrs.Geodetic()
    trans = ax.projection.transform_points(g, x, y)
    x0 = trans[:,:,0]
    x1 = trans[:,:,1]
    
    if projection == 'platecarree':
        ax.set_extent([-180, 180, -90, 90], crs=p)    
        gl = ax.gridlines(crs=p, draw_labels=True, linewidth=1, color='black', alpha=0.5, linestyle='-')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-180,-120,-60,0,60,120,180])
        gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        for mask in (x0>threshold,x0<=threshold):            
            im = ax.pcolor(ma.masked_where(mask, x), ma.masked_where(mask, y), ma.masked_where(mask, v), vmin=vmin, vmax=vmax, transform=ax.projection, cmap=cmap)
    else:
        for mask in (x0>threshold,x0<=threshold):
            im = ax.pcolor(ma.masked_where(mask, x0), ma.masked_where(mask, x1), ma.masked_where(mask, v), vmin=vmin, vmax=vmax, transform=ax.projection, cmap=cmap) 
    im.set_clim(vmin,vmax)
    
    if use_horizontal_colorbar == True:
        cb = plt.colorbar(im, orientation="horizontal", shrink=0.5, extend='both')
        cb.set_label(colorbarstr, labelpad=25, fontsize=fontsize)
    else:
        cb = plt.colorbar(im, shrink=0.5, extend='both')
        cb.set_label(colorbarstr, rotation=270, labelpad=25, fontsize=fontsize)

    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(filestr)
    plt.close('all')
    
#------------------------------------------------------------------------------
print('** END')