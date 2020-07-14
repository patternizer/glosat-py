#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot-prelim.py
#------------------------------------------------------------------------------
# Version 0.2
# 13 July, 2020
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
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.collections import PolyCollection
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
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
# METHODS: 
#------------------------------------------------------------------------------

# include mapping code:
# exec(open('plot_map.py').read())

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------
fontsize = 20

#------------------------------------------------------------------------------
# I/O: GloSAT.prelim01_reqSD_alternativegrid-178101-201912.timeseries.txt
#------------------------------------------------------------------------------

# load .txt file (comma separated) into pandas dataframe
filename_txt = 'GloSAT.prelim01_reqSD_alternativegrid-178101-201912.timeseries.txt'

headings = ['Year', 'Month', 'Global', 'NH', 'SH']
fillval = -9.99900
# Units: K anomalies from 1961-1990 reference period mean

df = pd.DataFrame(columns = headings)
datain = pd.read_csv(filename_txt, delim_whitespace=True, header=4)
for i in range(len(df.columns)):
    df[df.columns[i]] = datain.values[:,i]    

# convert year and month columns to integer
df['Day'] = 15
df[['Year','Month','Day']] = df[['Year','Month','Day']].applymap(np.int64) 

# convert year, month,day to datetime 
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

# replace fill value with NaN
df.replace(fillval, np.NaN, inplace=True) 

# save data munged version as csv
df.to_csv('data_munged.csv', sep=',', index=False, header=True, encoding='utf-8')

# test: load .csv file (netcdf4) into pandas dataframe
# pd.read_csv('data_munged.csv')

# compute annual means --> new data frame
da = pd.DataFrame()
da['Year'] = df.groupby(df['Year'].dropna()).mean()['Year']
da['Global_Annual_Mean'] = df.groupby(df['Year'].dropna()).mean()['Global'].values
da['NH_Annual_Mean'] = df.groupby(df['Year'].dropna()).mean()['NH'].values
da['SH_Annual_Mean'] = df.groupby(df['Year'].dropna()).mean()['SH'].values

# PLOT: Monthly anomaly

fig, ax = plt.subplots(figsize=(15,10))
plt.plot(df['Date'], df['NH'], label='NH', color='pink')
plt.plot(df['Date'], df['SH'], label='SH', color='lightblue')
plt.plot(df['Date'], df['Global'], label='Global', color='black', linewidth=0.5)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylabel('Anomaly (from 1961-1990)', fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.title(filename_txt, fontsize=fontsize)
plt.savefig(filename_txt + '_anomaly.png')
plt.close(fig)

# PLOT: Mean annual anomaly

fig, ax = plt.subplots(figsize=(15,10))
plt.bar(da['Year'], da['NH_Annual_Mean'], label='NH', color='pink')
plt.bar(da['Year'], da['SH_Annual_Mean'], label='SH', color='lightblue')
plt.plot(da['Year'], da['Global_Annual_Mean'], label='Global', color='black', linewidth=2)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylabel('Mean annual anomaly (from 1961-1990)', fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.title(filename_txt, fontsize=fontsize)
plt.savefig(filename_txt + '_anomly_annual_mean.png')
plt.close(fig)

#------------------------------------------------------------------------------
# climate stripes 
#------------------------------------------------------------------------------

# da = da[da['Year']>1978] # ERA-5

# Calculate 1961-1990 mean (NB: Ed's Climate Stripes: 1971-2000)
#mu = df[(df['Year']>1960) & (df['Year']<1991)]['Global'].mean()
mu = da[(da['Year']>1960) & (da['Year']<1991)]['Global_Annual_Mean'].mean()

# Compute standard deviation of the annual average temperatures between 1901-2000: color range +/- 2.6 standard deviations 
sigma = da[(da['Year']>1900) & (da['Year']<2001)]['Global_Annual_Mean'].std()

x = da[da['Year']>1900]['Year']
y = da[da['Year']>1900]['Global_Annual_Mean']
cmap = plt.cm.get_cmap('coolwarm')

maxval = +2.6 * sigma
minval = -2.6 * sigma

# PLOT: mean annual anomaly (1900-2019) as climate stripe bars

fig, ax = plt.subplots(figsize=(15,10))
plt.bar(x, y-mu, color=cm.coolwarm((y-mu)/maxval+0.5))
ax.axis('off')
sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(min(y-mu),max(y-mu)))
sm.set_array([])
#cbar = plt.colorbar(sm, shrink=0.5, orientation='horizontal')
#cbar.set_label('Anomaly (from 1961-1990)', rotation=0, labelpad=25, fontsize=fontsize)
cbar = plt.colorbar(sm, shrink=0.5)
cbar.set_label('Anomaly (from 1961-1990)', rotation=270, labelpad=25, fontsize=fontsize)
plt.title('Mean annual anomaly: Global', fontsize=fontsize)
plt.savefig(filename_txt + '_anomaly-bars.png')
plt.close(fig)

# PLOT: mean annual anomaly (1900-2019) as climate stripes

fig, ax = plt.subplots(figsize=(15,10))
z = (y-mu)*0.0+1.0
colors = cmap((y-mu)/maxval+0.5)
ax.bar(x, z, color=colors, width=1.0)
ax.axis('off')
sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(min(y-mu),max(y-mu)))
sm.set_array([])
#cbar = plt.colorbar(sm, shrink=0.5, orientation='horizontal')
#cbar.set_label('Mean annual anomaly (from 1961-1990)', rotation=0, labelpad=25, fontsize=fontsize)
cbar = plt.colorbar(sm, shrink=0.5)
cbar.set_label('Anomaly (from 1961-1990)', rotation=270, labelpad=25, fontsize=fontsize)
plt.title('Mean annual anomaly: Global', fontsize=fontsize)
plt.savefig(filename_txt + '_anomaly-stripes.png')
plt.close(fig)

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

projection = 'equalearth'
#cmap = 'viridis'
cmap = 'coolwarm'
fontsize = 12
monthstr = ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

#for i in range(len(ds_monthly_climatology['month'])):
for i in range(200,len(ds_annual_mean['year'])):

#    var = np.array(ds_monthly_climatology['temperature_anomaly'])[i,:,:]
    var = np.array(ds_annual_mean['temperature_anomaly'])[i,:,:]
#    vmin = np.min(var)
#    vmax = np.max(var)
    vmin = -2.0
    vmax = +2.0
    x, y = np.meshgrid(lon, lat)
    z=var

#    filestr = "temperature_anomaly_monthly_climatology" + "_" + str(i) + ".png"
#    titlestr = 'Monthly climatology: ' + monthstr[i]
    filestr = "temperature_anomaly_annual_mean" + "_" + str(i) + ".png"
    titlestr = 'Mean annual anomaly: ' + str(ds_annual_mean['year'][i])

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
#            im = ax.pcolor(ma.masked_where(mask, x), ma.masked_where(mask, y), ma.masked_where(mask, z), transform=ax.projection, cmap=cmap)
            im = ax.pcolor(ma.masked_where(mask, x), ma.masked_where(mask, y), ma.masked_where(mask, z), vmin=vmin, vmax=vmax, transform=ax.projection, cmap=cmap)
    else:
        for mask in (x0>threshold,x0<=threshold):
#            im = ax.pcolor(ma.masked_where(mask, x0), ma.masked_where(mask, x1), ma.masked_where(mask, z), transform=ax.projection, cmap=cmap)
            im = ax.pcolor(ma.masked_where(mask, x0), ma.masked_where(mask, x1), ma.masked_where(mask, z), vmin=vmin, vmax=vmax, transform=ax.projection, cmap=cmap) 
    im.set_clim(vmin,vmax)
    cb = plt.colorbar(im, orientation="horizontal", shrink=0.5, extend='both')
    cb.set_label('Anomaly (from 1961-1990)', labelpad=25, fontsize=fontsize)
#    cb = plt.colorbar(im, shrink=0.5, extend='both')
#    cb.set_label('Anomaly (from 1961-1990)', rotation=270, labelpad=25, fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(filestr)
    plt.close('all')
    
#------------------------------------------------------------------------------
print('** END')