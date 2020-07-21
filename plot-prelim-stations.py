#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: plot-prelim-stations.py
#------------------------------------------------------------------------------
# Version 0.3
# 20 July, 2020
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
from io import StringIO
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
import seaborn as sns
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

fontsize = 20
fillval = -999
load_df = True
plot_temporal_coverage = True
plot_spatial_coverage = True
plot_station_timeseres = True
plot_monthly_climatology = True
plot_locations = True

#------------------------------------------------------------------------------
# METHODS
#------------------------------------------------------------------------------

def calc_median(counts,bins):
    """
    # -------------------------------
    # CALCULATE MEDIUM FROM HISTOGRAM
    # -------------------------------
    # M_estimated ~ L_m + [ ( N/2 - F_{m-1} ) /  f_m]  * c
    #
    # where,
    #
    # L_m =lower limit of the median bar
    # N = is the total number of observations
    # F_{m-1} = cumulative frequency (total number of observations) in all bars below the median bar
    # f_m = frequency of the median bar
    # c = median bar width
    """
    M = 0
    counts_cumsum = counts.cumsum()
    counts_half = counts_cumsum[-1]/2.0    
    for i in np.arange(0,bins.shape[0]-1):
        counts_l = counts_cumsum[i]
        counts_r = counts_cumsum[i+1]
        if (counts_half >= counts_l) & (counts_half < counts_r):
            c = bins[1]-bins[0]
            L_m = bins[i+1]
            F_m_minus_1 = counts_cumsum[i]
            f_m = counts[i+1]
            M = L_m + ( (counts_half - F_m_minus_1) / f_m ) * c            
    return M

def load_dataframe(filename_txt):
    
    #------------------------------------------------------------------------------
    # I/O: stat4.CRUTEM5.1prelim01.1721-2019.txt (text dump from CDF4)
    #------------------------------------------------------------------------------

    # load .txt file (comma separated) into pandas dataframe

    # filename_txt = 'stat4.CRUTEM5.1prelim01.1721-2019.txt'
    # GOOD DATA:
    # station header:    
    # [06True7250 464  -78 1538 BLATTEN, LOETSCHENTA SWITZERLAND   20012012  982001    9999]
    # station data:
    # [1921  -44  -71  -68  -46  -12   17   42   53   27  -20  -21  -40]
    # BAD DATA EXAMPLES:
    # ['939470-525-1691', '15', 'CAMPBELL', 'ISLAND', 'NEW', 'ZEALAND', '19412019', '411941', '0']
    # ['939870-440', '1766', '44', 'CHATHAM', 'I', 'WAITANGI', 'NEW', 'ZEALAND', '18782012', '411878', '93987']
    # ['999096-341-9999-9999', 'BREAKSEA', 'AUSTRALIA', '18971899', '101897', '0']

    yearlist = []
    monthlist = []
    stationinfo = []
    stationcode = []
    stationcountry = []
    stationlat = []
    stationlon = []
    with open (filename_txt, 'r', encoding="ISO-8859-1") as f:  
                    
        for line in f:   
            if len(line)>1: # ignore empty lines         
                if (len(line.strip().split())!=13) | (len(line.split()[0])>4):   
                    # when line is stationinfo extract stationid and stationname
                    info = line.strip().split()
                    if len(info[0]) == 6:
                        code = info[0]
                        country = line.strip().split()[-4]
                        lon = line.strip().split()[1]
                        lat = line.strip().split()[2]
                        if (len(lon) < 5) & (lon[1:].isdigit()):
                            lon = float(lon)
                        else:
                            lon = np.nan
                        if (len(lat) < 5) & (lat[1:].isdigit()):
                            lat = float(lat)
                        else:
                            lat = np.nan
                    else:
                        code = np.nan
                        country = np.nan
                else:           
                    # append monthly anomalies in [K] & station: lat, lon, id, name 
                    yearlist.append(np.array(line.strip().split()[0]).astype('int'))                                 
                    monthlist.append(np.array(line.strip().split()[1:]).astype('int'))                                 
                    stationinfo.append(info) # store for flagging
                    stationcode.append(code)
                    stationcountry.append(country)
                    stationlat.append(lat)
                    stationlon.append(lon)
            else:
                continue
    f.close

    # construct dataframe
    df = pd.DataFrame(columns=['year','1','2','3','4','5','6','7','8','9','10','11','12'])
    df['year'] = yearlist
    for j in range(1,12+1):
        df[df.columns[j]] = [ monthlist[i][j-1] for i in range(len(monthlist)) ]
    df.replace(fillval, np.nan, inplace=True) 
    df['stationinfo'] = stationinfo
    df['stationcode'] = stationcode
    df['stationcountry'] = stationcountry
    df['stationlat'] = stationlat
    df['stationlon'] = stationlon
                
    df.to_csv('df.csv')
    return df

#------------------------------------------------------------------------------
# LOAD DATAFRAME
#------------------------------------------------------------------------------

if load_df == True:

    df = pd.read_csv('df.csv')

else:    
    #------------------------------------------------------------------------------
    # I/O: stat4.CRUTEM5.1prelim01.1721-2019.txt (text dump from CDF4)
    #------------------------------------------------------------------------------

    # load .txt file (comma separated) into pandas dataframe

    filename_txt = 'stat4.CRUTEM5.1prelim01.1721-2019.txt'
    df = load_dataframe(filename_txt)

#------------------------------------------------------------------------------
# APPLY FILTERS
#------------------------------------------------------------------------------

year_min = df['year'].min()
year_max = df['year'].max()
year_start = year_min
year_end = 1900
lat_min = df['stationlat'].min()
lat_max = df['stationlat'].max()
lon_min = df['stationlon'].min()
lon_max = df['stationlon'].max()
lat_start = -900
lat_end = 900
lon_start = 0
lon_end = 3600
station_start=0
station_end=10


#------------------------------------------------------------------------------
# PLOTS
#------------------------------------------------------------------------------

if plot_temporal_coverage == True:
    
    # PLOT: histogram of yearly coverage
          
    nbins = year_end - year_start + 1
    bins = np.linspace(year_start, year_end, nbins) 
    counts, edges = np.histogram(df[df['year']<1901]['year'], nbins, range=[year_start,year_end+1], density=False)    
#    M = calc_median(counts,bins)    
    Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
    Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]
             
    figstr = 'crutem5-histogram-1900.png'
    titlestr = 'Histogram of temporal coverage for CRUTEM5 (to 1900): N=' + "{0:.0f}".format(np.sum(counts))

    fig, ax = plt.subplots(figsize=(15,10))          
#    plt.hist(df[df['year']<1901]['year'], bins=nbins, density=True, facecolor='grey', alpha=0.5, label='KDE')
    plt.fill_between(bins, counts, step="post", facecolor='lightgrey', alpha=0.5)
    plt.plot(bins, counts, drawstyle='steps-post', linewidth=2, color='black')    
#    plt.axvline(x=M, color='r')    
    plt.axvline(x=Q1, color='blue', label='Q1: ' + "{0:.0f}".format(Q1))   
    plt.axvline(x=Q2, color='red', label='Q2: ' + "{0:.0f}".format(Q2))    
    plt.axvline(x=Q3, color='blue', label='Q3: ' + "{0:.0f}".format(Q3))    
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='best', fontsize=fontsize)
    plt.xlabel("Year", fontsize=fontsize)
    plt.ylabel("Monthly values per year", fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close(fig)

    nbins = year_max - year_start + 1
    bins = np.linspace(year_start, year_max, nbins) 
    counts, edges = np.histogram(df['year'], nbins, range=[year_start,year_max+1], density=False)    
    Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
    Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]
             
    figstr = 'crutem5-histogram-full.png'
    titlestr = 'Histogram of temporal coverage for CRUTEM5 (full): N=' + "{0:.0f}".format(np.sum(counts)) 

    fig, ax = plt.subplots(figsize=(15,10))          
    plt.fill_between(bins, counts, step="post", facecolor='lightgrey', alpha=0.5)
    plt.plot(bins, counts, drawstyle='steps-post', linewidth=2, color='black')    
    plt.axvline(x=Q1, color='blue', label='Q1: ' + "{0:.0f}".format(Q1))    
    plt.axvline(x=Q2, color='red', label='Q2: ' + "{0:.0f}".format(Q2))    
    plt.axvline(x=Q3, color='blue', label='Q3: ' + "{0:.0f}".format(Q3))    
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='best', fontsize=fontsize)
    plt.xlabel("Year", fontsize=fontsize)
    plt.ylabel("Monthly values per year", fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close(fig)

if plot_spatial_coverage == True:
    
    # PLOT: histogram of latitudinal coverage: NB: scale is lat[-900,900] and lon[0,3600]

    nbins = int((lat_end - lat_start)/10) + 1
    bins = np.linspace(lat_start, lat_end, nbins) 
    counts, edges = np.histogram(df[(df['stationlat']>lat_start) & (df['stationlat']<lat_end)]['stationlat'], nbins, range=[lat_start,lat_max+1], density=False)    
    Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
    Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]

    figstr = 'crutem5-histogram-latlon-full.png'
    titlestr = 'Histogram of latitudinal coverage: CRUTEM5 (full)'

    fig, ax = plt.subplots(figsize=(15,10))          
    plt.fill_between(bins/10, counts, step="post", facecolor='lightgrey', alpha=0.5)
    plt.plot(bins/10, counts, drawstyle='steps-post', linewidth=2, color='black')    
    plt.axvline(x=Q1/10, color='blue', label='Q1: ' + "{0:.0f}".format(Q1/10))    
    plt.axvline(x=Q2/10, color='red', label='Q2: ' + "{0:.0f}".format(Q2/10))    
    plt.axvline(x=Q3/10, color='blue', label='Q3: ' + "{0:.0f}".format(Q3/10))    
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='best', fontsize=fontsize)
    plt.xlabel("Latitude, $\mathrm{\degree}N$", fontsize=fontsize)
    plt.ylabel("Monthly values per degree", fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close(fig)

    # PLOT: histogram of longitudinal coverage: NB: scale is lon[0,3600]

    nbins = int((lon_end - lon_start)/10) + 1
    bins = np.linspace(lon_start, lon_end, nbins) 
    counts, edges = np.histogram(df[(df['stationlon']>lon_start) & (df['stationlon']<lon_end)]['stationlon'], nbins, range=[lon_start,lon_max+1], density=False)    
    Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
    Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]

    figstr = 'crutem5-histogram-lon-full.png'
    titlestr = 'Histogram of longitudinal spatial coverage: CRUTEM5 (full)'

    fig, ax = plt.subplots(figsize=(15,10))          
    plt.fill_between(bins/10, counts, step="post", facecolor='lightgrey', alpha=0.5)
    plt.plot(bins/10, counts, drawstyle='steps-post', linewidth=2, color='black')    
    plt.axvline(x=Q1/10, color='blue', label='Q1: ' + "{0:.0f}".format(Q1/10))    
    plt.axvline(x=Q2/10, color='red', label='Q2: ' + "{0:.0f}".format(Q2/10))    
    plt.axvline(x=Q3/10, color='blue', label='Q3: ' + "{0:.0f}".format(Q3/10))    
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='best', fontsize=fontsize)
    plt.xlabel("Latitude, $\mathrm{\degree}W$", fontsize=fontsize)
    plt.ylabel("Monthly values per degree", fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close(fig)
          
if plot_station_timeseres == True:
    
    # PLOT: timeseries per station
        
    #for j in range(len(np.unique(stationcode))):
    for j in range(station_start,station_end):
    
        # form station timeseries
        da = df[df['stationcode']==df['stationcode'].unique()[j]].iloc[:,range(1,14)]
    #   da_melt = da.melt(id_vars='year').sort_values(by=['year']).reset_index()
        ts = []
        for i in range(len(da)):
            monthly = da.iloc[i,1:]/100.0
            ts = ts + monthly.to_list()
        ts = np.array(ts)            
        t = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts), freq='M')    
        ts_yearly = da.groupby(da['year']).mean().values/100.0
        t_yearly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='Y')    
        
    #    db = pd.DataFrame(columns=da.columns)
    #    db['year'] = [ i for i in range(da_melt['year'].iloc[0], da_melt['year'].iloc[-1]+1)]
    #    db_melt = db.melt(id_vars='year').sort_values(by=['year']).reset_index()
    #    del db_melt['index']
    #    db_melt.rename(columns={'year':'Year','variable':'Month','value':'Value'}, inplace = True)
    #    db_melt['Day'] = 15
    #    db_melt['Date'] = pd.to_datetime(db_melt[['Year','Month','Day']], format='%Y%m')      
    
        figstr = 'timeseries_' + str(int(df['stationcode'].unique()[j])) + '.png'
        titlestr = 'Monthly temperature anomaly timeseries: stationcode=' + str(int(df['stationcode'].unique()[j]))
              
        fig, ax = plt.subplots(figsize=(15,10))      
        plt.plot(t,ts, color='lightgrey')
#        plt.plot(ts_yearly, color='red')        
        plt.tick_params(labelsize=fontsize)
#        plt.legend(loc='best', fontsize=fontsize)
        plt.xlabel("Year", fontsize=fontsize)
        plt.ylabel("Monthly temperature anomaly, $\mathrm{\degree}C$", fontsize=fontsize)
        plt.title(titlestr, fontsize=fontsize)
        plt.savefig(figstr)
        plt.close(fig)

if plot_monthly_climatology == True:
    
    # PLOT: seasonal cycle per station
        
    #for j in range(len(np.unique(stationcode))):        
    for j in range(station_start,station_end):
    
        x = df[df['stationcode']==df['stationcode'].unique()[j]].iloc[:,1]    
        Y = df[df['stationcode']==df['stationcode'].unique()[j]].iloc[:,range(2,14)].T   
        
        figstr = 'seasonal-cycle_' + str(int(df['stationcode'].unique()[j])) + '.png'
        titlestr = 'Monthly temperature anomaly seasonal cycle: stationcode=' + str(int(df['stationcode'].unique()[j]))
    
        fig, ax = plt.subplots(figsize=(15,10))      
        plt.plot(np.arange(1,13),Y.iloc[:,0:len(Y.T)]/100.0, color='lightgrey', label=None)
        plt.plot(np.arange(1,13),Y.iloc[:,-1]/100.0, color='red', label=str(x.iloc[-1]))
        plt.legend(loc='best', fontsize=fontsize)
        plt.tick_params(labelsize=fontsize)   
        plt.xlabel("Month", fontsize=fontsize)
        plt.ylabel("Monthly temperature anomaly, $\mathrm{\degree}C$", fontsize=fontsize)        
        plt.title(titlestr, fontsize=fontsize)
        plt.savefig(figstr)
        plt.close(fig)

if plot_locations == True:
    
    # PLOT: quick plot of stations on world map

    x = -df['stationlon']/10
    y = df['stationlat']/10
    
    figstr = 'location_map.png'
    titlestr = 'Station locations'
     
    fig  = plt.figure(figsize=(15,10))
    p = ccrs.PlateCarree(central_longitude=0); threshold = 0
    ax = plt.axes(projection=p)
    ax.set_global()
#    ax.stock_img()
    ax.add_feature(cf.COASTLINE, edgecolor="tomato")
    ax.add_feature(cf.BORDERS, edgecolor="tomato")
#    ax.coastlines()
#    ax.gridlines()    
        
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
    plt.scatter(x=x, y=y, 
                color="dodgerblue", s=1, alpha=0.5,
                transform=ccrs.PlateCarree()) 
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close('all')
        
#------------------------------------------------------------------------------
print('** END')
