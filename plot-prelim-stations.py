#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: plot-prelim-stations.py
#------------------------------------------------------------------------------
# Version 0.11
# 15 August, 2020
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
import numpy as np
import numpy.ma as ma
from mod import Mod
import itertools
import pandas as pd
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
import cmocean
# Mapping libraries:
import cartopy
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cf
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# OS libraries:
import os
import os.path
from pathlib import Path
import sys
import subprocess
from subprocess import Popen
import time

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

fontsize = 20
fillval = -999
lat_start = -90;  lat_end = 90
lon_start = -180; lon_end = 180
station_start=0;  station_end=10

load_df = False
plot_data_coverage = True
plot_temporal_coverage = False
plot_spatial_coverage = False
plot_station_timeseres = False
plot_monthly_climatology = False
plot_locations = False
plot_delta_cc = False
delta_cc_20C = True    
normalize_timeseries = False

#------------------------------------------------------------------------------
# METHODS
#------------------------------------------------------------------------------

def load_dataframe(filename_txt):
    
    #------------------------------------------------------------------------------
    # I/O: stat4.CRUTEM5.1prelim01.1721-2019.txt (text dump from CDF4)
    #------------------------------------------------------------------------------

    # load .txt file (comma separated) into pandas dataframe
    
    filename_txt = 'stat4.CRUTEM5.1prelim01.1721-2019.txt'
    
    yearlist = []
    monthlist = []
    stationheader = []

    stationcode = []
    stationlat = []
    stationlon = []
    stationelevation = []
    stationname = []
    stationcountry = []
    stationfirstyear = []
    stationlastyear = []
    stationsource = []
    stationfirstreliable = []
    stationfirstreliable = []
    stationcruindex = []
    stationgridcell = []

    with open (filename_txt, 'r', encoding="ISO-8859-1") as f:  
                    
        for line in f:   
            if len(line)>1: # ignore empty lines         
                if (len(line.strip().split())!=13) | (len(line.split()[0])>4):   
                    
                    # when line is station header extract info
                    #
                    # Station Header File:
                    #
                    #    (ch. 1-7) World Meteorological Organization (WMO) Station Number with a single additional character making a field of 6 integers. WMO numbers comprise a 5 digit sub-field, where the first two digits are the country code and the next three digits designated by the National Meteorological Service (NMS). Some country codes are not used. If the additional sixth digit is a zero, then the WMO number is or was an official WMO number. If the sixth digit is not zero then the station does not have an official WMO number and an alternative number has been assigned by CRU. Two examples are given below. Many additional stations are grouped beginning 99****. Station numbers in the blocks 72**** to 75**** are additional stations in the United States.
                    #    (ch. 8-11) Station latitude in degrees and tenths (-999 is missing), with negative values in the Southern Hemisphere
                    #    (ch. 12-16) Station longitude in degrees and tenths (-1999 is missing), with negative values in the Eastern Hemisphere (NB this is opposite to the more usual convention)
                    #    (ch. 18-21) Station Elevation in metres (-999 is missing)
                    #    (ch. 23-42) Station Name
                    #    (ch. 44-56) Country
                    #    (ch. 58-61) First year of monthly temperature data
                    #    (ch. 62-65) Last year of monthly temperature data
                    #    (ch. 68-69) Data Source (see below)
                    #    (ch. 70-73) First reliable year (generally the same as the first year)
                    #    (ch. 75-78) Unique index number (internal use)
                    #    (ch. 80-83) Index into the 5° x 5° gridcells (internal use) 
                        
                    code = line[0:6]                    
                    lat = line[6:10]
                    lon = line[10:15]
                    elevation = line[17:20]
                    name = line[21:41]
                    country = line[42:55]
                    firstyear = line[56:60]
                    lastyear = line[60:64]
                    source = line[64:68]
                    firstreliable = line[68:72]
                    cruindex = line[72:77]
                    gridcell = line[77:80]    
                    
                    header = line
                            
                else:           
                    yearlist.append(int(line.strip().split()[0]))                                 
                    monthlist.append(np.array(line.strip().split()[1:]).astype('int'))                                 
                    stationheader.append(header)

                    stationcode.append(code)
                    stationlat.append(lat)
                    stationlon.append(lon)
                    stationelevation.append(elevation)
                    stationname.append(name)
                    stationcountry.append(country)
                    stationfirstyear.append(firstyear)
                    stationlastyear.append(lastyear)
                    stationsource.append(source)
                    stationfirstreliable.append(firstreliable)
                    stationcruindex.append(cruindex)
                    stationgridcell.append(gridcell)
            else:
                continue
    f.close

    # construct dataframe
    
    df = pd.DataFrame(columns=['year','1','2','3','4','5','6','7','8','9','10','11','12'])
    df['year'] = yearlist

    for j in range(1,13):
        df[df.columns[j]] = [ monthlist[i][j-1] for i in range(len(monthlist)) ]

    df['stationcode'] = stationcode    
    df['stationlat'] = stationlat
    df['stationlon'] = stationlon
    df['stationelevation'] = stationelevation
    df['stationname'] = stationname
    df['stationcountry'] = stationcountry
    df['stationfirstyear'] = stationfirstyear
    df['stationlastyear'] = stationlastyear
    df['stationsource'] = stationsource
    df['stationfirstreliable'] = stationfirstreliable
    df['stationcruindex'] = stationcruindex
    df['stationgridcell'] = stationgridcell

    # trim strings
    
    df['stationcode'] = [ str(df['stationcode'][i]).strip() for i in range(len(df)) ] 
    df['stationname'] = [ str(df['stationname'][i]).strip() for i in range(len(df)) ] 
    df['stationcountry'] = [ str(df['stationcountry'][i]).strip() for i in range(len(df)) ] 
    df['stationcruindex'] = [ str(df['stationcruindex'][i]).strip() for i in range(len(df)) ] 
    df['stationgridcell'] = [ str(df['stationgridcell'][i]).strip() for i in range(len(df)) ] 

    # convert numeric types to int

    df['stationlat'] = df['stationlat'].astype('int')
    df['stationlon'] = df['stationlon'].astype('int')
    df['stationelevation'] = df['stationelevation'].astype('int')
    df['stationfirstyear'] = df['stationfirstyear'].astype('int')
    df['stationlastyear'] = df['stationlastyear'].astype('int')    
    df['stationsource'] = df['stationsource'].astype('int')    
    df['stationfirstreliable'] = df['stationfirstreliable'].astype('int')
#    df['stationcruindex'] = df['stationcruindex'].astype('int')
#    df['stationgridcell'] = df['stationgridcell'].astype('int') 

    # bad data handling

#    df['stationheader'] = stationheader          
#    nchar = [ len(df['stationheader'][i]) for i in range(len(df)) ]      
#    nwords = [ len(df['stationheader'][i].split()) for i in range(len(df)) ]      

#    for i in range(len(df)):        
#        if str(df['stationcruindex'][i])[1:].isdigit() == False:
#            df['stationcruindex'][i] = np.nan
#        else:
#            continue
 
    # replace fillvalues

    df['year'].replace(-999, np.nan, inplace=True) 
 
    for j in range(1,13):
    
        df[df.columns[j]].replace(-999, np.nan, inplace=True)

    df['stationlat'].replace(-999, np.nan, inplace=True) 
    df['stationlon'].replace(-9999, np.nan, inplace=True) 
    df['stationelevation'].replace(-999, np.nan, inplace=True) 
    df['stationfirstyear'].replace(-999, np.nan, inplace=True) 
    df['stationlastyear'].replace(-999, np.nan, inplace=True) 
    df['stationsource'].replace(-999, np.nan, inplace=True) 
    df['stationfirstreliable'].replace(-999, np.nan, inplace=True) 

    df.to_csv('df.csv')
    
    return df

def plot_stations(lon,lat,mapfigstr,maptitlestr):
     
    #------------------------------------------------------------------------------
    # PLOT (lon,lat) ON PLATECARREE MAP
    #------------------------------------------------------------------------------

    fig  = plt.figure(figsize=(15,10))
    p = ccrs.PlateCarree(central_longitude=0); threshold = 0
    ax = plt.axes(projection=p)
    ax.set_global()
    ax.add_feature(cf.COASTLINE, edgecolor="lightblue")
    ax.add_feature(cf.BORDERS, edgecolor="lightblue")        
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
    plt.scatter(x=lon, y=lat, color="maroon", s=1, alpha=0.5, transform=ccrs.PlateCarree()) 
    plt.title(maptitlestr, fontsize=fontsize)
    plt.savefig(mapfigstr)
    plt.close('all')

#------------------------------------------------------------------------------
# LOAD DATAFRAME
#------------------------------------------------------------------------------

if load_df == True:

    #------------------------------------------------------------------------------
    # EXTRACT TARBALL IF df.csv IS COMPRESSED:
    #------------------------------------------------------------------------------

    filename = Path("df.csv")
    if not filename.is_file():
        print('Uncompressing df.csv from tarball ...')
        #filename = "df.tar.gz"
        #subprocess.Popen(['tar', '-xzvf', filename]) # = tar -xzvf df.tar.gz
        filename = "df.tar.bz2"
        subprocess.Popen(['tar', '-xjvf', filename])  # = tar -xjvf df.tar.bz2
        time.sleep(5) # pause 5 seconds to give tar extract time to complete prior to attempting pandas read_csv

    df = pd.read_csv('df.csv', index_col=0)

else:    
    
    #------------------------------------------------------------------------------
    # I/O: stat4.CRUTEM5.1prelim01.1721-2019.txt (text dump from CDF4)
    #------------------------------------------------------------------------------

    # load .txt file (line by line) into pandas dataframe

    filename_txt = 'stat4.CRUTEM5.1prelim01.1721-2019.txt'
    df = load_dataframe(filename_txt)

    #------------------------------------------------------------------------------
    # APPLY SCALE FACTORS
    #------------------------------------------------------------------------------
    
    df = pd.read_csv('df.csv', index_col=0)
    df['stationlat'] = df['stationlat']/10.0
    df['stationlon'] = df['stationlon']/10.0
    for j in range(1,13):
        df[df.columns[j]] = df[df.columns[j]]/100.0

    #------------------------------------------------------------------------------
    # CONVERT LONGITUDE FROM +W TO +E
    #------------------------------------------------------------------------------
    df['stationlon'] = -df['stationlon']        

    #------------------------------------------------------------------------------
    # SAVE DATAFRAME
    #------------------------------------------------------------------------------
    df.to_csv('df.csv')

    #------------------------------------------------------------------------------
    # NORMALIZE TIMESERIES
    #------------------------------------------------------------------------------
    
    if normalize_timeseries == True:
    
        # Adjust station timeseries using normalisation for eacg month
        
        df_norm = df.copy()
#        df_norm['tmean']=np.nan
#        df_norm['tsd']=np.nan
#        for i in range(len(df_norm)): 
#            if df_norm.iloc[i,1:13].isnull().sum() > 0:                
#                df_norm['tmean'][i] = np.nan
#                df_norm['tsd'][i] = np.nan
#            else:                
#                tmean = np.nanmean(np.array(df_norm.iloc[i,1:13]).ravel())
#                tsd = np.nanstd(np.array(df_norm.iloc[i,1:13]).ravel())
#                df_norm['tmean'][i] = tmean
#                df_norm['tsd'][i] = tsd
              
        for i in range(len(df_norm['stationcode'].unique())):
            
            da = df_norm[df_norm['stationcode']==df_norm['stationcode'].unique()[i]]
            for j in range(1,13):
                df_norm.loc[da.index.tolist(), str(j)] = (da[str(j)]-da[str(j)].dropna().mean())/da[str(j)].dropna().std()

        df_norm.to_csv('df_norm.csv')

#------------------------------------------------------------------------------
# PLOT: station coverage - thanks to Kate-Willett (UKMO) for spec. humidity code:
# https://github.com/Kate-Willett
#------------------------------------------------------------------------------

if plot_data_coverage == True:

#   plot timeseries for each month of year:
    
    fig = plt.figure(1,figsize=(15,10))
#    plt.plot(df.groupby('year').mean().iloc[:,0:12])
    for i in range(0,12):
        plt.plot(df.groupby('year').mean().iloc[:,i], label='Month=' + str(i+1))
    plt.legend(fontsize=fontsize/2)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)    
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel('Annual mean anomaly', fontsize=fontsize)
    plt.savefig('monthly-anomaly-timeseries.png')
    plt.close(fig)
                            
    # Contruct palette of colours over station ID

    # WMO ID blocks:
        
    #      0-199999 = Europe
    # 200000-379999 = Russia and Eastern Europe
    # 380000-439999 = Central Asia, Middle East, India/Pakistan
    # 440000-599850 = East Asia, China and Taiwan
    # 600000-689999 = Africa
    # 690000-749999 = USA, Canada
    # 760000-819999 = Central America
    # 820000-879999 = South America
    # 880000-899999 = Antarctica
    # 911000-919999 = Pacific Islands (inc. Hawaii)
    # 930000-949999 = Australasia (in. NZ)
    # 960000-988999 = Indonesia/Phillippines/Borneo    
    
    colourlist=list([
        'DarkRed',
        'Crimson',
        'OrangeRed',
        'DarkOrange',
        'Gold',
        'DarkGreen',
        'OliveDrab',
        'MediumBlue',
        'DeepSkyBlue',
	    'MidnightBlue',
        'MediumSlateBlue',
        'DarkViolet'])
    labellist=list([
        ['Europe',0,199999],        
        ['Russia/Eastern Europe',200000,379999],
        ['Central and southern Asia/Middle East',380000,439999],
        ['East Asia',440000,599999],
	    ['Africa',600000,689999],
	    ['North America',690000,749999],
	    ['Central America',750000,799999],
	    ['South America',800000,879999],
	    ['Antarctica',880000,899999],
	    ['Pacific Islands',900000,919999],
	    ['Australasia',920000,949999],
	    ['Indonesia/Philippines/Borneo',950000,999999]])
    stationlist = df['stationcode'].unique()    

    stationcolours = []
    for i in range(len(labellist)):        
        labelcount = ((stationlist>=labellist[i][1]) & (stationlist<=labellist[i][2])).sum()        
        stationcolours.append(np.tile(colourlist[i],labelcount))
    stationcolours = list(itertools.chain.from_iterable(stationcolours))
                             
    # Contruct timeseries of yearly station count

    station_yearly_count = df.groupby('year')['stationcode'].count()
    t_station_yearly_count = pd.date_range(start=str(df['year'].min()), periods=len(global_yearly_mean), freq='A')

    # Contruct timeseries of yearly mean and s.d.

    global_yearly_mean = []
    global_yearly_std = []
    for i in range(df['year'].min(),df['year'].max()+1):                    
        yearly_mean = np.nanmean(np.array(df[df['year']==i].iloc[:,range(1,13)]).ravel())                                     
        yearly_std = np.nanstd(np.array(df[df['year']==i].iloc[:,range(1,13)]).ravel())                                                 
        global_yearly_mean.append(yearly_mean)
        global_yearly_std.append(yearly_std)
    t_yearly = pd.date_range(start=str(df['year'].min()), periods=len(global_yearly_mean), freq='A')

    # Contruct timeseries of monthly mean and s.d.

    global_monthly_mean = []
    global_monthly_std = []
    for i in range(df['year'].min(),df['year'].max()+1):            
        for j in range(1,13):
            monthly_mean = np.nanmean(df[df['year']==i][str(j)])                                          
            monthly_std = np.nanstd(df[df['year']==i][str(j)])                                          
            global_monthly_mean.append(monthly_mean)
            global_monthly_std.append(monthly_std)
    t_monthly = pd.date_range(start=str(df['year'].min()), periods=len(global_monthly_mean), freq='M')

    # PLOT: timeseries of monthly obs per station

#    figstr = 'crutem5-gaps-plus-mean-anomaly.png'
#    titlestr = 'CRUTEM5: Data Coverage & Annual temperature anomaly'    
    figstr = 'crutem5-gaps-plus-yearly-station-count.png'
    titlestr = 'CRUTEM5: Data Coverage & Annual station count'    
    xstr = 'Year'
    y1str = 'Station ID'
#    y2str = 'Annual mean'
    y2str = 'Annual station count'
            
    fig = plt.figure(1,figsize=(15,10))
    plt.clf()	# needs to be called after figure!!! (create the figure, then clear the plot space)
    ax1=plt.subplot(1,1,1)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
        
    for i in range(len(df['stationcode'].unique())):            
            
        da = df[df['stationcode']==df['stationcode'].unique()[i]].iloc[:,range(0,13)]
        ts = np.array([ da.iloc[i,1:].to_list() for i in range(len(da)) ]).ravel()            
        t = pd.date_range(start=str(da['year'].iloc[0]), periods=len(da)*12, freq='M')
        mask = np.isfinite(ts)
        x = t[mask]
        y = np.ones(len(ts))[mask]*df['stationcode'].unique()[i]
            
        ax1.plot(x, y, color=stationcolours[i], linewidth=1)
    
    ax1.set_title(titlestr,size=fontsize)
    ax1.set_xlabel(xstr,size=fontsize)
    ax1.set_ylabel(y1str,size=fontsize)    
    ax2=ax1.twinx()
    ax2.set_ylabel(y2str,size=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)
    
#    ax2.plot(t_yearly, global_yearly_mean, 'black', linewidth=2)
    ax2.plot(t_station_yearly_count, station_yearly_count, 'black', linewidth=2)

    plt.savefig(figstr)
    plt.close(fig)
                    
#------------------------------------------------------------------------------
# PLOT: delta CC +/- 30 years around median date value
#------------------------------------------------------------------------------
            
if plot_delta_cc == True:
        
    #------------------------------------------------------------------------------
    # EXTRACT TARBALL IF df_norm.csv IS COMPRESSED:
    #------------------------------------------------------------------------------

    filename = Path("df_norm.csv")
    if not filename.is_file():
        print('Uncompressing df_norm.csv from tarball ...')
        #filename = "df_norm.tar.gz"
        #subprocess.Popen(['tar', '-xzvf', filename]) # = tar -xzvf df_norm.tar.gz
        filename = "df_norm.tar.bz2"
        subprocess.Popen(['tar', '-xjvf', filename])  # = tar -xjvf df_norm.tar.bz2
        time.sleep(5) # pause 5 seconds to give tar extract time to complete prior to attempting pandas read_csv

    df = pd.read_csv('df_norm.csv', index_col=0)
        
    def plot_hist_array(diff, figstr, titlestr):

        deltamin = -2
        deltamax = 2
        deltastep = 0.1
        
        def plot_hist(diff,i,row,col):
            
            nbins = int((deltamax-deltamin)/deltastep) + 1
            bins = np.linspace(deltamin, deltamax, nbins) 
            counts, edges = np.histogram(diff[str(i)], nbins, range=[deltamin,deltamax], density=False)    
            Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
            Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
            Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]
            axs[row,col].fill_between(bins, counts, step="pre", facecolor='lightgrey', alpha=1.0)
#            axs[row,col].plot(bins, counts, drawstyle='steps', linewidth=0.5, color='black')    
            axs[row,col].axvline(x=Q1, color='blue', label='Q1: ' + "{0:.1f}".format(Q1))   
            axs[row,col].axvline(x=Q2, color='red', label='Q2: ' + "{0:.1f}".format(Q2))    
            axs[row,col].axvline(x=Q3, color='blue', label='Q3: ' + "{0:.1f}".format(Q3))    
            axs[row,col].set_title('Month=' + "{0:.0f}".format(i), fontsize=15)
        
        fig, axs = plt.subplots(4,3,figsize=(15,10))
        plot_hist(diff,1,0,0)
        plot_hist(diff,2,0,1)
        plot_hist(diff,3,0,2)
        plot_hist(diff,4,1,0)
        plot_hist(diff,5,1,1)
        plot_hist(diff,6,1,2)
        plot_hist(diff,7,2,0)
        plot_hist(diff,8,2,1)
        plot_hist(diff,9,2,2)
        plot_hist(diff,10,3,0)
        plot_hist(diff,11,3,1)
        plot_hist(diff,12,3,2)
        for ax in axs.flat:
#            ax.set_xlabel("$\mathrm{\Delta}$(anomaly), $\mathrm{\degree}$C", fontsize=15)
#            ax.set_ylabel("Counts / " + str(deltastep) + "$\mathrm{\degree}$C", fontsize=15)
            ax.set_xlabel("$\mathrm{\Delta}$(normalised anomaly)", fontsize=15)
            ax.set_ylabel("Counts / " + str(deltastep), fontsize=15)
        for ax in axs.flat:
            ax.label_outer()
            ax.legend(loc='best', fontsize=8)   
        
        fig.suptitle(titlestr, fontsize=fontsize)    
        plt.savefig(figstr)
        plt.close(fig)

    # Year range switch:

#    nbins = 1900 - df['year'].min() + 1
#    bins = np.linspace(df['year'].min(), 1900, nbins) 
#    counts, edges = np.histogram(df['year'], nbins, range=[df['year'].min(),1900], density=False)        
#    year_Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
#    year_lo = year_Q2-30
#    year_hi = year_Q2+30
    
    if delta_cc_20C:        
           year_range_str = '1959_1988_1989_2018'
           year_range = '1989-2018 minus 1959-1988'
           year_lo = 1959; year_Q2 = 1988; year_hi = 2018
    else:
        year_range_str = '1856_1885_1886_1915'
        year_range = '1886-1915 minus 1856-1885'
        year_lo = 1856; year_Q2 = 1885; year_hi = 1915

    # Calculate delta CC for globe, regions and latitudinal bands
    
    # NB: There are also country entries in the form e.g.:    
    # 'USA---------'
    # 'AUSTRALIA---'
        
    delta_lo_60N90N = df[(df['year']>=year_lo) & (df['year']<=year_Q2) & (df['stationlat']>60) & (df['stationlat']<=90)].groupby(['stationcode']).mean()
    delta_hi_60N90N = df[(df['year']>year_Q2) & (df['year']<=year_hi) & (df['stationlat']>60) & (df['stationlat']<=90)].groupby(['stationcode']).mean()
    diff = delta_hi_60N90N - delta_lo_60N90N        
    Nstations = len((delta_lo_60N90N.append(delta_hi_60N90N)).iloc[:,1:13].index.unique())        
    Nmonths = np.isfinite(delta_lo_60N90N.append(delta_hi_60N90N)).iloc[:,1:13].sum().sum()
    figstr = 'delta_cc_' + year_range_str + '_60N90N.png'
    titlestr = 'Change in normalised anomaly: ' + year_range + '\n 60<lat≤90°N' + ': N(stations)=' + "{0:.0f}".format(Nstations) + ',' + ' N(months)=' + "{0:.0f}".format(Nmonths)
    lon = (delta_lo_60N90N.append(delta_hi_60N90N))['stationlon']
    lat = (delta_lo_60N90N.append(delta_hi_60N90N))['stationlat']
    mapfigstr = 'map_' + figstr 
    maptitlestr = titlestr[57:]
    plot_stations(lon,lat,mapfigstr,maptitlestr)
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_30N60N = df[(df['year']>=year_lo) & (df['year']<=year_Q2) & (df['stationlat']>30) & (df['stationlat']<=60)].groupby(['stationcode']).mean()
    delta_hi_30N60N = df[(df['year']>year_Q2) & (df['year']<=year_hi) & (df['stationlat']>30) & (df['stationlat']<=60)].groupby(['stationcode']).mean()
    diff = delta_hi_30N60N - delta_lo_30N60N    
    Nstations = len((delta_lo_30N60N.append(delta_hi_30N60N)).iloc[:,1:13].index.unique())        
    Nmonths = np.isfinite(delta_lo_30N60N.append(delta_hi_30N60N)).iloc[:,1:13].sum().sum()
    figstr = 'delta_cc_' + year_range_str + '_30N60N.png'
    titlestr = 'Change in normalised anomaly: ' + year_range + '\n 30<lat≤60°N' + ': N(stations)=' + "{0:.0f}".format(Nstations) + ',' + ' N(months)=' + "{0:.0f}".format(Nmonths)
    lon = (delta_lo_30N60N.append(delta_hi_30N60N))['stationlon']
    lat = (delta_lo_30N60N.append(delta_hi_30N60N))['stationlat']
    mapfigstr = 'map_' + figstr 
    maptitlestr = titlestr[57:]
    plot_stations(lon,lat,mapfigstr,maptitlestr)
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_00N30N = df[(df['year']>=year_lo) & (df['year']<=year_Q2) & (df['stationlat']>0) & (df['stationlat']<=30)].groupby(['stationcode']).mean()
    delta_hi_00N30N = df[(df['year']>year_Q2) & (df['year']<=year_hi) & (df['stationlat']>0) & (df['stationlat']<=30)].groupby(['stationcode']).mean()
    diff = delta_hi_00N30N - delta_lo_00N30N    
    Nstations = len((delta_lo_00N30N.append(delta_hi_00N30N)).iloc[:,1:13].index.unique())        
    Nmonths = np.isfinite(delta_lo_00N30N.append(delta_hi_00N30N)).iloc[:,1:13].sum().sum()
    figstr = 'delta_cc_' + year_range_str + '_00N30N.png'
    titlestr = 'Change in normalised anomaly: ' + year_range + '\n 0<lat≤30°N' + ': N(stations)=' + "{0:.0f}".format(Nstations) + ',' + ' N(months)=' + "{0:.0f}".format(Nmonths)
    lon = (delta_lo_00N30N.append(delta_hi_00N30N))['stationlon']
    lat = (delta_lo_00N30N.append(delta_hi_00N30N))['stationlat']
    mapfigstr = 'map_' + figstr 
    maptitlestr = titlestr[57:]
    plot_stations(lon,lat,mapfigstr,maptitlestr)
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_00S30S = df[(df['year']>=year_lo) & (df['year']<=year_Q2) & (df['stationlat']<=0) & (df['stationlat']>=-30)].groupby(['stationcode']).mean()
    delta_hi_00S30S = df[(df['year']>year_Q2) & (df['year']<=year_hi) & (df['stationlat']<=0) & (df['stationlat']>=-30)].groupby(['stationcode']).mean()
    diff = delta_hi_00S30S - delta_lo_00S30S    
    Nstations = len((delta_lo_00S30S.append(delta_hi_00S30S)).iloc[:,1:13].index.unique())        
    Nmonths = np.isfinite(delta_lo_00S30S.append(delta_hi_00S30S)).iloc[:,1:13].sum().sum()
    figstr = 'delta_cc_' + year_range_str + '_00S30S.png'
    titlestr = 'Change in normalised anomaly: ' + year_range + '\n 0≤lat≤30°S' + ': N(stations)=' + "{0:.0f}".format(Nstations) + ',' + ' N(months)=' + "{0:.0f}".format(Nmonths)
    lon = (delta_lo_00S30S.append(delta_hi_00S30S))['stationlon']
    lat = (delta_lo_00S30S.append(delta_hi_00S30S))['stationlat']
    mapfigstr = 'map_' + figstr 
    maptitlestr = titlestr[57:]
    plot_stations(lon,lat,mapfigstr,maptitlestr)
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_30S60S = df[(df['year']>=year_lo) & (df['year']<=year_Q2) & (df['stationlat']<-30) & (df['stationlat']>=-60)].groupby(['stationcode']).mean()
    delta_hi_30S60S = df[(df['year']>year_Q2) & (df['year']<=year_hi) & (df['stationlat']<-30) & (df['stationlat']>=-60)].groupby(['stationcode']).mean()
    diff = delta_hi_30S60S - delta_lo_30S60S    
    Nstations = len((delta_lo_30S60S.append(delta_hi_30S60S)).iloc[:,1:13].index.unique())        
    Nmonths = np.isfinite(delta_lo_30S60S.append(delta_hi_30S60S)).iloc[:,1:13].sum().sum()
    figstr = 'delta_cc_' + year_range_str + '_30S60S.png'
    titlestr = 'Change in normalised anomaly: ' + year_range + '\n 30<lat≤60°S' + ': N(stations)=' + "{0:.0f}".format(Nstations) + ',' + ' N(months)=' + "{0:.0f}".format(Nmonths)
    lon = (delta_lo_30S60S.append(delta_hi_30S60S))['stationlon']
    lat = (delta_lo_30S60S.append(delta_hi_30S60S))['stationlat']
    mapfigstr = 'map_' + figstr 
    maptitlestr = titlestr[57:]
    plot_stations(lon,lat,mapfigstr,maptitlestr)
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_60S90S = df[(df['year']>=year_lo) & (df['year']<=year_Q2) & (df['stationlat']<-60) & (df['stationlat']>=-90)].groupby(['stationcode']).mean()
    delta_hi_60S90S = df[(df['year']>year_Q2) & (df['year']<=year_hi) & (df['stationlat']<-60) & (df['stationlat']>=-90)].groupby(['stationcode']).mean()
    diff = delta_hi_60S90S - delta_lo_60S90S    
    Nstations = len((delta_lo_60S90S.append(delta_hi_60S90S)).iloc[:,1:13].index.unique())        
    Nmonths = np.isfinite(delta_lo_60S90S.append(delta_hi_60S90S)).iloc[:,1:13].sum().sum()
    figstr = 'delta_cc_' + year_range_str + '_60S90S.png'
    titlestr = 'Change in normalised anomaly: ' + year_range + '\n 60<lat≤90°S' + ': N(stations)=' + "{0:.0f}".format(Nstations) + ',' + ' N(months)=' + "{0:.0f}".format(Nmonths)
    lon = (delta_lo_60S90S.append(delta_hi_60S90S))['stationlon']
    lat = (delta_lo_60S90S.append(delta_hi_60S90S))['stationlat']
    mapfigstr = 'map_' + figstr 
    maptitlestr = titlestr[57:]
    plot_stations(lon,lat,mapfigstr,maptitlestr)
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_global = df[(df['year']>=year_lo) & (df['year']<=year_Q2)].groupby(['stationcode']).mean()
    delta_hi_global = df[(df['year']>year_Q2) & (df['year']<=year_hi)].groupby(['stationcode']).mean()
    diff = delta_hi_global - delta_lo_global    
    Nstations = len((delta_lo_global.append(delta_hi_global)).iloc[:,1:13].index.unique())        
    Nmonths = np.isfinite(delta_lo_global.append(delta_hi_global)).iloc[:,1:13].sum().sum()
    figstr = 'delta_cc_' + year_range_str + '_global.png'
    titlestr = 'Change in normalised anomaly: ' + year_range + '\n Global' + ': N(stations)=' + "{0:.0f}".format(Nstations) + ',' + ' N(months)=' + "{0:.0f}".format(Nmonths)
    lon = (delta_lo_global.append(delta_hi_global))['stationlon']
    lat = (delta_lo_global.append(delta_hi_global))['stationlat']
    mapfigstr = 'map_' + figstr 
    maptitlestr = titlestr[57:]
    plot_stations(lon,lat,mapfigstr,maptitlestr)
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_us = df[(df['year']>=year_lo) & (df['year']<=year_Q2) & ((df['stationcountry']=='USA') | (df['stationcountry']=='USA---------'))].groupby(['stationcode']).mean()
    delta_hi_us = df[(df['year']>year_Q2) & (df['year']<=year_hi) & ((df['stationcountry']=='USA') | (df['stationcountry']=='USA---------'))].groupby(['stationcode']).mean()
    diff = delta_hi_us - delta_lo_us    
    Nstations = len((delta_lo_us.append(delta_hi_us)).iloc[:,1:13].index.unique())        
    Nmonths = np.isfinite(delta_lo_us.append(delta_hi_us)).iloc[:,1:13].sum().sum()
    figstr = 'delta_cc_' + year_range_str + '_us.png'
    titlestr = 'Change in normalised anomaly: ' + year_range + '\n USA' + ': N(stations)=' + "{0:.0f}".format(Nstations) + ',' + ' N(months)=' + "{0:.0f}".format(Nmonths)
    lon = (delta_lo_us.append(delta_hi_us))['stationlon']
    lat = (delta_lo_us.append(delta_hi_us))['stationlat']
    mapfigstr = 'map_' + figstr 
    maptitlestr = titlestr[57:]
    plot_stations(lon,lat,mapfigstr,maptitlestr)
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_australia = df[(df['year']>=year_lo) & (df['year']<=year_Q2) & ((df['stationcountry']=='AUSTRALIA') | (df['stationcountry']=='AUSTRALIA---'))].groupby(['stationcode']).mean()
    delta_hi_australia = df[(df['year']>year_Q2) & (df['year']<=year_hi) & ((df['stationcountry']=='AUSTRALIA') | (df['stationcountry']=='AUSTRALIA---'))].groupby(['stationcode']).mean()
    diff = delta_hi_australia - delta_lo_australia   
    Nstations = len((delta_lo_australia.append(delta_hi_australia)).iloc[:,1:13].index.unique())        
    Nmonths = np.isfinite(delta_lo_australia.append(delta_hi_australia)).iloc[:,1:13].sum().sum()
    figstr = 'delta_cc_' + year_range_str + '_australia.png'
    titlestr = 'Change in normalised anomaly: ' + year_range + '\n Australia' + ': N(stations)=' + "{0:.0f}".format(Nstations) + ',' + ' N(months)=' + "{0:.0f}".format(Nmonths)
    lon = (delta_lo_australia.append(delta_hi_australia))['stationlon']
    lat = (delta_lo_australia.append(delta_hi_australia))['stationlat']
    mapfigstr = 'map_' + figstr 
    maptitlestr = titlestr[57:]
    plot_stations(lon,lat,mapfigstr,maptitlestr)
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_00S60S_noaus = df[(df['year']>=year_lo) & (df['year']<=year_Q2) & (df['stationlat']<=0) & (df['stationlat']>=-60) & (df['stationcountry']!='AUSTRALIA') & (df['stationcountry']!='AUSTRALIA---')].groupby(['stationcode']).mean()
    delta_hi_00S60S_noaus = df[(df['year']>year_Q2) & (df['year']<=year_hi) & (df['stationlat']<=0) & (df['stationlat']>=-60) & (df['stationcountry']!='AUSTRALIA') & (df['stationcountry']!='AUSTRALIA---')].groupby(['stationcode']).mean()
    diff = delta_hi_00S60S_noaus - delta_lo_00S60S_noaus    
    Nstations = len((delta_lo_00S60S_noaus.append(delta_hi_00S60S_noaus)).iloc[:,1:13].index.unique())        
    Nmonths = np.isfinite(delta_lo_00S60S_noaus.append(delta_hi_00S60S_noaus)).iloc[:,1:13].sum().sum()
    figstr = 'delta_cc_' + year_range_str + '_00S60S_noaus.png'
    titlestr = 'Change in normalised anomaly: ' + year_range + '\n 0≤lat≤60°S (-Australia)' + ': N(stations)=' + "{0:.0f}".format(Nstations) + ',' + ' N(months)=' + "{0:.0f}".format(Nmonths)
    lon = (delta_lo_00S60S_noaus.append(delta_hi_00S60S_noaus))['stationlon']
    lat = (delta_lo_00S60S_noaus.append(delta_hi_00S60S_noaus))['stationlat']
    mapfigstr = 'map_' + figstr 
    maptitlestr = titlestr[57:]
    plot_stations(lon,lat,mapfigstr,maptitlestr)
    plot_hist_array(diff, figstr, titlestr)
        
    delta_lo_eu = df[(df['year']>=year_lo) & (df['year']<=year_Q2) & (df['stationlat']>30) & (df['stationlat']<=70) & (df['stationlon']>-10) & (df['stationlon']<=50) ].groupby(['stationcode']).mean()
    delta_hi_eu = df[(df['year']>year_Q2) & (df['year']<=year_hi) & (df['stationlat']>30) & (df['stationlat']<=70) & (df['stationlon']>-10) & (df['stationlon']<=50) ].groupby(['stationcode']).mean()
    diff = delta_hi_eu - delta_lo_eu    
    Nstations = len((delta_lo_eu.append(delta_hi_eu)).iloc[:,1:13].index.unique())        
    Nmonths = np.isfinite(delta_lo_eu.append(delta_hi_eu)).iloc[:,1:13].sum().sum()
    figstr = 'delta_cc_' + year_range_str + '_eu.png'
    titlestr = 'Change in normalised anomaly: ' + year_range + '\n 30<lat≤70°N, -10<lon≤50°W: Europe (+ME)' + ': N(stations)=' + "{0:.0f}".format(Nstations) + ',' + ' N(months)=' + "{0:.0f}".format(Nmonths)
    lon = (delta_lo_eu.append(delta_hi_eu))['stationlon']
    lat = (delta_lo_eu.append(delta_hi_eu))['stationlat']
    mapfigstr = 'map_' + figstr 
    maptitlestr = titlestr[57:]
    plot_stations(lon,lat,mapfigstr,maptitlestr)
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_eu2 = df[(df['year']>=year_lo) & (df['year']<=year_Q2) & (df['stationlat']>30) & (df['stationlat']<=70) & (df['stationlon']>-10) & (df['stationlon']<=30) ].groupby(['stationcode']).mean()
    delta_hi_eu2 = df[(df['year']>year_Q2) & (df['year']<=year_hi) & (df['stationlat']>30) & (df['stationlat']<=70) & (df['stationlon']>-10) & (df['stationlon']<=30) ].groupby(['stationcode']).mean()
    diff = delta_hi_eu2 - delta_lo_eu2    
    Nstations = len((delta_lo_eu2.append(delta_hi_eu2)).iloc[:,1:13].index.unique())        
    Nmonths = np.isfinite(delta_lo_eu2.append(delta_hi_eu2)).iloc[:,1:13].sum().sum()
    figstr = 'delta_cc_' + year_range_str + '_eu2.png'
    titlestr = 'Change in normalised anomaly: ' + year_range + '\n 30<lat≤70°N, -10<lon≤30°W: Europe (-ME)' + ': N(stations)=' + "{0:.0f}".format(Nstations) + ',' + ' N(months)=' + "{0:.0f}".format(Nmonths)
    lon = (delta_lo_eu2.append(delta_hi_eu2))['stationlon']
    lat = (delta_lo_eu2.append(delta_hi_eu2))['stationlat']
    mapfigstr = 'map_' + figstr 
    maptitlestr = titlestr[57:]
    plot_stations(lon,lat,mapfigstr,maptitlestr)
    plot_hist_array(diff, figstr, titlestr)

    df = pd.read_csv('df.csv', index_col=0)

#------------------------------------------------------------------------------
# PLOTS
#------------------------------------------------------------------------------

if plot_temporal_coverage == True:
    
#    coverage per month:    
#    plt.plot(df.groupby('year').sum().iloc[:,0:12])
    
    #------------------------------------------------------------------------------
    # PLOT: histogram of temporal coverage: to 1900
    #------------------------------------------------------------------------------
             
    nbins = 1900 - df['year'].min() + 1
    bins = np.linspace(df['year'].min(), 1900, nbins) 
#    counts, edges = np.histogram(df[df['year']<=1900]['year'], nbins, range=[df['year'].min(),1900+1], density=False)    
    counts, edges = np.histogram(df['year'], nbins, range=[df['year'].min(),1900], density=False)        
    Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
    Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]
             
    figstr = 'crutem5-histogram-1900.png'
    titlestr = 'Yearly coverage: CRUTEM5 (to 1900): N=' + "{0:.0f}".format(np.sum(counts))

    fig, ax = plt.subplots(figsize=(15,10))          
#    plt.hist(df[df['year']<=1900]['year'], bins=nbins, density=False, facecolor='grey', alpha=0.5, label='KDE')
    plt.fill_between(bins, counts, step="pre", facecolor='lightgrey', alpha=1.0)    
#    plt.plot(bins, counts, drawstyle='steps', linewidth=2, color='black')    
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

    #------------------------------------------------------------------------------
    # PLOT: histogram of temporal coverage: full record
    #------------------------------------------------------------------------------

    nbins = df['year'].max() - df['year'].min() + 1
    bins = np.linspace(df['year'].min(), df['year'].max(), nbins) 
    counts, edges = np.histogram(df['year'], nbins, range=[df['year'].min(),df['year'].max()], density=False)    
    Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
    Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]
             
    figstr = 'crutem5-histogram-full.png'
    titlestr = 'Yearly coverage: CRUTEM5 (full): N=' + "{0:.0f}".format(np.sum(counts)) 

    fig, ax = plt.subplots(figsize=(15,10))          
    plt.fill_between(bins, counts, step="pre", facecolor='lightgrey', alpha=1.0)
#    plt.plot(bins, counts, drawstyle='steps', linewidth=2, color='black')    
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
    
    #------------------------------------------------------------------------------
    # PLOT: histogram of latitudinal coverage
    #------------------------------------------------------------------------------

    nbins = lat_end - lat_start + 1
    bins = np.linspace(lat_start, lat_end, nbins) 
    counts, edges = np.histogram(df['stationlat'], nbins, range=[lat_start,lat_end], density=False)       
    Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
    Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]

    figstr = 'crutem5-histogram-lat-full.png'
    titlestr = 'Latitudinal coverage: CRUTEM5 (full)'

    fig, ax = plt.subplots(figsize=(15,10))          
    plt.fill_between(bins, counts, step="pre", facecolor='lightgrey', alpha=1.0)
#    plt.plot(bins, counts, drawstyle='steps', linewidth=2, color='black')    
    plt.axvline(x=Q1, color='blue', label='Q1: ' + "{0:.0f}".format(Q1))    
    plt.axvline(x=Q2, color='red', label='Q2: ' + "{0:.0f}".format(Q2))    
    plt.axvline(x=Q3, color='blue', label='Q3: ' + "{0:.0f}".format(Q3))    
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='best', fontsize=fontsize)
    plt.xlabel("Latitude, $\mathrm{\degree}N$", fontsize=fontsize)
    plt.ylabel("Monthly values per degree", fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close(fig)

    #------------------------------------------------------------------------------
    # PLOT: histogram of longitudinal coverage
    #------------------------------------------------------------------------------

    nbins = lon_end - lon_start + 1
    bins = np.linspace(lon_start, lon_end, nbins) 
    counts, edges = np.histogram(df['stationlon'], nbins, range=[lon_start,lon_end], density=False)       
    Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
    Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]

    figstr = 'crutem5-histogram-lon-full.png'
    titlestr = 'Longitudinal coverage: CRUTEM5 (full)'

    fig, ax = plt.subplots(figsize=(15,10))          
    plt.fill_between(bins, counts, step="pre", facecolor='lightgrey', alpha=1.0)
#    plt.plot(bins, counts, drawstyle='steps', linewidth=2, color='black')    
    plt.axvline(x=Q1, color='blue', label='Q1: ' + "{0:.0f}".format(Q1))    
    plt.axvline(x=Q2, color='red', label='Q2: ' + "{0:.0f}".format(Q2))    
    plt.axvline(x=Q3, color='blue', label='Q3: ' + "{0:.0f}".format(Q3))    
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='best', fontsize=fontsize)
    plt.xlabel("Longitude, $\mathrm{\degree}W$", fontsize=fontsize)
    plt.ylabel("Monthly values per degree", fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close(fig)
          
if plot_station_timeseres == True:
    
    #------------------------------------------------------------------------------
    # PLOT: timeseries per station
    #------------------------------------------------------------------------------
        
    #for j in range(len(np.unique(stationcode))):
    for j in range(station_start,station_end):
    
        # form station timeseries
        da = df[df['stationcode']==df['stationcode'].unique()[j]].iloc[:,range(0,13)]
    #   da_melt = da.melt(id_vars='year').sort_values(by=['year']).reset_index()

        ts = []    
        for i in range(len(da)):            
            monthly = da.iloc[i,1:]
            ts = ts + monthly.to_list()    
        ts = np.array(ts)                
        ts_yearly = []    
        ts_yearly_sd = []    
        for i in range(len(da)):            
            yearly = np.nanmean(da.iloc[i,1:])
            yearly_sd = np.nanstd(da.iloc[i,1:])
            ts_yearly.append(yearly)    
            ts_yearly_sd.append(yearly_sd)    
        ts_yearly_sd = np.array(ts_yearly_sd)                    
        t = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts), freq='M')     
        t_yearly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='A')   
           
        n = len(t_yearly)            
        colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
        hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]
    
        figstr = 'timeseries_' + str(int(df['stationcode'].unique()[j])) + '.png'
        titlestr = 'Monthly temperature anomaly timeseries: stationcode=' + str(int(df['stationcode'].unique()[j]))
              
        fig, ax = plt.subplots(figsize=(15,10))      
#        plt.plot(t,ts, color='lightgrey', label='Monthly')
        plt.errorbar(t_yearly, ts_yearly, yerr=ts_yearly_sd, xerr=None, fmt='None', ecolor=hexcolors, label='Yearly mean ± 1 s.d.')                                   
        for k in range(n):     
            if k==n-1:
                plt.scatter(t_yearly[k],ts_yearly[k], color=hexcolors[k], label='Yearly mean')
            else:
                plt.scatter(t_yearly[k],ts_yearly[k], color=hexcolors[k], label=None)
        plt.clim([min(ts_yearly),max(ts_yearly)])
        plt.tick_params(labelsize=fontsize)
        plt.legend(loc=2, ncol=1, fontsize=fontsize)
        plt.xlabel("Year", fontsize=fontsize)
        plt.ylabel("Temperature anomaly, $\mathrm{\degree}C$", fontsize=fontsize)
        plt.title(titlestr, fontsize=fontsize)
        plt.savefig(figstr)
        plt.close(fig)

if plot_monthly_climatology == True:
    
    #------------------------------------------------------------------------------
    # PLOT: seasonal cycle per station
    #------------------------------------------------------------------------------
        
    #for j in range(len(np.unique(stationcode))):        
    for j in range(station_start,station_end):
    
        X = df[df['stationcode']==df['stationcode'].unique()[j]].iloc[:,0]    
        Y = df[df['stationcode']==df['stationcode'].unique()[j]].iloc[:,range(1,13)].T   
        n = len(Y.T)            
        colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
        hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]
        
        figstr = 'seasonal-cycle_' + str(int(df['stationcode'].unique()[j])) + '.png'
        titlestr = 'Monthly temperature anomaly seasonal cycle: stationcode=' + str(int(df['stationcode'].unique()[j]))

        fig, ax = plt.subplots(figsize=(15,10))      
        for k in range(n):     
            if k==0:
                plt.plot(np.arange(1,13),Y.iloc[:,k], linewidth=3, color=hexcolors[k], label=str(X.iloc[k]))
            elif k==n-1:
                plt.plot(np.arange(1,13),Y.iloc[:,k], linewidth=3, color=hexcolors[k], label=str(X.iloc[k]))
            else:
                plt.plot(np.arange(1,13),Y.iloc[:,k], linewidth=0.5, color=hexcolors[k], label=None)                
        plt.legend(loc=2, ncol=1, fontsize=fontsize)
        plt.tick_params(labelsize=fontsize)   
        plt.xlabel("Month", fontsize=fontsize)
        plt.ylabel("Monthly temperature anomaly, $\mathrm{\degree}C$", fontsize=fontsize)        
        plt.title(titlestr, fontsize=fontsize)
        plt.savefig(figstr)
        plt.close(fig)

        # ----------------------------------------
        # Beautify with Zach Labe's dataviz format
        # ----------------------------------------

#        plt.rc('savefig', facecolor='black')
#        plt.rc('axes', edgecolor='darkgrey')
#        plt.rc('xtick', color='darkgrey')
#        plt.rc('ytick', color='darkgrey')
#        plt.rc('axes', labelcolor='darkgrey')
#        plt.rc('axes', facecolor='black')
#        plt.rc('text',usetex=False)
#        plt.rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']}) 
#
#        fig, ax = plt.subplots(figsize=(15,10))      
#
#        def adjust_spines(ax, spines):
#            for loc, spine in ax.spines.items():
#                if loc in spines:
#                    spine.set_position(('outward', 5))
#                else:
#                    spine.set_color('none')  
#            if 'left' in spines:
#                ax.yaxis.set_ticks_position('left')
#            else:
#                ax.yaxis.set_ticks([])
#            if 'bottom' in spines:
#                ax.xaxis.set_ticks_position('bottom')
#            else:
#                ax.xaxis.set_ticks([])
#        ax.tick_params('both',length=5.5,width=2,which='major')             
#        adjust_spines(ax, ['left','bottom'])            
#        ax.spines['top'].set_color('none')
#        ax.spines['right'].set_color('none')
#        ax.spines['left'].set_linewidth(2)
#        ax.spines['bottom'].set_linewidth(2)
#    
#        n = len(Y.T)
#        color=iter(cmocean.cm.balance(np.linspace(0.05,0.95,n)))
#        for i in range(n):
#            if i == n-1:
#                c = 'gold'
#                l = 3
#                plt.plot(np.arange(1,13),Y.iloc[:,i],c=c,zorder=3,linewidth=l,label='Year '+"{0:.0f}".format(X.iloc[i]))
#            else:
#                c=next(color)
#                l = 1.5
#                plt.plot(np.arange(1,13),Y.iloc[:,i],c=c,zorder=1,linewidth=l,alpha=1)       
#            if (Mod(i,10) == 0)|(i==n-1):
#                plt.text(12.5, Y.iloc[-1,i], "{0:.0f}".format(X.iloc[i]), color=c,fontsize=9,ha='center',va='center')          
#        plt.ylabel("Monthly temperature anomaly, $\mathrm{\degree}C$",fontsize=16, color='darkgrey') 
#        xlabels = list(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
#        plt.xticks(np.arange(1,13,1),xlabels,rotation=0,fontsize=9)
#        plt.xlim([1,13])       
#        plt.yticks(np.arange(0,int(np.nanmax(Y)+2),3),map(str,np.arange(0,int(np.nanmax(Y)+2),3)),fontsize=10)
#        plt.ylim([np.nanmin(Y.T),np.nanmax(Y.T)])
#        plt.subplots_adjust(bottom=0.15)  
#        plt.text(0.,-2.0,"DATA: CRUTEM v5.1 (Osborn et al, 2020)", fontsize=5,rotation='horizontal',ha='left',color='darkgrey')
#        plt.text(0.,-1.9,"SOURCE: UEA CRU", fontsize=5,rotation='horizontal',ha='left',color='darkgrey')
#        plt.text(0.,-1.8,"GRAPHIC: Michael Taylor (@MichaelTaylorEO)", fontsize=5,rotation='horizontal',ha='left',color='darkgrey')
#        plt.title(titlestr,fontsize=25,color='w') 
#        plt.savefig(figstr)
#        plt.close(fig)

if plot_locations == True:
    
    #------------------------------------------------------------------------------
    # PLOT: stations on world map
    #------------------------------------------------------------------------------

    lon = df['stationlon']
    lat = df['stationlat']
    
    figstr = 'location_map.png'
    titlestr = 'Station locations'
     
    fig  = plt.figure(figsize=(15,10))
    p = ccrs.PlateCarree(central_longitude=0); threshold = 0
    ax = plt.axes(projection=p)
    ax.set_global()
#    ax.stock_img()
    ax.add_feature(cf.COASTLINE, edgecolor="lightblue")
    ax.add_feature(cf.BORDERS, edgecolor="lightblue")
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
    plt.scatter(x=lon, y=lat, 
                color="maroon", s=1, alpha=0.5,
                transform=ccrs.PlateCarree()) 
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close('all')
        
#------------------------------------------------------------------------------
print('** END')

