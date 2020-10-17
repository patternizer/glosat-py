#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: hourly-analysis.py
#------------------------------------------------------------------------------
# Version 0.1
# 16 October, 2020
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
# Dataframe libraries:
import numpy as np
import numpy.ma as ma
from mod import Mod
import itertools
import pandas as pd
import xarray as xr
import pickle
from datetime import datetime
import nc_time_axis
import cftime
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
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

fontsize = 20
load_hourly = True
hourly_analysis = True
plot_hourly_analysis = True

#------------------------------------------------------------------------------
# LOAD HOURLY DATA
#------------------------------------------------------------------------------

if load_hourly == True:
    
    print('loading hourly dataframe ...')

    df_hourly = pd.read_pickle('df_hourly.pkl', compression='bz2')    
    df = df_hourly.copy()

else:
    
    print('load hourly dataset ...') 

    #------------------------------------------------------------------------------
    # LOAD: Phoenix Park hourly dataset
    #------------------------------------------------------------------------------

    nheader = 16
    f = open('hly175.csv')
    lines = f.readlines()
    dates = []
    obs = []
    for i in range(nheader,len(lines)):
        print(i)
        words = lines[i].split(',')
        if len(words) > 1:
            date = pd.to_datetime(words[0])
            val = (len(words)-1)*[None]
            for j in range(len(val)):
                try: val[j] = float(words[j+1])
                except:
                    pass
            dates.append(date)
            obs.append(val) 
    f.close()
    
    dates = np.array(dates)
    obs = np.array(obs)

    # DATAFRAME:

    # df = pd.DataFrame(columns=['year','1','2','3','4','5','6','7','8','9','10','11','12'])
    # df['mean'] = da[da.columns[range(1,13)]].mean(axis=1)

    df = pd.DataFrame(columns=['datestamp','ind','rain','ind','temp','ind','wetb','dewpt','vappr','rhum','msl'])
    df['datestamp'] = dates
    for j in range(1,len(df.columns)):
        df[df.columns[j]] = [ obs[i][j-1] for i in range(len(obs)) ]
    df['yyyy'] = df.datestamp.dt.year
    df['mm'] = df.datestamp.dt.month
    df['dd'] = df.datestamp.dt.day
    df['hh'] = df.datestamp.dt.hour
    df['date'] = [ pd.to_datetime(df['datestamp'][i].date()).strftime('%Y-%m-%d') for i in range(len(obs)) ]       
#   df.drop('datestamp', axis=1, inplace=True)    

    print('save dataframe ...')

    df_hourly = df.copy()
    df_hourly.to_pickle('df_hourly.pkl', compression='bz2')

#------------------------------------------------------------------------------
# PERFORM HOURLY ANALYSIS
#------------------------------------------------------------------------------

if hourly_analysis == True:
    
    print('perform hourly analysis ...')

    daily_mean = df.groupby(['yyyy','mm','dd']).mean()['temp'] # AWS: 00:00 --> 23:00
    daily_min = df.groupby(['yyyy','mm','dd']).min()['temp']
    daily_max = df.groupby(['yyyy','mm','dd']).max()['temp']
    daily_hh = np.array(df.groupby(['yyyy','mm','dd','hh'])['temp'])
    daily = np.array(df.groupby(['yyyy','mm','dd'])['temp'])
    
#   Tx is read at 18 and Tn at 06.  In some parts of world (e.g. Europe) the Tx refers to the 06-18 period and the Tn to 18-06.

    t_daily = [ pd.to_datetime(str(daily_mean.index[i][0])+'-'+str(daily_mean.index[i][1])+'-'+str(daily_mean.index[i][2])) for i in range(len(daily_mean))]
    daily_tmean = (daily_min+daily_max)/2
    dates_tn_06 = []
    dates_tx_18 = []
    daily_tn_06 = []
    daily_tx_18 = []
    mask = []
    for i in range(len(daily_hh)):
        if daily_hh[i][0][3] == 6:            
            obs = daily_hh[i][1]
            date = pd.to_datetime(str(daily_hh[i][0][0])+'-'+str(daily_hh[i][0][1])+'-'+str(daily_hh[i][0][2]))
            daily_tn_06.append(obs)            
            dates_tn_06.append(date)            
        if daily_hh[i][0][3] == 18:            
            obs = daily_hh[i][1]
            date = pd.to_datetime(str(daily_hh[i][0][0])+'-'+str(daily_hh[i][0][1])+'-'+str(daily_hh[i][0][2]))
            daily_tx_18.append(obs)            
            dates_tx_18.append(date)                    
    daily_tn_06 = np.array(daily_tn_06)
    daily_tx_18 = np.array(daily_tx_18)
    dates_tmean_06_18 = dates_tx_18
    daily_tmean_06_18 = (daily_tn_06+daily_tx_18)/2
                        
    dg1 = pd.DataFrame({'t_daily':t_daily,'daily_mean1':np.array(daily_mean)})
    dg2 = pd.DataFrame({'t_daily':dates_tmean_06_18,'daily_mean2':daily_tmean_06_18.ravel()})
    dg = pd.merge(dg1, dg2, how='inner', on=['t_daily'])
    daily_mean_overlap_tmean_06_18 = dg['daily_mean1']-dg['daily_mean2']
    mean_diff_tmean_06_18 = np.mean(daily_mean_overlap_tmean_06_18)

    dh1 = pd.DataFrame({'t_daily':t_daily,'daily_mean1':np.array(daily_mean)})
    dh2 = pd.DataFrame({'t_daily':t_daily,'daily_mean2':daily_tmean.ravel()})
    dh = pd.merge(dh1, dh2, how='inner', on=['t_daily'])
    daily_mean_overlap_tmean = dh['daily_mean1']-dh['daily_mean2']
    mean_diff_tmean = np.mean(daily_mean_overlap_tmean)
       
#------------------------------------------------------------------------------
# PLOT HOURLY ANALYSIS
#------------------------------------------------------------------------------

if plot_hourly_analysis == True:
    
    print('plot_hourly_analysis ...')

    # PLOT: Tmeans (all hours, (Tmin+Tmax)/2, (T06+T18)/2 )
            
    figstr = 'phoenix-park-temp-daily-tmean.png'
    titlestr = 'Phoenix Park: 2m-Temperature (Daily Mean)'
             
    fig, ax = plt.subplots(figsize=(15,10))          
    plt.scatter(df['datestamp'], df['temp'], marker='.', lw=1, color='lightgrey', label='hourly')
#   plt.scatter(t_daily,daily_max, marker='x', color='purple', label='daily max')
    plt.scatter(dates_tmean_06_18,daily_tmean_06_18, marker='s', facecolor='none', color='black', label='daily mean = (Tmin+Tmax)/2')
    plt.scatter(t_daily,daily_tmean, marker='o', facecolor='none', color='blue', label='daily mean = (T06+T18)/2')
    plt.scatter(t_daily,daily_mean, marker='.', facecolor='none', color='red', label='24-hr daily mean')
#   plt.scatter(t_daily,daily_min, marker='o', facecolor='none', color='blue', label='daily min')
    plt.xlim(df['datestamp'].min(),df['datestamp'].max())    
    plt.ylim(-15,30)
    plt.tick_params(labelsize=16)    
    plt.legend(loc='upper left', fontsize=16)
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'2m-Temperature, [K]', fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close(fig)

    # PLOT: Tmean differences from all hours: (Tmin+Tmax)/2, (T06+T18)/2 
            
    figstr = 'phoenix-park-temp-daily-tmean_diff.png'
    titlestr = 'Phoenix Park: 2m-Temperature (Daily Mean): difference from 24-hr estimate'
             
    fig, ax = plt.subplots(figsize=(15,10))          
    plt.scatter(t_daily, daily_mean_overlap_tmean, marker='s', facecolor='none', color='black', label='24-hr daily mean - (Tmin+Tmax)/2')
    plt.scatter(dates_tmean_06_18, daily_mean_overlap_tmean_06_18, marker='o', facecolor='none', color='blue', label='24-hr daily mean - (T06+T18)/2')
    plt.axhline(y=mean_diff_tmean, lw=3, color='grey', label='mean bias='+str(np.round(mean_diff_tmean,3)))
    plt.axhline(y=mean_diff_tmean_06_18, lw=3, color='lightblue', label='mean bias='+str(np.round(mean_diff_tmean_06_18,3)))
    plt.xlim(df['datestamp'].min(),df['datestamp'].max())    
    plt.ylim(-4,4)
    plt.tick_params(labelsize=16)    
    plt.legend(loc='upper left', fontsize=16)
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'2m-Temperature difference, [K]', fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close(fig)

    # PLOT: Temperatures (2m, Wet Bulb, Dew Point)

    figstr = 'phoenix-park-temp-wetb-dewpt.png'
    titlestr = 'Phoenix Park: Temperatures'
             
    fig, ax = plt.subplots(figsize=(15,10))          
    plt.scatter(df['datestamp'], df['temp'], marker='.', lw=1, color='lightgrey', alpha=1, label='2m-Temperature')
    plt.scatter(df['datestamp'], df['wetb'], marker='.', lw=1, color='cyan', alpha=1, label='Wet Bulb')
    plt.scatter(df['datestamp'], df['dewpt'], marker='.', lw=1, color='teal', alpha=0.3, label='Dew Point')
    plt.xlim(df['datestamp'].min(),df['datestamp'].max())    
    plt.ylim(-15,30)
    plt.tick_params(labelsize=16)    
    plt.legend(loc='upper left', fontsize=16)
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'Temperature, [K]', fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close(fig)
        
#------------------------------------------------------------------------------
print('** END')

