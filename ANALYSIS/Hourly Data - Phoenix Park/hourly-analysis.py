#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: hourly-analysis.py
#------------------------------------------------------------------------------
# Version 0.2
# 19 October, 2020
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
# Stats libraries:
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

#------------------------------------------------------------------------------
# METHODS: 
#------------------------------------------------------------------------------

def linear_regression_ols(x,y):

    regr = linear_model.LinearRegression()
    # regr = TheilSenRegressor(random_state=42)
    # regr = RANSACRegressor(random_state=42)

    X = x[:, np.newaxis]    
    # X = x.values.reshape(len(x),1)
    t = np.linspace(X.min(),X.max(),len(X)) # dummy var spanning [xmin,xmax]        
    regr.fit(X, y)
    ypred = regr.predict(t.reshape(-1, 1))
    slope = regr.coef_
    intercept = regr.intercept_
    mse = mean_squared_error(y,ypred)
    r2 = r2_score(y,ypred) 
    
    return t, ypred, slope, intercept, mse, r2

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

print('perform hourly analysis ...')

daily_mean = df.groupby(['yyyy','mm','dd']).mean()['temp'] # AWS: 00:00 --> 23:00
daily_min = df.groupby(['yyyy','mm','dd']).min()['temp']
daily_max = df.groupby(['yyyy','mm','dd']).max()['temp']
daily_hh = np.array(df.groupby(['yyyy','mm','dd','hh'])['temp'])
daily = np.array(df.groupby(['yyyy','mm','dd'])['temp'])
    
# Tx is read at 18 and Tn at 06.  In some parts of world (e.g. Europe) the Tx refers to the 06-18 period and the Tn to 18-06.

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
                        
dg0 = pd.DataFrame({'t_daily':t_daily,'daily_mean0':np.array(daily_mean)})
dg1 = pd.DataFrame({'t_daily':t_daily,'daily_mean1':np.array(daily_tmean)})
dg2 = pd.DataFrame({'t_daily':dates_tmean_06_18,'daily_mean2':daily_tmean_06_18.ravel()})
dg01 = pd.merge(dg0, dg1, how='inner', on=['t_daily'])
dg = pd.merge(dg01, dg2, how='inner', on=['t_daily'])
        
daily_diff01 = dg['daily_mean0']-dg['daily_mean1']
daily_diff02 = dg['daily_mean0']-dg['daily_mean2']
mean_daily_diff01 = np.nanmean(daily_diff01)
mean_daily_diff02 = np.nanmean(daily_diff02)

print('perform monthly analysis on daily data ...')

monthly_means = dg.groupby(pd.Grouper(key='t_daily', freq='1M')).mean() 
monthly_diff01 = monthly_means['daily_mean0']-monthly_means['daily_mean1']
monthly_diff02 = monthly_means['daily_mean0']-monthly_means['daily_mean2']
mean_monthly_diff01 = np.nanmean(monthly_diff01)
mean_monthly_diff02 = np.nanmean(monthly_diff02)

dh = dg.copy().dropna()
daily_0 = dh['daily_mean0']
daily_1 = dh['daily_mean1'] 
daily_2 = dh['daily_mean2'] 

dm = monthly_means.dropna()
monthly_0 = dm['daily_mean0']
monthly_1 = dm['daily_mean1'] 
monthly_2 = dm['daily_mean2'] 

t_daily_01, ypred_daily_01, slope_daily_01, intercept_daily_01, mse_daily_01, r2_daily_01 = linear_regression_ols(daily_0,daily_1)
t_daily_02, ypred_daily_02, slope_daily_02, intercept_daily_02, mse_daily_02, r2_daily_02 = linear_regression_ols(daily_0,daily_2)
t_monthly_01, ypred_monthly_01, slope_monthly_01, intercept_monthly_01, mse_monthly_01, r2_monthly_01 = linear_regression_ols(monthly_0,monthly_1)
t_monthly_02, ypred_monthly_02, slope_monthly_02, intercept_monthly_02, mse_monthly_02, r2_monthly_02 = linear_regression_ols(monthly_0,monthly_2)

#------------------------------------------------------------------------------
# PLOT HOURLY ANALYSIS
#------------------------------------------------------------------------------

print('plot_hourly_analysis ...')

# PLOT: Tmeans (all hours, (Tmin+Tmax)/2, (T06+T18)/2 )
            
figstr = 'phoenix-park-temp-daily-tmean.png'
titlestr = 'Phoenix Park: 2m-Temperature (Daily Mean)'
             
xmin = df['datestamp'].min()
xmax = df['datestamp'].max()

fig, ax = plt.subplots(figsize=(15,10))          
plt.scatter(df['datestamp'], df['temp'], marker='.', lw=1, color='lightgrey', label='hourly')
plt.scatter(dg['t_daily'],dg['daily_mean1'], marker='o', facecolor='none', color='red', label='daily mean = (T06+T18)/2')
plt.scatter(dg['t_daily'],dg['daily_mean2'], marker='s', facecolor='none', color='blue', label='daily mean = (Tmin+Tmax)/2')
plt.scatter(dg['t_daily'],dg['daily_mean0'], marker='.', s=5, facecolor='none', color='black', label='24-hr daily mean')
plt.xlim(xmin,xmax)    
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
         
xmin = df['datestamp'].min()
xmax = df['datestamp'].max()
    
fig, ax = plt.subplots(figsize=(15,10))     
plt.axhline(y=0, lw=1, color='teal', label='no bias')
plt.scatter(dg['t_daily'],daily_diff02, marker='o', facecolor='none', color='pink', label='24-hr daily mean - (T06+T18)/2')
plt.scatter(dg['t_daily'],daily_diff01, marker='s', facecolor='none', color='lightblue', label='24-hr daily mean - (Tmin+Tmax)/2')
plt.axhline(y=mean_daily_diff01, lw=2, color='blue', label='mean bias='+str(np.round(mean_daily_diff01,3)))
plt.axhline(y=mean_daily_diff02, lw=2, color='red', label='mean bias='+str(np.round(mean_daily_diff02,3)))
plt.xlim(xmin,xmax)    
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
# PLOT DAILY VERSUS MONTHLY CORRELATIONS
#------------------------------------------------------------------------------
    
    # Tim: 
    #
    # Four scatter plots of 1 vs 2 and 1 vs 3 for daily means 
    # and for monthly means of the daily values might be informative, 
    # showing the degree of scatter and how much this scatter 
    # reduces with the monthly averages, 
    # and the best fit lines will presumably have an offset that 
    # represents the mean bias and slopes close to 1. 
    # Actually if the slopes aren't close to 1 it will be interesting 
    # -- perhaps indicating that the bias is not constant but 
    # depends either on season (more likely) or on extreme weather days 
    # (possible, e.g. the bias might be dependent on the diurnal temperature 
    # range which may be greater on clear sky days which in turn may be linked 
    # with cold daily-means in winter but warm daily-means in summer).

    # PLOT: Correlation of daily_mean0 v daily_mean1
    # PLOT: Correlation of daily_mean0 v daily_mean2
    # PLOT: Correlation of monthly_mean0 v monthly_mean1
    # PLOT: Correlation of monthly_mean0 v monthly_mean2

figstr = 'phoenix-park-temp-corr-tmean-txtn.png'
titlestr = 'Phoenix Park: Correlation of Tmean and (Tx+Tn)/2: daily and monthly'         
labelstr_OLS_daily = 'OLS'+r' ($\alpha$='+str(np.round(slope_daily_01[0],3))+r',$\beta$='+str(np.round(intercept_daily_01,3))+')'
labelstr_OLS_monthly = 'OLS'+r' ($\alpha$='+str(np.round(slope_monthly_01[0],3))+r',$\beta$='+str(np.round(intercept_monthly_01,3))+')'

xmin = np.min([daily_0.min(),daily_1.min()])
xmax = np.max([daily_0.max(),daily_1.max()])

fig, ax = plt.subplots(figsize=(15,10))   
plt.plot([xmin,xmax],[xmin,xmax], color='teal', ls='-', lw=1, label='1:1')       
plt.plot(t_daily_01,ypred_daily_01, color='black', linewidth=1, label=labelstr_OLS_daily)
plt.plot(t_monthly_01,ypred_monthly_01, color='blue', linewidth=2, label=labelstr_OLS_monthly)
plt.scatter(daily_0,daily_1, marker='.', facecolor='lightgrey', color='black', alpha=1, label='daily')
plt.scatter(monthly_0,monthly_1, marker='s', facecolor='none', color='lightblue', alpha=1, label='monthly')
plt.xlim(xmin,xmax)
plt.ylim(xmin,xmax)
ax.set_aspect('equal') 
ax.xaxis.grid(True, which='major')      
ax.yaxis.grid(True, which='major')  
plt.tick_params(labelsize=16)    
plt.legend(loc='upper left', fontsize=16)
plt.xlabel('Tmean', fontsize=fontsize)
plt.ylabel(r'(Tx+Tn)/2', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr)
plt.close(fig)

figstr = 'phoenix-park-temp-corr-tmean-t06t18.png'
titlestr = 'Phoenix Park: Correlation of Tmean and (T06+T18)/2: daily and monthly'         
labelstr_OLS_daily = 'OLS'+r' ($\alpha$='+str(np.round(slope_daily_02[0],3))+r',$\beta$='+str(np.round(intercept_daily_02,3))+')'
labelstr_OLS_monthly = 'OLS'+r' ($\alpha$='+str(np.round(slope_monthly_02[0],3))+r',$\beta$='+str(np.round(intercept_monthly_02,3))+')'

#xmin = np.min([daily_0.min(),daily_2.min()])
#xmax = np.max([daily_0.max(),daily_2.max()])

fig, ax = plt.subplots(figsize=(15,10))          
plt.plot([xmin,xmax],[xmin,xmax], color='teal', ls='-', lw=1, label='1:1')
plt.plot(t_daily_02,ypred_daily_02, color='black', linewidth=1, label=labelstr_OLS_daily)
plt.plot(t_monthly_02,ypred_monthly_02, color='red', linewidth=2, label=labelstr_OLS_monthly)
plt.scatter(daily_0,daily_2, marker='.', facecolor='lightgrey', color='black', alpha=1, label='daily')
plt.scatter(monthly_0,monthly_2, marker='o', facecolor='none', color='pink', alpha=1, label='monthly')
plt.xlim(xmin,xmax)
plt.ylim(xmin,xmax)
ax.set_aspect('equal') 
ax.xaxis.grid(True, which='major')      
ax.yaxis.grid(True, which='major')  
plt.tick_params(labelsize=16)    
plt.legend(loc='upper left', fontsize=16)
plt.xlabel('Tmean', fontsize=fontsize)
plt.ylabel(r'(T06+T18)/2', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr)
plt.close(fig)
       
#------------------------------------------------------------------------------
print('** END')

