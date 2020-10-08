#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xarray as xr
import scipy
import scipy.stats as stats    
from sklearn.preprocessing import StandardScaler
# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.cm as cm
import matplotlib.colors as c
from matplotlib.colors import Normalize
from matplotlib import colors as mcol
from matplotlib.cm import ScalarMappable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
matplotlib.rcParams['text.usetex'] = False
# Animated GIF libraries:
import glob
from PIL import Image


# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------

fontsize = 20

# -----------------------------------------------------------------------------
# LOAD: global anomalies
# -----------------------------------------------------------------------------
    
df_temp = pd.read_pickle('df_temp.pkl', compression='bz2')
value = np.where(df_temp['stationcode'].unique()=='037401')[0][0]
da = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]].iloc[:,range(0,13)]

ts_monthly = []    
for i in range(len(da)):            
   monthly = da.iloc[i,1:]
   ts_monthly = ts_monthly + monthly.to_list()    
ts_monthly = np.array(ts_monthly)   
# Solve Y1677-Y2262 Pandas bug with xarray:        
# t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')                  
t_monthly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M', calendar='noleap')     

# TRIM: (n-2) as last 2 monthly values which are NaN

t = t_monthly[:-2]
ts = ts_monthly[:-2]
        
df = pd.DataFrame(columns=['min','max','normal','p5','p10','p90','p95'], index=np.arange(1,13))

for j in range(1,13):

    n = len(da)
    y = da[str(j)]
#   z = (y-y.mean())/y.std()
    z = y[np.isfinite(y)]
    disttype = 'gamma'
    dist = getattr(scipy.stats, disttype)
    param = dist.fit(z)
    gamma_median = dist.median(param[0], loc=param[1], scale=param[2])       # q50
    gamma_10_90 = dist.interval(0.8, param[0], loc=param[1], scale=param[2]) # q10 and q90
    gamma_5_95 = dist.interval(0.9, param[0], loc=param[1], scale=param[2])  # q5 and q95	
    gamma = dist.rvs(param[0], loc=param[1], scale=param[2], size=n)
    gamma_sorted = np.sort(gamma)
    gamma_bins = np.percentile(gamma_sorted, range(0,101))
    z_min = np.min(z)
    z_max = np.max(z)
    
    df['min'][j] = z_min
    df['max'][j] = z_max
    df['normal'][j] = gamma_median
    df['p5'][j] = gamma_5_95[0]
    df['p95'][j] = gamma_5_95[1]
    df['p10'][j] = gamma_10_90[0]
    df['p90'][j] = gamma_10_90[1]

#    p10extremes = z < gamma_bins[10]
#    p90extremes = z > gamma_bins[90]
#    p10extremes_frac = p10extremes.sum()/n
#    p90extremes_frac = p90extremes.sum()/n

# PLOT: timeseries + quantiles + exceedence fraction

yearlist = da['year'].values
for i in range(len(yearlist)):

    year = yearlist[i]
    
    fig,ax = plt.subplots(figsize=(15,10))
    #plt.plot(df.index, df['min'], drawstyle='steps-mid', label='Lowest in record '+str(da.year.min())+'-'+str(da.year.max()), color='cyan', lw=2)
    plt.plot(df.index, df['max'], label='Highest in record '+str(da.year.min())+'-'+str(da.year.max()), color='pink', lw=3)
    plt.plot(df.index, df['min'], label='Lowest in record '+str(da.year.min())+'-'+str(da.year.max()), color='cyan', lw=3)
    #plt.plot(df.index, df['p5'], label=r'', color='black', lw=0.5)
    #plt.plot(df.index, df['p95'], label='', color='black', lw=0.5)
    ax.fill_between(np.array(df.index), np.array(df['p5'].astype(float)), np.array(df['p95'].astype(float)), color='lightgrey', alpha=0.5, label=r'$5^{th}$-$95^{th}$ percentile band')    
    ax.fill_between(np.array(df.index), np.array(df['p10'].astype(float)), np.array(df['p90'].astype(float)), color='grey', alpha=0.3, label=r'$10^{th}$-$90^{th}$ percentile band')    
    #plt.plot(df.index, df['p10'], label=r'$10^{th}$ & $90^{th}$ percentiles', color='purple', lw=1)
    #plt.plot(df.index, df['p90'], label='', color='purple', lw=1)
    plt.plot(df.index, df['normal'], label='1961-1990 standard normals', color='black', lw=2)
    plt.scatter(df.index, da[da['year']==year].iloc[:,1:], marker='o', s=100, color='darkblue', lw=2, facecolor='lightblue', label=str(year))
    #plt.plot(np.array(df.index), np.array(da[da['year']==year].iloc[:,1:].T), marker='o', markersize=5, color='darkblue', lw=2, label=str(year))
    plt.legend(fontsize=12)
    plt.xlabel('Month', fontsize=fontsize)
    plt.ylabel(r'Monthly temperature, $\degree$C', fontsize=fontsize)
    plt.title('Had-CET: '+str(year), fontsize=fontsize)
    #ax.xaxis.grid(True, which='major')      
    ax.yaxis.grid(True, which='major')      
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.savefig('had-cet-gamma-fit-climatology_'+str(year)+'.png')
    plt.close()

    fp_in = "had-cet-gamma-fit-climatology_*.png"
    fp_out = "had-cet-gamma-fit-climatology.gif"

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=100, loop=0)




