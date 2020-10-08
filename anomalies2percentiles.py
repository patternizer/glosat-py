#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 17:59:14 2020

@author: patternizer
"""

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

# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------

fontsize = 20

# -----------------------------------------------------------------------------
# LOAD: global anomalies
# -----------------------------------------------------------------------------
    
df_anom = pd.read_pickle('df_anom.pkl', compression='bz2')

# UNRAVEL: construct monthly timeseries for Had-CET

value = np.where(df_anom['stationcode'].unique()=='037401')[0][0]

da = df_anom[df_anom['stationcode']==df_anom['stationcode'].unique()[value]].iloc[:,range(0,13)]
ts_monthly = []    
for i in range(len(da)):            
   monthly = da.iloc[i,1:]
   ts_monthly = ts_monthly + monthly.to_list()    
ts_monthly = np.array(ts_monthly)   

# Solve Y1677-Y2262 Pandas bug with xarray:        
# t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')                  
t_monthly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M', calendar='noleap')     

ts_yearly = []    
for i in range(len(da)):            
    if da.iloc[i,1:].isnull().all():
        yearly = np.nan
    else:
        yearly = np.nanmean(da.iloc[i,1:])
    ts_yearly.append(yearly)    
ts_yearly = np.array(ts_yearly)      

# Solve Y1677-Y2262 Pandas bug with xarray:        
# t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')                  
t_yearly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='A', calendar='noleap')     

# TRIM: (n-2) as last 2 monthly values which are NaN

t = t_monthly[:-2]
ts = ts_monthly[:-2]

t_tmp = t[(t.year>1960) & (t.year<1991)]
ts_tmp = ts[(t.year>1960) & (t.year<1991)]

t=t_tmp
ts=ts_tmp
# ts_stats = pd.DataFrame(ts, columns=['timeseries']).describe()

# REMAP: derive data vectors

n = len(ts)
x = np.arange(n)                         # --> indices [0,n] 
#y = ts - ts.min()                       # --> y >= 0
y = ts                                   # --> y (raw)
y_sorted = np.sort(y)                    # --> CDF [0,max]
y_map = np.linspace(y.min(), y.max(), n) # --> [ymin:ymax]

# STANDARDIZE: convert to z-scores

z = (y-y.mean())/y.std()
z_sorted = np.sort(z)
z_bins = np.percentile(z_sorted, range(0,101))
z_map = np.linspace(z.min(), z.max(), n)
z_counts = np.histogram(z, bins=n)
z_hist = scipy.stats.rv_histogram(z_counts)
z_cdf = z_hist.cdf(z_map)
z_map100 = np.linspace(z.min(), z.max(), 100)
z_counts100 = np.histogram(z, bins=100)
z_hist100 = scipy.stats.rv_histogram(z_counts100)
z_cdf100 = z_hist100.cdf(z_map100)
z_maxsigma = np.ceil(np.max([np.abs(z_map.min()),np.abs(z_map.min())]))

# FIT: (gamma) distribution to the data

# (available distributions) https://docs.scipy.org/doc/scipy/reference/stats.html

disttype = 'gamma'
dist = getattr(scipy.stats, disttype)
param = dist.fit(z)

# TEST:
#z = dist.rvs(param[0], loc=param[1], scale=param[2], size=n)
#z_sorted = np.sort(z)
#z_map = np.linspace(z.min(), z.max(), n)
#param = dist.fit(z)

# CALCULATE: probability density function (PDF)
# CALCULATE: cumulative distribution function (CDF)
gamma_pdf = dist.pdf(z_map, param[0], param[1], param[2])	
gamma_cdf = dist.cdf(z_map, param[0], loc=param[1], scale=param[2])
gamma_pdf100 = dist.pdf(z_map100, param[0], param[1], param[2])	
gamma_cdf100 = dist.cdf(z_map100, param[0], loc=param[1], scale=param[2])

# DISTRIBUTION: summary statistics
    
gamma_mean, gamma_var, gamma_skew, gamma_kurt = dist.stats(param[0], loc=param[1], scale=param[2], moments='mvsk')
gamma_median = dist.median(param[0], loc=param[1], scale=param[2])            # q50
gamma_iqr = dist.interval(0.5, param[0], loc=param[1], scale=param[2])        # q25 and q75 
gamma_extremes = dist.interval(0.8, param[0], loc=param[1], scale=param[2])   # q10 and q90
	
# RANDOM DRAWS: random variates from fit distribution + calculate percentiles

gamma = dist.rvs(param[0], loc=param[1], scale=param[2], size=n)
gamma_sorted = np.sort(gamma)
gamma_bins = np.percentile(gamma_sorted, range(0,101))
        
#-----------------------------------------------------------------------------
# TEST: K-S p-statistic

p_value = scipy.stats.kstest(z, disttype, args=param)[1]
            
# TEST: chi-square (50 bins)

chi_bins = np.linspace(0,100,101)
chi_percentiles = np.percentile(z, chi_bins)
observed_frequency, bins = (np.histogram(z, bins=chi_percentiles))
cum_observed_frequency = np.cumsum(observed_frequency)
cdf_fitted = dist.cdf(chi_percentiles, param[0], loc=param[1], scale=param[2])
expected_frequency = [] # expected counts in each percentile bin of CDF
for bin in range(len(chi_bins)-1):
    expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
    expected_frequency.append(expected_cdf_area)    
expected_frequency = np.array(expected_frequency) * n
cum_expected_frequency = np.cumsum(expected_frequency)

chi_square = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
#-----------------------------------------------------------------------------
            
# PLOT: histograms + PDFs + quantiles
 
fig,ax = plt.subplots(figsize=(15,10))
h1 = ax.hist(gamma, bins=100, color='cyan', alpha=0.2, density=True, label=r'$\Gamma$ (draws): n='+str(n))
h2 = ax.hist(z, bins=100, color='grey', alpha=0.2, density=True, label='Had-CET monthly anomalies: n='+str(n))
h3 = plt.plot(z_map, gamma_pdf, color='teal', lw=5, label=r'$\Gamma$:'+
         r' ($\alpha=$'+str(np.round(param[0],2))+','+
         r' loc='+str(np.round(param[1],2))+','+
         r' scale='+str(np.round(param[2],2))+')')
ymin = np.min([h1[1].min(),h2[1].min()])
ymax = np.max([h1[1].max(),h2[1].max()])
plt.axvline(gamma_bins[90], ymin, ymax, color='darkblue', ls='--', label=r'$\Gamma$: P90')
plt.axvline(gamma_bins[75], ymin, ymax, color='darkred', ls='--', label=r'$\Gamma$: P75 (Q3)')
plt.axvline(gamma_bins[50], ymin, ymax, color='black', ls='--', label=r'$\Gamma$: P50 (median)')
plt.axvline(gamma_bins[25], ymin, ymax, color='red', ls='--', label=r'$\Gamma$: P25 (Q1)')
plt.axvline(gamma_bins[10], ymin, ymax, color='blue', ls='--', label=r'$\Gamma$: P10')
plt.legend(fontsize=12)
plt.xlabel(r'Standard deviations, $\sigma$', fontsize=fontsize)
plt.ylabel(r'Kernel density estimate (KDE)', fontsize=fontsize)
plt.title(r'Goodness of fit: $\chi^{2}$='+str(round(chi_square,2))+r', p='+str(round(p_value,6)), fontsize=fontsize)
ax.set_xlim(-z_maxsigma,z_maxsigma)
plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.savefig('had-cet-gamma-fit-histogram.png')

# PLOT: observed v theoretical quantiles

title = r'QQ plot for $\Gamma$ fit'
fig = plt.figure(figsize=(15,10)) 
ax1 = fig.add_subplot(121)
ax1.scatter(gamma_sorted, z_sorted, marker=".", color='grey', label='n='+str(n))
ax1.scatter(gamma_bins, z_bins, marker="o", lw=2, color='teal', facecolor='lightgrey', label='Q(1-100)')
ax1.plot([-z_maxsigma,z_maxsigma], [-z_maxsigma,z_maxsigma], color='black', ls='--')
plt.legend(fontsize=12)
ax1.set_xlim(-z_maxsigma,z_maxsigma)
ax1.set_ylim(-z_maxsigma,z_maxsigma)
ax1.set_xlabel('Theoretical quantiles', fontsize=fontsize)
ax1.set_ylabel('Observed quantiles', fontsize=fontsize)
ax1.set_aspect('equal') 
ax1.xaxis.grid(True, which='minor')      
ax1.yaxis.grid(True, which='minor')  
ax1.xaxis.grid(True, which='major')      
ax1.yaxis.grid(True, which='major')  
ax1.tick_params(axis='both', which='major', labelsize=fontsize)
ax1.set_title(title, fontsize=fontsize)
    
# PLOT: observed v theoretical CDFs

title = r'PP plot for $\Gamma$ fit'
ax2 = fig.add_subplot(122)
ax2.scatter(gamma_cdf, z_cdf, marker=".", color='grey', label='n='+str(n))
ax2.scatter(gamma_cdf100, z_cdf100, marker="o", lw=2, color='teal', facecolor='lightgrey', label='P(1-100)')
ax2.plot([0,1], [0,1], color='black', ls='--')
plt.legend(fontsize=12)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel('Theoretical CDF', fontsize=fontsize)
ax2.set_ylabel('Observed CDF', fontsize=fontsize)
ax2.set_aspect('equal') 
ax2.xaxis.grid(True, which='minor')      
ax2.yaxis.grid(True, which='minor')  
ax2.xaxis.grid(True, which='major')      
ax2.yaxis.grid(True, which='major')  
ax2.tick_params(axis='both', which='major', labelsize=fontsize)
ax2.set_title(title, fontsize=fontsize)   
plt.tight_layout(pad=4)
plt.savefig('had-cet-gamma-fit-pp-qq.png')

# PLOT: timeseries + quantiles + exceedence fraction

p10extremes = z < gamma_bins[10]
p90extremes = z > gamma_bins[90]
p10extremes_frac = p10extremes.sum()/n
p90extremes_frac = p90extremes.sum()/n

fig,ax = plt.subplots(figsize=(15,10))
plt.plot(t, z, label='Had-CET monthly anomalies (standardised)', marker=".", color='lightgrey', lw=0.5)
tmin, tmax = ax.get_xlim()
plt.scatter(t[p10extremes], z[p10extremes], marker=".", color='blue', label='Extreme cases < P10 (exceedence fraction='+str(round(p10extremes_frac*100,2))+'%)')  
plt.scatter(t[p90extremes], z[p90extremes], marker=".", color='darkblue', label='Extreme cases > P90 (exceedence fraction='+str(round(p90extremes_frac*100,2))+'%)')           
plt.axhline(gamma_bins[90], tmin, tmax, color='darkblue', ls='--', label=r'$\Gamma$: P90')    
plt.axhline(gamma_bins[75], tmin, tmax, color='darkred', ls='--', label=r'$\Gamma$: P75 (Q3)')
plt.axhline(gamma_bins[50], tmin, tmax, color='black', ls='--', label=r'$\Gamma$: P50 (median)')
plt.axhline(gamma_bins[25], tmin, tmax, color='red', ls='--', label=r'$\Gamma$: P25 (Q1)')
plt.axhline(gamma_bins[10], tmin, tmax, color='blue', ls='--', label=r'$\Gamma$: P10')
plt.legend(fontsize=12)
plt.xlabel('Year', fontsize=fontsize)
plt.ylabel(r'Standard deviations, $\sigma$', fontsize=fontsize)
plt.ylim(-z_maxsigma,z_maxsigma)
plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.savefig('had-cet-gamma-fit-boxplot.png')

# PLOT: data CDF v gamma CDF

fig,ax = plt.subplots(figsize=(15,10))
plt.plot(z_map, gamma_cdf, color='grey', lw=0.5, label='Had-CET (standardised) CDF: n='+str(n))
plt.plot(z_map, z_cdf, color='teal', lw=0.5, label=r'$\Gamma$ CDF: n='+str(n))
plt.scatter(z_map100, gamma_cdf100, marker='o', color='grey', facecolor='lightgrey', lw=2, label='Had-CET (standardised) CDF: percentiles')
plt.scatter(z_map100, z_cdf100, marker='o', color='teal', facecolor='cyan', lw=2, label=r'$\Gamma$ CDF: percentiles')
plt.legend(fontsize=12)
plt.xlabel(r'Standard deviations, $\sigma$', fontsize=fontsize)
plt.ylabel('Cumulative Distribution Function (CDF)', fontsize=fontsize)
plt.xlim(-maxsigma,maxsigma)
plt.ylim(0,1)
ax.xaxis.grid(True, which='major')  
ax.yaxis.grid(True, which='major')  
plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.savefig('had-cet-gamma-fit-cdf.png')

