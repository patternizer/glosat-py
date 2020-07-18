#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot-prelim-stripes.py
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
import pandas as pd
# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.cm as cm
from matplotlib import colors as mcol
from matplotlib.cm import ScalarMappable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

use_horizontal_colorbar = False
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

# Calculate 1961-1990 mean (cf: Ed's Climate Stripes: 1971-2000)
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
if use_horizontal_colorbar == True:
    cbar = plt.colorbar(sm, shrink=0.5, orientation='horizontal')
    cbar.set_label('Anomaly (from 1961-1990)', rotation=0, labelpad=25, fontsize=fontsize)
else:
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
if use_horizontal_colorbar == True:
    cbar = plt.colorbar(sm, shrink=0.5, orientation='horizontal')
    cbar.set_label('Mean annual anomaly (from 1961-1990)', rotation=0, labelpad=25, fontsize=fontsize)
else:
    cbar = plt.colorbar(sm, shrink=0.5)
    cbar.set_label('Anomaly (from 1961-1990)', rotation=270, labelpad=25, fontsize=fontsize)
plt.title('Mean annual anomaly: Global', fontsize=fontsize)
plt.savefig(filename_txt + '_anomaly-stripes.png')
plt.close(fig)

#------------------------------------------------------------------------------
print('** END')