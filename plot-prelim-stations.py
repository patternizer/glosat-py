#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot-prelim.py
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

#------------------------------------------------------------------------------
# I/O: stat4.CRUTEM5.1prelim01.1721-2019.txt (text dump from CDF4)
#------------------------------------------------------------------------------

# load .txt file (comma separated) into pandas dataframe

filename_txt = 'stat4.CRUTEM5.1prelim01.1721-2019.txt'
#067250 464  -78 1538 BLATTEN, LOETSCHENTA SWITZERLAND   20012012  982001    9999
#1921  -44  -71  -68  -46  -12   17   42   53   27  -20  -21  -40

linelist = []
stationinfo = []
stationcode = []
stationcountry = []
with open (filename_txt, 'rt') as f:      
    for line in f:   
        if len(line)>1: # ignore empty lines         
            if (len(line.strip().split())!=13)|(len(line.split()[0])>4):                
                # when line is stationinfo extract stationid and stationname
                stationinfo.append(line.strip())   
                stationid = line.strip().split()[0][0:6]
                stationname = line.strip().split()[-4]
            else:                    
                # append monthly anomalies in [K] & stationid, stationname 
                linelist.append(line.strip())   
                stationcode.append(stationid)
                stationcountry.append(stationname)
        else:
            continue
f.close

# construct dataframe
df = pd.DataFrame(columns=['year','1','2','3','4','5','6','7','8','9','10','11','12','stationcode','stationcountry'])
for j in range(1,12+1):
    df[df.columns[j]] = [ int(linelist[i].split()[j]) for i in range(len(linelist)) ]
    fillval = -999
    df.replace(fillval, np.NaN, inplace=True) 
df['year'] = [ int(linelist[i].split()[0]) for i in range(len(linelist)) ]
df['stationcode'] = stationcode
df['stationcountry'] = stationcountry

# temperature conversions (if needed):
# for j in range(1,13):
#    df[df.columns[j]] = df[df.columns[j]] - 273.15          # K --> Centigrade
#    df[df.columns[j]] = (df[df.columns[j]]*(9/5.)) − 459.67 # K --> Fahrenheit

# PLOT: timeseries per station

#for j in range(len(np.unique(stationcode))):
for j in range(10):

    # form station timeseries
    da = df[df['stationcode']==np.unique(stationcode)[j]].iloc[:,range(0,13)]
    ts = []
    for i in range(len(da)):
        ts = ts + da.iloc[i,1:].to_list() 
    ts = np.array(ts)            
    t = pd.date_range(start='1933-01-01', periods=len(ts), freq='M')
#    db['Date'] = pd.date_range(start=str(da['Date'][0])[0:7], end=str(da['Date'].iloc[-1])[0:7])                                    
#    db['Date'] = pd.date_range(start='1900-01-01', end='2019-12-31', freq='M')                                    
    
#    da_melt = da.melt(id_vars='year').sort_values(by=['year']).reset_index()
#    del da_melt['index']

#    db = pd.DataFrame(columns=da.columns)
#    db['year'] = [ i for i in range(da_melt['year'].iloc[0], da_melt['year'].iloc[-1]+1)]
#    db_melt = db.melt(id_vars='year').sort_values(by=['year']).reset_index()
#    del db_melt['index']
#    db_melt.rename(columns={'year':'Year','variable':'Month','value':'Value'}, inplace = True)
#    db_melt['Day'] = 15
#    db_melt['Date'] = pd.to_datetime(db_melt[['Year','Month','Day']], format='%Y%m')      

    figstr = 'timeseries_' + np.unique(stationcode)[j] + '.png'
    titlestr = 'Monthly temperature anomaly timeseries: ' + np.unique(stationcode)[j]
          
    fig, ax = plt.subplots(figsize=(15,10))      
    plt.plot(t,ts)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close(fig)

# PLOT: seasonal cycle per station

#for j in range(len(np.unique(stationcode))):
for j in range(10):

    x = df[df['stationcode']==np.unique(stationcode)[j]].iloc[:,0]    
    Y = df[df['stationcode']==np.unique(stationcode)[j]].iloc[:,range(1,13)].T   
    figstr = 'seasonal-cycle_' + np.unique(stationcode)[j] + '.png'
    titlestr = 'Seasonal cycle: ' + np.unique(stationcode)[j]

    fig, ax = plt.subplots(figsize=(15,10))      
    plt.plot(np.arange(1,13),Y)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close(fig)

#------------------------------------------------------------------------------
print('** END')