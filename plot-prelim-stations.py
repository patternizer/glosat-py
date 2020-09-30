#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: plot-prelim-stations.py
#------------------------------------------------------------------------------
# Version 0.15
# 28 September, 2020
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
import klib
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
lat_start = -90;  lat_end = 90
lon_start = -180; lon_end = 180
station_start=0;  station_end=10

load_df_temp = True
load_df_anom = True
load_df_norm = True
load_df_normals = True
plot_fry = False
plot_spiral = False
plot_stripes = False
plot_klib = False
plot_temporal_change = False
plot_temporal_coverage = False
plot_spatial_coverage = False
plot_seasonal_anomalies = False
plot_station_timeseres = False
plot_station_climatology = False
plot_station_locations = False
plot_delta_cc = False; delta_cc_20C = True    
plot_gap_analysis = True; station_count = True

#------------------------------------------------------------------------------
# METHODS
#------------------------------------------------------------------------------

def load_dataframe(filename_txt):
    
    #------------------------------------------------------------------------------
    # I/O: filename.txt (text version)
    #------------------------------------------------------------------------------

    # load .txt file (comma separated) into pandas dataframe

#    filename_txt = 'stat4.GloSATprelim02.1658-2020.txt'
#    filename_txt = 'stat4.CRUTEM5.1prelim01.1721-2019.txt'    
#    filename_txt = 'GloSATprelim01.1721-2019.txt'
    
    yearlist = []
    monthlist = []
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
                else:           
                    yearlist.append(int(line.strip().split()[0]))                               
                    monthlist.append(np.array(line.strip().split()[1:]))                                 
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
#    df['stationcruindex'] = stationcruindex
#    df['stationgridcell'] = stationgridcell

    # trim strings
    
    df['stationname'] = [ str(df['stationname'][i]).strip() for i in range(len(df)) ] 
    df['stationcountry'] = [ str(df['stationcountry'][i]).strip() for i in range(len(df)) ] 

    # convert numeric types to int (important due to fillValue)

    for j in range(1,13):
        df[df.columns[j]] = df[df.columns[j]].astype('int')

    df['stationlat'] = df['stationlat'].astype('int')
    df['stationlon'] = df['stationlon'].astype('int')
    df['stationelevation'] = df['stationelevation'].astype('int')
    df['stationfirstreliable'] = df['stationfirstreliable'].astype('int')

    # error handling

#    for i in range(len(df)):        
#        if str(df['stationcruindex'][i])[1:].isdigit() == False:
#            df['stationcruindex'][i] = np.nan
#        else:
#            continue
 
    # replace fill values in int variables

    # -999 for stationlat
    # -9999 for stationlon
    # -9999 for station elevation
    # (some 999's occur elsewhere - fill all bad numeric cases with NaN)

    for j in range(1,13):    
        df[df.columns[j]].replace(-999, np.nan, inplace=True)

    df['stationlat'].replace(-999, np.nan, inplace=True) 
    df['stationlon'].replace(-9999, np.nan, inplace=True) 
    df['stationelevation'].replace(-9999, np.nan, inplace=True) 
#   df['stationfirstreliable'].replace(8888, np.nan, inplace=True)      
    
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

if load_df_temp == True:

    print('loading temperatures ...')

    df_temp = pd.read_pickle('df_temp.pkl', compression='bz2')    
    
    #------------------------------------------------------------------------------
    # EXTRACT TARBALL IF df.csv IS COMPRESSED:
    #------------------------------------------------------------------------------

#   filename = Path("df_temp.csv")
#    if not filename.is_file():
#        print('Uncompressing filename from tarball ...')
#        filename = "df_temp.tar.bz2"
#        subprocess.Popen(['tar', '-xjvf', filename])  # = tar -xjvf df_temp.tar.bz2
#        time.sleep(5) # pause 5 seconds to give tar extract time to complete prior to attempting pandas read_csv
#   df_temp = pd.read_csv('df_temp.csv', index_col=0)

else:    
    
    print('read stat4.GloSATprelim02.1658-2020.txt ...')

    filename_txt = 'stat4.GloSATprelim02.1658-2020.txt'
    df = load_dataframe(filename_txt)

    #------------------------------------------------------------------------------
    # ADD LEADING 0 TO STATION CODES (str)
    #------------------------------------------------------------------------------

    df['stationcode'] = [ str(df['stationcode'][i]).zfill(6) for i in range(len(df)) ]

    #------------------------------------------------------------------------------
    # APPLY SCALE FACTORS
    #------------------------------------------------------------------------------
    
    print('apply scale factors ...')

    df['stationlat'] = df['stationlat']/10.0
    df['stationlon'] = df['stationlon']/10.0

    for j in range(1,13):

        df[df.columns[j]] = df[df.columns[j]]/10.0

    #------------------------------------------------------------------------------
    # CONVERT LONGITUDE FROM +W TO +E
    #------------------------------------------------------------------------------

    print('convert longitudes ...')

    df['stationlon'] = -df['stationlon']     
        
    #------------------------------------------------------------------------------
    # CONVERT DTYPES FOR EFFICIENT STORAGE
    #------------------------------------------------------------------------------

    df['year'] = df['year'].astype('int16')

    for j in range(1,13):

        df[df.columns[j]] = df[df.columns[j]].astype('float32')

    df['stationlat'] = df['stationlat'].astype('float32')
    df['stationlon'] = df['stationlon'].astype('float32')
    df['stationelevation'] = df['stationelevation'].astype('int16')
    df['stationfirstyear'] = df['stationfirstyear'].astype('int16')
    df['stationlastyear'] = df['stationlastyear'].astype('int16')    
    df['stationsource'] = df['stationsource'].astype('int8')    
    df['stationfirstreliable'] = df['stationfirstreliable'].astype('int16')

    #------------------------------------------------------------------------------
    # SAVE TEMPERATURES
    #------------------------------------------------------------------------------

    print('save temperatures ...')

    df_temp = df.copy()
    df_temp.to_pickle('df_temp.pkl', compression='bz2')

#------------------------------------------------------------------------------
# LOAD Normals, SDs and FRYs
#------------------------------------------------------------------------------

if load_df_normals == True:

    print('loading normals ...')

    df_normals = pd.read_pickle('df_normals.pkl', compression='bz2')    

else:

    print('extracting normals ...')

    file = 'normals5.GloSAT.prelim02_FRYuse_ocPLAUS1_iqr3.600reg0.3_19411990_MIN15_OCany_19611990_MIN15_PERDEC00_NManySDreq.txt'
#    file = 'normals5.GloSAT.prelim01_FRYuse_ocPLAUS1_iqr3.600reg0.3_19411990_MIN15_OCany_19611990_MIN15_PERDEC00_NManySDreq.txt'
#    file = 'sd5.GloSAT.prelim01_FRYuse_ocPLAUS1_iqr3.600reg0.3_19411990_MIN15_OCany_19611990_MIN15_PERDEC00_NManySDreq.txt'

#    1 ID
#    2-3 First year, last year
#    4-5 First year for normal, last year for normal
#    6-17 Twelve normals for Jan to Dec, degC (not degC*10), missing normals are -999.000
#    18 Source code (1=missing, 2=Phil's estimate, 3=WMO, 4=calculated here, 5=calculated previously)
#    19-30 Twelve values giving the % data available for computing each normal from the 1961-1990 period

    f = open(file)
    lines = f.readlines()

    stationcodes = []
    firstyears = []
    lastyears = []
    normalfirstyears = []
    normallastyears = []
    sourcecodes = []
    normal12s = []
    normalpercentage12s = []
    
    for i in range(len(lines)):
        words = lines[i].split()
        if len(words) > 1:
            stationcode = words[0].zfill(6)
            firstyear = int(words[1])
            lastyear = int(words[2])
            normalfirstyear = int(words[3])
            normallastyear = int(words[4])
            sourcecode = int(words[17])
            normal12 = words[5:17]
            normalpercentage12 = words[18:31]

            stationcodes.append(stationcode)
            firstyears.append(firstyear)
            lastyears.append(lastyear)
            normalfirstyears.append(normalfirstyear)
            normallastyears.append(normallastyear)
            sourcecodes.append(sourcecode)
            normal12s.append(normal12)
            normalpercentage12s.append(normalpercentage12)
    f.close()
    
    normal12s = np.array(normal12s)
    normalpercentage12s = np.array(normalpercentage12s)

    df_normals = pd.DataFrame({
        'stationcode':stationcodes,
        'firstyear':firstyears, 
        'lastyear':lastyears, 
        'normalfirstyear':normalfirstyears, 
        'normallastyear':normallastyears, 
        'sourcecode':sourcecodes,
        '1':normal12s[:,0],                               
        '2':normal12s[:,1],                               
        '3':normal12s[:,2],                               
        '4':normal12s[:,3],                               
        '5':normal12s[:,4],                               
        '6':normal12s[:,5],                               
        '7':normal12s[:,6],                               
        '8':normal12s[:,7],                               
        '9':normal12s[:,8],                               
        '10':normal12s[:,9],                               
        '11':normal12s[:,10],                               
        '12':normal12s[:,11],                               
        '1pc':normalpercentage12s[:,0],                               
        '2pc':normalpercentage12s[:,1],                               
        '3pc':normalpercentage12s[:,2],                               
        '4pc':normalpercentage12s[:,3],                               
        '5pc':normalpercentage12s[:,4],                               
        '6pc':normalpercentage12s[:,5],                               
        '7pc':normalpercentage12s[:,6],                               
        '8pc':normalpercentage12s[:,7],                               
        '9pc':normalpercentage12s[:,8],                               
        '10pc':normalpercentage12s[:,9],                               
        '11pc':normalpercentage12s[:,10],                               
        '12pc':normalpercentage12s[:,11],                                                              
    })   

    # Filter out stations with missing normals

#    df_normals_copy = df_normals.copy()
#    df_normals = df_normals_copy[df_normals_copy['sourcecode']>1]

    print('save normals ...')

    df_normals.to_pickle('df_normals.pkl', compression='bz2')
        
#------------------------------------------------------------------------------
# CALCULATE 1961-1990 baselines and anomalies
#------------------------------------------------------------------------------

if load_df_anom == True:

    print('loading anomalies ...')

    df_anom = pd.read_pickle('df_anom.pkl', compression='bz2')    

else:
    
    print('calculate baselines and anomalies ...')

    df_anom = df_temp.copy()
    for i in range(len(df_anom['stationcode'].unique())):
        da = df_anom[df_anom['stationcode']==df_anom['stationcode'].unique()[i]]
        for j in range(1,13):
            baseline = np.nanmean(np.array(da[(da['year']>=1961) & (da['year']<=1990)].iloc[:,j]).ravel())
            df_anom.loc[da.index.tolist(), str(j)] = da[str(j)]-baseline            

    print('save anomalies ...')

    df_anom.to_pickle('df_anom.pkl', compression='bz2')

#------------------------------------------------------------------------------
# NORMALIZE TIMESERIES
#------------------------------------------------------------------------------
    
if load_df_norm == True:

    print('loading normalized anomalies ...')

    df_norm = pd.read_pickle('df_norm.pkl', compression='bz2')    
#   df_norm['stationcode'] = df_norm['stationcode'].str.zfill(6)
    
else:

    print('normalize anomalies ...')

    # Adjust anomaly timeseries using normalisation for each month
        
    df_norm = df_anom.copy()              
    for i in range(len(df_norm['stationcode'].unique())):            
        da = df_norm[df_norm['stationcode']==df_norm['stationcode'].unique()[i]]
        for j in range(1,13):
            df_norm.loc[da.index.tolist(), str(j)] = (da[str(j)]-da[str(j)].dropna().mean())/da[str(j)].dropna().std()

    print('save normalized anomalies ...')

    df_norm.to_pickle('df_norm.pkl', compression='bz2')

#------------------------------------------------------------------------------
# Fix decimal degree errors
#------------------------------------------------------------------------------

#df_temp_copy = df_temp.copy()    
#df_temp_copy['stationlon'][df_temp_copy['stationcode']=='037641'] = -0.9
#df_temp = df_temp_copy
#df_temp.to_pickle('df_temp.pkl', compression='bz2')

#df_anom_copy = df_anom.copy()    
#df_anom_copy['stationlon'][df_anom_copy['stationcode']=='037641'] = -0.9
#df_anom = df_anom_copy
#df_anom.to_pickle('df_anom.pkl', compression='bz2')

#df_norm_copy = df_norm.copy()    
#df_norm_copy['stationlon'][df_norm_copy['stationcode']=='037641'] = -0.9
#df_norm = df_norm_copy
#df_norm.to_pickle('df_norm.pkl', compression='bz2')

#------------------------------------------------------------------------------
# REPLACE FRY FillValue WITH FIRSTYEAR
#------------------------------------------------------------------------------

if plot_fry == True:

    print('plot_fry ...')
    
    ds = df_temp.copy()

    stationfirstyears = ds.groupby('stationcode')['stationfirstyear'].mean()
    stationfirstreliables = ds.groupby('stationcode')['stationfirstreliable'].mean()
    stationcodes = stationfirstyears.index

    for i in range(len(ds['stationcode'].unique())):
        if stationfirstreliables[i]>2020:
            ds.loc[ds['stationcode']==stationcodes[i], 'stationfirstreliable'] = stationfirstyears[i]
                     
    # PLOT FRY DIFF

    fig,ax = plt.subplots()
    stationfirstyears.plot(lw=0.5, color='lightgrey', label='post-FRY')
    stationfirstreliables.plot(lw=0.5, color='teal', label='pre-FRY')
    #stationfirstreliables.plot(marker='.', markersize=0.5, color='black', label='FRY')
    #ax.fill_between(stationcodes.astype(int), stationfirstyears, stationfirstreliables, facecolor='blue', alpha=0.5)
    plt.legend()
    plt.savefig('FRY1.png')

    # PLOT FRY TEST
        
    da = ds[ds['stationcode']==stationcodes[8531]].iloc[:,range(0,13)]
    fry = ds[ds['stationcode']==stationcodes[8531]]['stationfirstreliable'].unique().astype(int)[0]
    fry = 1900

    ts_monthly = []    
    for i in range(len(da)):                
        monthly = da.iloc[i,1:]
        ts_monthly = ts_monthly + monthly.to_list()    
    ts_monthly = np.array(ts_monthly)   
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')    

    fig,ax = plt.subplots()
    plt.plot(t_monthly,ts_monthly, ls='-', marker='.', color='teal', label='post-FRY')
    p = plt.plot(t_monthly[t_monthly.year<=fry], ts_monthly[t_monthly.year<=fry], ls='-', marker='.', color='lightgrey', label='pre-FRY')
    ymin, ymax = ax.get_ylim()    
    plt.axvline(pd.Timestamp(str(fry)), ls='--', color='black', label='FRY')
    plt.legend()
    plt.savefig('FRY2.png')

#------------------------------------------------------------------------------
# Climate Spiral
# http://www.climate-lab-book.ac.uk/spirals/
#------------------------------------------------------------------------------

if plot_spiral == True:

    print('plot_spiral ...')

    ds = df_anom.copy()

    value = 7939, # Death Valley

    """
    Plot station spiral using monthly anomalies
    """
    
    da = ds[ds['stationcode']==ds['stationcode'].unique()[value]].iloc[:,range(0,13)]
    baseline = np.nanmean(np.array(da[(da['year']>=1850)&(da['year']<=1900)].groupby('year').mean()).ravel())    
    ts_monthly = np.array(da.iloc[:,1:13]).ravel() - baseline             
    mask = np.isfinite(ts_monthly)
    ts_monthly_min = ts_monthly[mask].min()    
    ts_monthly = ts_monthly - ts_monthly_min    
    
    n = len(da)
    colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
    hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]
    
    fig = plt.figure(figsize=(14,14))
    ax1 = plt.subplot(111, projection='polar')
    ax1.axes.get_yaxis().set_ticklabels([])
    ax1.axes.get_xaxis().set_ticklabels([])
    fig.set_facecolor("#323331")

    for i in range(len(da)):
        r = np.array(da[da['year']==da.iloc[i][0].astype('int')].iloc[:,1:13]).ravel() - baseline - ts_monthly_min            
        theta = np.linspace(0, 2*np.pi, 12)
        ax1.grid(False)
#       ax1.set_title("Global Temperature Change ("+str(da.iloc[0][0].astype('int'))+"-"+str(da.iloc[-1][0].astype('int'))+")", color='white', fontdict={'fontsize':fontsize})
        ax1.set_ylim(0, 15)
        ax1.patch.set_facecolor('#000100')
        ax1.text(0,0, str(da.iloc[0][0].astype('int')), color='white', size=30, ha='center')            
        ax1.plot(theta, r, c=hexcolors[i])
    ax1.text(theta[np.isfinite(r)][-1], r[np.isfinite(r)][-1], str(da.iloc[0][0].astype('int')), color='white', size=30, ha='center')            

#   full_circle_thetas = np.linspace(0, 2*np.pi, 1000)  
#   blue_line_one_radii = [1.0]*1000
#   red_line_one_radii = [2.5]*1000
#   red_line_two_radii = [3.0]*1000
#   ax1.plot(full_circle_thetas, blue_line_one_radii, c='blue')
#   ax1.plot(full_circle_thetas, red_line_one_radii, c='red')
#   ax1.plot(full_circle_thetas, red_line_two_radii, c='red')
#   ax1.text(np.pi/2, 1.0, "0.0 C", color="blue", ha='center', fontdict={'fontsize': 20})
#   ax1.text(np.pi/2, 2.5, "1.5 C", color="red", ha='center', fontdict={'fontsize': 20})
#   ax1.text(np.pi/2, 3.0, "2.0 C", color="red", ha='center', fontdict={'fontsize': 20})
    plt.savefig('climate-spiral.png')

#------------------------------------------------------------------------------
# Climate Stripes
# https://showyourstripes.info/
#------------------------------------------------------------------------------
    
if plot_stripes == True:

    print('plot_stripes ...')

    ds = df_anom.copy()    

    value = 7939, # Death Valley

    # Calculate 1961-1990 mean (cf: Ed's Climate Stripes: 1971-2000)

    da = ds[ds['stationcode']==ds['stationcode'].unique()[value]]
    monthly = np.array(da.iloc[:,1:13]).ravel()
             
    yearly=(da.groupby('year').mean().iloc[:,0:12]).dropna().mean(axis=1)    
#   yearly = np.nanmean(np.array(da.groupby('year').mean().iloc[:,0:12]),axis=1)                
    mu = np.nanmean(np.array(da[(da['year']>=1961) & (da['year']<=1990)].iloc[:,1:13].dropna()).ravel())

    # Compute 1900-2000 standard deviation: color range +/- 2.6 standard deviations 

#   sigma = np.nanstd(yearly[(yearly.index>=1900)&(yearly.index<=2000)])
    sigma = np.nanstd(yearly)
    
    x = yearly.index
    y = yearly
    z = (y-mu)*0.0+1.0
    cmap = plt.cm.get_cmap('coolwarm')
#   cmap = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
    maxval = +2.6 * sigma
    minval = -2.6 * sigma

    n = len(y)            
    colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
    hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]
    colors = cmap((y-mu)/maxval+0.5)

    fig, ax = plt.subplots(figsize=(15,10))
#   ax.bar(x, z, color=colors, width=1.0)    
    ax.bar(x, y-mu, color=colors)    
    ax.axis('off')
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(min(y-mu),max(y-mu)))
    sm.set_array([])
    use_horizontal_colorbar = False
    if use_horizontal_colorbar == True:
        cbar = plt.colorbar(sm, shrink=0.5, orientation='horizontal')
        cbar.set_label('Mean annual anomaly (from 1961-1990)', rotation=0, labelpad=25, fontsize=fontsize)
    else:
        cbar = plt.colorbar(sm, shrink=0.5)
        cbar.set_label('Anomaly (from 1961-1990)', rotation=270, labelpad=25, fontsize=fontsize)
    plt.title('station-stripes', fontsize=fontsize)
    plt.savefig('station-stripes.png')
    plt.close(fig)

#------------------------------------------------------------------------------
# KLIB: data summary:
# https://github.com/akanz1/klib    
#------------------------------------------------------------------------------

if plot_klib == True:

    print('plot klib temperature summaries ...')

    ds = df_temp.copy()
    
    # PLOT: missing values

    fig = plt.figure(1,figsize=(15,10))
    klib.missingval_plot(ds)
    plt.savefig('klib-dataset-summary.png')
    plt.close(fig)

    # display data types and most efficient representation
    
    ds_cleaned = klib.data_cleaning(ds)
    print(ds.info(memory_usage='deep'))
    print(ds_cleaned.info(memory_usage='deep'))

    # PLOT: correlations (and separated (pos)itive and (neg)ative correlations
    
    fig = plt.figure(1,figsize=(15,10))
#   klib.corr_plot(ds_cleaned, split='pos', annot=False)
#   klib.corr_plot(ds_cleaned, split='neg', annot=False)
#   klib.corr_plot(ds_cleaned, target='year')
    klib.corr_plot(ds_cleaned, annot=False)
    plt.savefig('klib-correlation.png')
    plt.close(fig)

    # PLOT: distributions

#   fig = plt.figure(1,figsize=(15,10))
#   klib.dist_plot(ds_cleaned)    
#   plt.savefig('klib-distribution.png')
#   plt.close(fig)

    # PLOT: categoricals

#   fig = plt.figure(1,figsize=(15,10))
#   klib.cat_plot(ds_cleaned, top=10, bottom=10)    
#   plt.savefig('klib-categorical.png')
#   plt.close(fig)

#------------------------------------------------------------------------------
# PLOT: Anomaly timeseries by month
#------------------------------------------------------------------------------

if plot_seasonal_anomalies == True:

    print('plot_seasonal_anomalies ...')
    
    ds = df_anom.copy()
    
    # PLOT: anomaly timeseries for each month of year:

    figstr = 'annual-mean-anomaly-by-month.png'
    titlestr = 'Annual mean temperature anomaly (from 1961-1990) by month'    

    n = 12            
    colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
    hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]
                               
    def plot_timeseries(y,i,row,col):            
        axs[row,col].plot(y,color=hexcolors[i], label='Month=' + str(i+1))
        axs[row,col].set_title('Month=' + "{0:.0f}".format(i+1), fontsize=15)
       
    fig, axs = plt.subplots(4,3,figsize=(15,10))
    plot_timeseries(ds.groupby('year').mean().iloc[:,0],0,0,0)
    plot_timeseries(ds.groupby('year').mean().iloc[:,1],1,0,1)
    plot_timeseries(ds.groupby('year').mean().iloc[:,2],2,0,2)
    plot_timeseries(ds.groupby('year').mean().iloc[:,3],3,1,0)
    plot_timeseries(ds.groupby('year').mean().iloc[:,4],4,1,1)
    plot_timeseries(ds.groupby('year').mean().iloc[:,5],5,1,2)
    plot_timeseries(ds.groupby('year').mean().iloc[:,6],6,2,0)
    plot_timeseries(ds.groupby('year').mean().iloc[:,7],7,2,1)
    plot_timeseries(ds.groupby('year').mean().iloc[:,8],8,2,2)
    plot_timeseries(ds.groupby('year').mean().iloc[:,9],9,3,0)
    plot_timeseries(ds.groupby('year').mean().iloc[:,10],10,3,1)
    plot_timeseries(ds.groupby('year').mean().iloc[:,11],11,3,2)
    for ax in axs.flat:
        ax.set_xlabel("Year", fontsize=15)
        ax.set_ylabel("Anomaly, " + "$\mathrm{\degree}$C", fontsize=15)
    for ax in axs.flat:
        ax.set_ylim([-5,5])
        ax.yaxis.grid(True, which='major')                
        ax.label_outer()
#       ax.legend(loc='best', fontsize=8)           
    fig.suptitle(titlestr, fontsize=fontsize)        
    plt.savefig(figstr)
    plt.close(fig)
    
#--------------------------------------------------------------------------------
# PLOT: station coverage - thanks to Kate Willett (UKMO) for spec. humidity code:
# https://github.com/Kate-Willett
#--------------------------------------------------------------------------------

if plot_gap_analysis == True:

    print('plot_gap_analysis ...')
    
    ds = df_anom.copy()

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
    stationlist = ds['stationcode'].unique().astype('int')

    stationcolours = []
    for i in range(len(labellist)):        
        labelcount = ((stationlist>=labellist[i][1]) & (stationlist<=labellist[i][2])).sum()        
        stationcolours.append(np.tile(colourlist[i],labelcount))
    stationcolours = list(itertools.chain.from_iterable(stationcolours))
                             
    if station_count == True:

        figstr = 'gap-analysis-plus-annual-station-count.png'
        titlestr = 'Gap analysis and annual station count'    
        xstr = 'Year'
        y1str = 'Station ID'
        y2str = 'Annual station count'

        # Contruct timeseries of yearly station count

        station_yearly_count = ds.groupby('year')['stationcode'].count()
        # Solve Y1677-Y2262 Pandas bug with XR: 
        # t_station_yearly_count = pd.date_range(start=str(ds['year'].min()), periods=len(station_yearly_count), freq='A')        
        t_station_yearly_count = xr.cftime_range(start=str(ds['year'].min()), periods=len(station_yearly_count), freq='A', calendar="noleap")
        T = t_station_yearly_count.year
        Y = station_yearly_count

    else:

        figstr = 'gap-analysis-plus-annual-mean-anomaly.png'
        titlestr = 'Gap analysis and annual mean temperature anomaly (from 1961-1990)'    
        xstr = 'Year'
        y1str = 'Station ID'
        y2str = "Annual mean temperature anomaly, " + "$\mathrm{\degree}$C"

        # Contruct timeseries of yearly mean and s.d.

        global_yearly_mean = []
        for i in range(ds['year'].min(),ds['year'].max()+1):                    
            yearly_mean = np.nanmean(np.array(ds[ds['year']==i].iloc[:,range(1,13)]).ravel())                                     
            global_yearly_mean.append(yearly_mean)

        # Solve Y1677-Y2262 Pandas bug with XR: 
        # t_yearly = pd.date_range(start=str(ds['year'].min()), periods=len(global_yearly_mean), freq='A')
        t_yearly = xr.cftime_range(start=str(ds['year'].min()), periods=len(global_yearly_mean), freq='A', calendar="noleap")

        # Contruct timeseries of monthly mean and s.d.

        # global_monthly_mean = []
        # global_monthly_std = []
        # for i in range(ds['year'].min(),ds['year'].max()+1):            
        #     for j in range(1,13):
        #         monthly_mean = np.nanmean(ds[ds['year']==i][str(j)])                                          
        #         monthly_std = np.nanstd(ds[ds['year']==i][str(j)])                                          
        #         global_monthly_mean.append(monthly_mean)
        #         global_monthly_std.append(monthly_std)
        # t_monthly = pd.date_range(start=str(ds['year'].min()), periods=len(global_monthly_mean), freq='M')
        
        T = t_yearly.year
        Y = global_yearly_mean

    # PLOT: timeseries of monthly obs per station
            
    def find_gaps(ds,i):

        da = ds[ds['stationcode']==ds['stationcode'].unique()[i]].iloc[:,range(0,13)]
        ts = np.array([ da.iloc[i,1:].to_list() for i in range(len(da)) ]).ravel()  
        # Solve Y1677-Y2262 Pandas bug with XR: 
        # t = pd.date_range(start=str(da['year'].iloc[0]), periods=len(da)*12, freq='M')
        t = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(da)*12, freq='M', calendar="noleap")          
        mask = np.isfinite(ts)
        x = t[mask]
        y = np.ones(len(ts))[mask]*int(ds['stationcode'].unique()[i])

        return x,y

    df_gaps = pd.DataFrame(columns=['x','y','stationcode'])

    fig = plt.figure(1,figsize=(15,10))
    plt.clf() # clear plot space
    ax1=plt.subplot(1,1,1)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
                
    for i in range(len(ds['stationcode'].unique())):            
            
        x,y = find_gaps(ds,i)  
        X = [ datetime(x[j].year,x[j].month,x[j].day) for j in range(len(x)) ]          
#        ax1.plot(X, y, color=stationcolours[i], linewidth=1)    
        
        da = xr.DataArray(y, coords=[('time', x)])
        da.plot(color=stationcolours[i], linewidth=1)        
#        df_gaps = df_gaps.append({'x':x, 'y':y, 'stationcode':ds['stationcode'].unique()[i]}, ignore_index=False)
    
    ax1.set_title(titlestr,size=fontsize)
    ax1.set_xlabel(xstr,size=fontsize)
    ax1.set_ylabel(y1str,size=fontsize)    
    ax2=ax1.twinx()
    ax2.set_ylabel(y2str,size=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)    
    ax2.plot(T, Y, 'black', linewidth=2)
    plt.savefig(figstr)
    plt.close(fig)
                    
    print('save gaps ...')

    df_gaps.to_pickle('df_gaps.pkl', compression='bz2')
        
if plot_temporal_change == True:

    print('plot_temporal_change...')

    ds = df_temp.copy()    
    
    #------------------------------------------------------------------------------
    # PLOT: temporal change by region: at request of Martin Stendel
    #------------------------------------------------------------------------------

    nstations = len(ds['stationcode'].unique())
    nbins = ds['year'].max() - ds['year'].min() + 1
    bins = np.linspace(ds['year'].min(), ds['year'].max(), nbins) 
    counts, edges = np.histogram(ds['year'], nbins, range=[ds['year'].min(),ds['year'].max()], density=False)    
    Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
    Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]

    figstr = 'temporal-change-full.png'
    titlestr = 'GloSATp02 yearly coverage (' + str(ds['year'].min()) + '-' +  str(ds['year'].max()) + '): N(obs)=' + "{0:.0f}".format(np.sum(counts)) + ', N(stations)=' + "{0:.0f}".format(nstations) 
             
    fig, ax = plt.subplots(figsize=(15,10))          
    plt.fill_between(bins, counts, step="pre", facecolor='lightgrey', alpha=1.0)
    plt.axvline(x=Q1, color='blue', label='Q1: ' + "{0:.0f}".format(Q1))    
    plt.axvline(x=Q2, color='red', label='Q2: ' + "{0:.0f}".format(Q2))    
    plt.axvline(x=Q3, color='blue', label='Q3: ' + "{0:.0f}".format(Q3))    
    plt.tick_params(labelsize=12)    
    plt.legend(loc='upper left', fontsize=12)
    plt.xlabel("Year", fontsize=fontsize)
    plt.ylabel("Monthly values per year", fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close(fig)
        
    figstr = 'temporal-change-latitude-from-1950.png'
    titlestr = 'GloSATp02 yearly coverage (1950-2020) by latitudinal band'

    step = 15
    bands = np.arange(-90,90+step,step=step)
    index = pd.Index(bins.astype(int), name='date')
    data = []
    dh = pd.DataFrame(data, index=index)   
    
    for i in range(len(bands)-1):        
        bandi = ds[ (ds['stationlat']>=bands[i]) & (ds['stationlat']<=bands[i+1]) ]
        nstationsi = len(bandi['stationcode'].unique())
        countsi, edgesi = np.histogram(bandi['year'], nbins, range=[ds['year'].min(),bandi['year'].max()], density=False)        
        if i == 0:
            labeli = '('+str(bands[i])+','+str(bands[i+1])+') n(stations)='+str(nstationsi)
        else:
            labeli = '('+str(bands[i]+1)+','+str(bands[i+1])+') n(stations)='+str(nstationsi)
        dh[labeli] = tuple(countsi)
            
    ax = dh[dh.index>=1950].plot(kind='bar', stacked=True, figsize=(15, 10))
    ax.set_xlabel("Year", fontsize=fontsize)
    ax.set_ylabel("Monthly values per year", fontsize=fontsize)
    plt.tick_params(labelsize=12)    
    plt.legend(loc='upper left', fontsize=12)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close(fig)

    figstr = 'temporal-change-full-latitude-full2.png'
    titlestr = 'GloSATp02 yearly coverage (' + str(ds['year'].min()) + '-' + str(ds['year'].max()) + ') by latitudinal band'

    fig, ax = plt.subplots(figsize=(15,10))          
    
    for i in range(len(bands)-1):        
        bandi = ds[ (ds['stationlat']>=bands[i]) & (ds['stationlat']<=bands[i+1]) ]
        nstationsi = len(bandi['stationcode'].unique())
        binsi = bins
        countsi, edgesi = np.histogram(bandi['year'], nbins, range=[ds['year'].min(),bandi['year'].max()], density=False)        
        if i == 0:
            labeli = '('+str(bands[i])+','+str(bands[i+1])+') n(stations)='+str(nstationsi)
        else:
            labeli = '('+str(bands[i]+1)+','+str(bands[i+1])+') n(stations)='+str(nstationsi)
        plt.plot(binsi, countsi, label=labeli)
#       plt.fill_between(binsi, countsi, step="pre", alpha=0.2, label=labeli)

    ax.set_xlabel("Year", fontsize=fontsize)
    ax.set_ylabel("Monthly values per year", fontsize=fontsize)
#    plt.xlim([1900,2020])
#    ax.xaxis.grid(True, which='major')      
#    ax.yaxis.grid(True, which='major')      
    plt.tick_params(labelsize=12)    
    plt.legend(loc='upper left', fontsize=12)
#   plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close(fig)
    
#------------------------------------------------------------------------------
# PLOT: delta CC +/- 30 years around median date value
#------------------------------------------------------------------------------
            
if plot_delta_cc == True:
        
    #------------------------------------------------------------------------------
    # EXTRACT TARBALL IF df_norm.csv IS COMPRESSED:
    #------------------------------------------------------------------------------

    ds = df_norm.copy()
        
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

#    nbins = 1900 - ds['year'].min() + 1
#    bins = np.linspace(ds['year'].min(), 1900, nbins) 
#    counts, edges = np.histogram(ds['year'], nbins, range=[ds['year'].min(),1900], density=False)        
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
        
    delta_lo_60N90N = ds[(ds['year']>=year_lo) & (ds['year']<=year_Q2) & (ds['stationlat']>60) & (ds['stationlat']<=90)].groupby(['stationcode']).mean()
    delta_hi_60N90N = ds[(ds['year']>year_Q2) & (ds['year']<=year_hi) & (ds['stationlat']>60) & (ds['stationlat']<=90)].groupby(['stationcode']).mean()
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

    delta_lo_30N60N = ds[(ds['year']>=year_lo) & (ds['year']<=year_Q2) & (ds['stationlat']>30) & (ds['stationlat']<=60)].groupby(['stationcode']).mean()
    delta_hi_30N60N = ds[(ds['year']>year_Q2) & (ds['year']<=year_hi) & (ds['stationlat']>30) & (ds['stationlat']<=60)].groupby(['stationcode']).mean()
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

    delta_lo_00N30N = ds[(ds['year']>=year_lo) & (ds['year']<=year_Q2) & (ds['stationlat']>0) & (ds['stationlat']<=30)].groupby(['stationcode']).mean()
    delta_hi_00N30N = ds[(ds['year']>year_Q2) & (ds['year']<=year_hi) & (ds['stationlat']>0) & (ds['stationlat']<=30)].groupby(['stationcode']).mean()
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

    delta_lo_00S30S = ds[(ds['year']>=year_lo) & (ds['year']<=year_Q2) & (ds['stationlat']<=0) & (ds['stationlat']>=-30)].groupby(['stationcode']).mean()
    delta_hi_00S30S = ds[(ds['year']>year_Q2) & (ds['year']<=year_hi) & (ds['stationlat']<=0) & (ds['stationlat']>=-30)].groupby(['stationcode']).mean()
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

    delta_lo_30S60S = ds[(ds['year']>=year_lo) & (ds['year']<=year_Q2) & (ds['stationlat']<-30) & (ds['stationlat']>=-60)].groupby(['stationcode']).mean()
    delta_hi_30S60S = ds[(ds['year']>year_Q2) & (ds['year']<=year_hi) & (ds['stationlat']<-30) & (ds['stationlat']>=-60)].groupby(['stationcode']).mean()
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

    delta_lo_60S90S = ds[(ds['year']>=year_lo) & (ds['year']<=year_Q2) & (ds['stationlat']<-60) & (ds['stationlat']>=-90)].groupby(['stationcode']).mean()
    delta_hi_60S90S = ds[(ds['year']>year_Q2) & (ds['year']<=year_hi) & (ds['stationlat']<-60) & (ds['stationlat']>=-90)].groupby(['stationcode']).mean()
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

    delta_lo_global = ds[(ds['year']>=year_lo) & (ds['year']<=year_Q2)].groupby(['stationcode']).mean()
    delta_hi_global = ds[(ds['year']>year_Q2) & (ds['year']<=year_hi)].groupby(['stationcode']).mean()
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

    delta_lo_us = ds[(ds['year']>=year_lo) & (ds['year']<=year_Q2) & ((ds['stationcountry']=='USA') | (ds['stationcountry']=='USA---------'))].groupby(['stationcode']).mean()
    delta_hi_us = ds[(ds['year']>year_Q2) & (ds['year']<=year_hi) & ((ds['stationcountry']=='USA') | (ds['stationcountry']=='USA---------'))].groupby(['stationcode']).mean()
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

    delta_lo_australia = ds[(ds['year']>=year_lo) & (ds['year']<=year_Q2) & ((ds['stationcountry']=='AUSTRALIA') | (ds['stationcountry']=='AUSTRALIA---'))].groupby(['stationcode']).mean()
    delta_hi_australia = ds[(ds['year']>year_Q2) & (ds['year']<=year_hi) & ((ds['stationcountry']=='AUSTRALIA') | (ds['stationcountry']=='AUSTRALIA---'))].groupby(['stationcode']).mean()
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

    delta_lo_00S60S_noaus = ds[(ds['year']>=year_lo) & (ds['year']<=year_Q2) & (ds['stationlat']<=0) & (ds['stationlat']>=-60) & (ds['stationcountry']!='AUSTRALIA') & (ds['stationcountry']!='AUSTRALIA---')].groupby(['stationcode']).mean()
    delta_hi_00S60S_noaus = ds[(ds['year']>year_Q2) & (ds['year']<=year_hi) & (ds['stationlat']<=0) & (ds['stationlat']>=-60) & (ds['stationcountry']!='AUSTRALIA') & (ds['stationcountry']!='AUSTRALIA---')].groupby(['stationcode']).mean()
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
        
    delta_lo_eu = ds[(ds['year']>=year_lo) & (ds['year']<=year_Q2) & (ds['stationlat']>30) & (ds['stationlat']<=70) & (ds['stationlon']>-10) & (ds['stationlon']<=50) ].groupby(['stationcode']).mean()
    delta_hi_eu = ds[(ds['year']>year_Q2) & (ds['year']<=year_hi) & (ds['stationlat']>30) & (ds['stationlat']<=70) & (ds['stationlon']>-10) & (ds['stationlon']<=50) ].groupby(['stationcode']).mean()
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

    delta_lo_eu2 = ds[(ds['year']>=year_lo) & (ds['year']<=year_Q2) & (ds['stationlat']>30) & (ds['stationlat']<=70) & (ds['stationlon']>-10) & (ds['stationlon']<=30) ].groupby(['stationcode']).mean()
    delta_hi_eu2 = ds[(ds['year']>year_Q2) & (ds['year']<=year_hi) & (ds['stationlat']>30) & (ds['stationlat']<=70) & (ds['stationlon']>-10) & (ds['stationlon']<=30) ].groupby(['stationcode']).mean()
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

#------------------------------------------------------------------------------
# PLOTS
#------------------------------------------------------------------------------
    
if plot_temporal_coverage == True:
    
    print('plot_temporal_coverage...')

    ds = df_temp.copy()    

#   coverage per month:    
#   plt.plot(df.groupby('year').sum().iloc[:,0:12])
    
    #------------------------------------------------------------------------------
    # PLOT: temporal coverage: to 1900
    #------------------------------------------------------------------------------
             
    nbins = 1900 - ds['year'].min() + 1
    bins = np.linspace(ds['year'].min(), 1900, nbins) 
#   counts, edges = np.histogram(ds[ds['year']<=1900]['year'], nbins, range=[ds['year'].min(),1900+1], density=False)    
    counts, edges = np.histogram(ds['year'], nbins, range=[ds['year'].min(),1900], density=False)        
    Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
    Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]
             
    figstr = 'temporal-coverage-1900.png'
    titlestr = 'Yearly coverage (' + str(ds['year'].min()) + '-1900): N=' + "{0:.0f}".format(np.sum(counts))

    fig, ax = plt.subplots(figsize=(15,10))          
#   plt.hist(ds[ds['year']<=1900]['year'], bins=nbins, density=False, facecolor='grey', alpha=0.5, label='KDE')
    plt.fill_between(bins, counts, step="pre", facecolor='lightgrey', alpha=1.0)    
#   plt.plot(bins, counts, drawstyle='steps', linewidth=2, color='black')    
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
    # PLOT: temporal coverage: full record
    #------------------------------------------------------------------------------

    nbins = ds['year'].max() - ds['year'].min() + 1
    bins = np.linspace(ds['year'].min(), ds['year'].max(), nbins) 
    counts, edges = np.histogram(ds['year'], nbins, range=[ds['year'].min(),ds['year'].max()], density=False)    
    Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
    Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]
             
    figstr = 'temporal-coverage-full.png'
    titlestr = 'Yearly coverage (' + str(ds['year'].min()) + '-2019): N=' + "{0:.0f}".format(np.sum(counts)) 

    fig, ax = plt.subplots(figsize=(15,10))          
    plt.fill_between(bins, counts, step="pre", facecolor='lightgrey', alpha=1.0)
#   plt.plot(bins, counts, drawstyle='steps', linewidth=2, color='black')    
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
    # PLOT: latitudinal coverage
    #------------------------------------------------------------------------------

    print('plot_spatial_coverage: latitudinal ...')

    ds = df_temp.copy()

    nbins = lat_end - lat_start + 1
    bins = np.linspace(lat_start, lat_end, nbins) 
    counts, edges = np.histogram(ds['stationlat'], nbins, range=[lat_start,lat_end], density=False)       
    Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
    Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]

    figstr = 'latitudinal-coverage-full.png'
    titlestr = 'Latitudinal coverage (' + str(ds['year'].min()) +'-2019)'

    fig, ax = plt.subplots(figsize=(15,10))          
    plt.fill_between(bins, counts, step="pre", facecolor='lightgrey', alpha=1.0)
#   plt.plot(bins, counts, drawstyle='steps', linewidth=2, color='black')    
    plt.axvline(x=Q1, color='blue', label='Q1: ' + "{0:.0f}".format(Q1) + "$\mathrm{\degree}N$")    
    plt.axvline(x=Q2, color='red', label='Q2: ' + "{0:.0f}".format(Q2) + "$\mathrm{\degree}N$")    
    plt.axvline(x=Q3, color='blue', label='Q3: ' + "{0:.0f}".format(Q3) + "$\mathrm{\degree}N$")    
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='best', fontsize=fontsize)
    plt.xlabel("Latitude, $\mathrm{\degree}N$", fontsize=fontsize)
    plt.ylabel("Monthly values per degree", fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close(fig)

    #------------------------------------------------------------------------------
    # PLOT: longitudinal coverage
    #------------------------------------------------------------------------------

    print('plot_spatial_coverage: longitudinal ...')

    nbins = lon_end - lon_start + 1
    bins = np.linspace(lon_start, lon_end, nbins) 
    counts, edges = np.histogram(ds['stationlon'], nbins, range=[lon_start,lon_end], density=False)       
    Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
    Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]

    figstr = 'longitudinal-coverage-full.png'
    titlestr = 'Longitudinal coverage (' + str(ds['year'].min()) + '-2019)'

    fig, ax = plt.subplots(figsize=(15,10))          
    plt.fill_between(bins, counts, step="pre", facecolor='lightgrey', alpha=1.0)
#   plt.plot(bins, counts, drawstyle='steps', linewidth=2, color='black')    
    plt.axvline(x=Q1, color='blue', label='Q1: ' + "{0:.0f}".format(Q1) + "$\mathrm{\degree}E$")    
    plt.axvline(x=Q2, color='red', label='Q2: ' + "{0:.0f}".format(Q2) + "$\mathrm{\degree}E$")    
    plt.axvline(x=Q3, color='blue', label='Q3: ' + "{0:.0f}".format(Q3) + "$\mathrm{\degree}E$")    
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='best', fontsize=fontsize)
    plt.xlabel("Longitude, $\mathrm{\degree}E$", fontsize=fontsize)
    plt.ylabel("Monthly values per degree", fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close(fig)
          
if plot_station_timeseres == True:
    
    #------------------------------------------------------------------------------
    # PLOT: timeseries per station
    #------------------------------------------------------------------------------

    print('plot_station_timeseries ...')
            
    ds = df_anom.copy()
    
    #for j in range(len(np.unique(stationcode))):
    for j in range(station_start,station_end):
    
        # form station anomaly timeseries

        da = ds[ds['stationcode']==ds['stationcode'].unique()[j]].iloc[:,range(0,13)]
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
    
        figstr = 'timeseries_' + str(int(ds['stationcode'].unique()[j])) + '.png'
        titlestr = 'Annual mean temperature anomaly timeseries: stationcode=' + str(int(ds['stationcode'].unique()[j]))
              
        fig, ax = plt.subplots(figsize=(15,10))      
        plt.errorbar(t_yearly, ts_yearly, yerr=ts_yearly_sd, xerr=None, fmt='None', ecolor=hexcolors, label='± 1 s.d.')                                   
        for k in range(n):     
            if k==n-1:
                plt.scatter(t_yearly[k],ts_yearly[k], color=hexcolors[k], label='Annual mean')
            else:
                plt.scatter(t_yearly[k],ts_yearly[k], color=hexcolors[k], label=None)
        plt.clim([min(ts_yearly),max(ts_yearly)])
        plt.tick_params(labelsize=fontsize)
        ax.yaxis.grid(True, which='major')        
        plt.legend(loc=2, ncol=1, fontsize=fontsize)
        plt.xlabel("Year", fontsize=fontsize)
        plt.ylabel("Temperature anomaly, $\mathrm{\degree}C$", fontsize=fontsize)
        plt.title(titlestr, fontsize=fontsize)
        plt.savefig(figstr)
        plt.close(fig)

if plot_station_climatology == True:
    
    #------------------------------------------------------------------------------
    # PLOT: seasonal cycle per station
    #------------------------------------------------------------------------------

    print('plot_station_climatology ...')
          
    ds = df_temp.copy()
    
    #for j in range(len(np.unique(stationcode))):        
    for j in range(station_start,station_end):
    
        X = ds[ds['stationcode']==ds['stationcode'].unique()[j]].iloc[:,0]    
        Y = ds[ds['stationcode']==ds['stationcode'].unique()[j]].iloc[:,range(1,13)].T   
        n = len(Y.T)            
        colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
        hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]
        
        figstr = 'seasonal-cycle_' + str(int(ds['stationcode'].unique()[j])) + '.png'
        titlestr = 'Monthly temperature seasonal cycle: stationcode=' + str(int(ds['stationcode'].unique()[j]))

        fig, ax = plt.subplots(figsize=(15,10))      
        for k in range(n):     
            if k==0:
                plt.plot(np.arange(1,13),Y.iloc[:,k], linewidth=3, color=hexcolors[k], label=str(X.iloc[k]))
            elif k==n-1:
                plt.plot(np.arange(1,13),Y.iloc[:,k], linewidth=3, color=hexcolors[k], label=str(X.iloc[k]))
            else:
                plt.plot(np.arange(1,13),Y.iloc[:,k], linewidth=0.5, color=hexcolors[k], label=None)                
        ax.xaxis.grid(True, which='major')        
        ax.yaxis.grid(True, which='major')        
        plt.legend(loc=2, ncol=1, fontsize=fontsize)
        plt.tick_params(labelsize=fontsize)   
        plt.xlabel("Month", fontsize=fontsize)
        plt.ylabel("Monthly temperature, $\mathrm{\degree}C$", fontsize=fontsize)        
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

if plot_station_locations == True:
    
    #------------------------------------------------------------------------------
    # PLOT: stations on world map
    #------------------------------------------------------------------------------

    print('plot_station_locations ...')

    ds = df_temp.copy()

    lon = ds['stationlon']
    lat = ds['stationlat']
    
    figstr = 'location_map.png'
    titlestr = 'GloSATp02: monthly obs=' + str(len(ds)) + ', stations=' + str(len(ds['stationcode'].unique()))
     
    fig  = plt.figure(figsize=(15,10))
    p = ccrs.PlateCarree(central_longitude=0); threshold = 0
    ax = plt.axes(projection=p)
    ax.set_global()
#   ax.stock_img()
#   ax.add_feature(cf.COASTLINE, edgecolor="lightblue")
#   ax.add_feature(cf.BORDERS, edgecolor="lightblue")
    ax.coastlines(color='lightblue')
#   ax.gridlines()    
        
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

