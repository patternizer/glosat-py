#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: plot-prelim-stations.py
#------------------------------------------------------------------------------
# Version 0.6
# 27 July, 2020
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

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
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
plot_temporal_coverage = True
plot_spatial_coverage = True
plot_station_timeseres = True
plot_monthly_climatology = True
plot_locations = True
plot_delta_cc = True

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

#------------------------------------------------------------------------------
# LOAD DATAFRAME
#------------------------------------------------------------------------------

if load_df == True:

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
    for j in range(1,12+1):
        df[df.columns[j]] = df[df.columns[j]]/100.0
    # Convert +W to +E
    df['stationlon'] = -df['stationlon']        
    df.to_csv('df.csv')

#------------------------------------------------------------------------------
# BAD DATA ANALYSIS
#------------------------------------------------------------------------------

# df['stationlat'].isnull().sum()
# dlat = df[df['stationlat'].isnull()]
# dlon = df[df['stationlon'].isnull()]
# dcode = df[df['stationcode'].isnull()]
    
# for i in range(len(dlat['stationcode'].unique())):
#    fail_lat = list(dlat[dlat['stationcode']==dlat['stationcode'].unique()[i]]['stationinfo'])[0]
#    fail_lon = list(dlon[dlon['stationcode']==dlon['stationcode'].unique()[i]]['stationinfo'])[0]
#    print(fail_lat)
#    print(fail_lon)
        
# dlat and dlon fails: 
# array(['085997', '685807', '688607', '967811', '999099', '999216'], dtype=object)

# ['085997-999-9999', '100', 'SERRO', 'DO', 'PILAR', 'PORTUGAL', '19011930', '101901', '0']
# ['685807-999-9999-9999', 'PIETERMARITZBERG', 'SOUTH', 'AFRICA', '18701886', '101870', '0']
# ['688607-999-9999-9999', 'GRAHAMSTOWN', 'SOUTH', 'AFRICA', '18551870', '101855', '0']
# ['967811-999-9999', '1400', 'TJIBODAS', 'INDONESIA', '19051948', '101905', '0']
# ['999096-341-9999-9999', 'BREAKSEA', 'AUSTRALIA', '18971899', '101897', '0']
# ['999099-999-9999-9999', 'LONDON', 'AUSTRALIA', '18971899', '101897', '0']
# ['999216-999-9999-9999', 'LAS', 'DELICIAS', 'ARGENTINA', '19251964', '101925', '0']

lat_min = df['stationlat'].min()
lat_max = df['stationlat'].max()
lon_min = df['stationlon'].min()
lon_max = df['stationlon'].max()
year_min = df['year'].min()
year_max = df['year'].max()

if plot_delta_cc == True:

    # Delta CC +/- 30 years around median value (data to 1900) = 1885
    
    nbins = 1900 - year_min + 1
    bins = np.linspace(year_min, 1900, nbins) 
    counts, edges = np.histogram(df['year'], nbins, range=[year_min,1900+1], density=False)        
    year_Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    year_lo = year_Q2-30
    year_hi = year_Q2+30
    
    def plot_hist_array(diff, figstr, titlestr):
        
        def plot_hist(diff,i,row,col):
            
            nbins = int((0.5-(-0.5))/0.02) + 1
            bins = np.linspace(-0.5, 0.5, nbins) 
            counts, edges = np.histogram(diff[str(i)], nbins, range=[-0.5,0.5+0.02], density=False)    
            Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
            Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
            Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]
            axs[row,col].fill_between(bins, counts, step="post", facecolor='lightgrey', alpha=0.5)
            axs[row,col].plot(bins, counts, drawstyle='steps-post', linewidth=1, color='black')    
            axs[row,col].axvline(x=Q1, color='blue', label='Q1: ' + "{0:.2f}".format(Q1))   
            axs[row,col].axvline(x=Q2, color='red', label='Q2: ' + "{0:.2f}".format(Q2))    
            axs[row,col].axvline(x=Q3, color='blue', label='Q3: ' + "{0:.2f}".format(Q3))    
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
            ax.set_xlabel("$\mathrm{\Delta}$(anomaly), $\mathrm{\degree}$C", fontsize=15)
            ax.set_ylabel("Counts / 0.02$\mathrm{\degree}$C", fontsize=15)
        for ax in axs.flat:
            ax.label_outer()
            ax.legend(loc='best', fontsize=8)   
        
        fig.suptitle(titlestr, fontsize=fontsize)    
        plt.savefig(figstr)
        plt.close(fig)

    delta_lo_60N90N = df[(df['year']>=year_lo) & (df['year']<year_Q2) & (df['stationlat']>=60) & (df['stationlat']<90)].groupby(['stationcode']).mean()
    delta_hi_60N90N = df[(df['year']>=year_Q2) & (df['year']<year_hi) & (df['stationlat']>=60) & (df['stationlat']<90)].groupby(['stationcode']).mean()
    diff = delta_hi_60N90N - delta_lo_60N90N    
    figstr = 'delta_cc_60N90N.png'
    titlestr = 'Histograms of mean change in anomaly $\mathrm{\pm}$ 30 years of 1885: 60-90°N'
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_30N60N = df[(df['year']>=year_lo) & (df['year']<year_Q2) & (df['stationlat']>=30) & (df['stationlat']<60)].groupby(['stationcode']).mean()
    delta_hi_30N60N = df[(df['year']>=year_Q2) & (df['year']<year_hi) & (df['stationlat']>=30) & (df['stationlat']<60)].groupby(['stationcode']).mean()
    diff = delta_hi_30N60N - delta_lo_30N60N    
    figstr = 'delta_cc_30N60N.png'
    titlestr = 'Histograms of mean change in anomaly $\mathrm{\pm}$ 30 years of 1885: 30-60°N'
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_00N30N = df[(df['year']>=year_lo) & (df['year']<year_Q2) & (df['stationlat']>=0) & (df['stationlat']<30)].groupby(['stationcode']).mean()
    delta_hi_00N30N = df[(df['year']>=year_Q2) & (df['year']<year_hi) & (df['stationlat']>=0) & (df['stationlat']<30)].groupby(['stationcode']).mean()
    diff = delta_hi_00N30N - delta_lo_00N30N    
    figstr = 'delta_cc_00N30N.png'
    titlestr = 'Histograms of mean change in anomaly $\mathrm{\pm}$ 30 years of 1885: 0-30°N'
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_00S30S = df[(df['year']>=year_lo) & (df['year']<year_Q2) & (df['stationlat']<=0) & (df['stationlat']>-30)].groupby(['stationcode']).mean()
    delta_hi_00S30S = df[(df['year']>=year_Q2) & (df['year']<year_hi) & (df['stationlat']<=0) & (df['stationlat']>-30)].groupby(['stationcode']).mean()
    diff = delta_hi_00S30S - delta_lo_00S30S    
    figstr = 'delta_cc_00S30S.png'
    titlestr = 'Histograms of mean change in anomaly $\mathrm{\pm}$ 30 years of 1885: 0-30°S'
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_30S60S = df[(df['year']>=year_lo) & (df['year']<year_Q2) & (df['stationlat']<=-30) & (df['stationlat']>-60)].groupby(['stationcode']).mean()
    delta_hi_30S60S = df[(df['year']>=year_Q2) & (df['year']<year_hi) & (df['stationlat']<=-30) & (df['stationlat']>-60)].groupby(['stationcode']).mean()
    diff = delta_hi_30S60S - delta_lo_30S60S    
    figstr = 'delta_cc_30S60S.png'
    titlestr = 'Histograms of mean change in anomaly $\mathrm{\pm}$ 30 years of 1885: 30-60°S'
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_60S90S = df[(df['year']>=year_lo) & (df['year']<year_Q2) & (df['stationlat']<=-60) & (df['stationlat']>-90)].groupby(['stationcode']).mean()
    delta_hi_60S90S = df[(df['year']>=year_Q2) & (df['year']<year_hi) & (df['stationlat']<=-60) & (df['stationlat']>-90)].groupby(['stationcode']).mean()
    diff = delta_hi_60S90S - delta_lo_60S90S    
    figstr = 'delta_cc_60S90S.png'
    titlestr = 'Histograms of mean change in anomaly $\mathrm{\pm}$ 30 years of 1885: 60-90°S'
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_global = df[(df['year']>=year_lo) & (df['year']<year_Q2)].groupby(['stationcode']).mean()
    delta_hi_global = df[(df['year']>=year_Q2) & (df['year']<year_hi)].groupby(['stationcode']).mean()
    diff = delta_hi_global - delta_lo_global    
    figstr = 'delta_cc_global.png'
    titlestr = 'Histograms of mean change in anomaly $\mathrm{\pm}$ 30 years of 1885: Global'
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_us = df[(df['year']>=year_lo) & (df['year']<year_Q2) & (df['stationcountry']=='USA')].groupby(['stationcode']).mean()
    delta_hi_us = df[(df['year']>=year_Q2) & (df['year']<year_hi) & (df['stationcountry']=='USA')].groupby(['stationcode']).mean()
    diff = delta_hi_us - delta_lo_us    
    figstr = 'delta_cc_us.png'
    titlestr = 'Histograms of mean change in anomaly $\mathrm{\pm}$ 30 years of 1885: USA'
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_australia = df[(df['year']>=year_lo) & (df['year']<year_Q2) & (df['stationcountry']=='AUSTRALIA')].groupby(['stationcode']).mean()
    delta_hi_australia = df[(df['year']>=year_Q2) & (df['year']<year_hi) & (df['stationcountry']=='AUSTRALIA')].groupby(['stationcode']).mean()
    diff = delta_hi_australia - delta_lo_australia   
    figstr = 'delta_cc_australia.png'
    titlestr = 'Histograms of mean change in anomaly $\mathrm{\pm}$ 30 years of 1885: Australia'
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_00S60S_noaus = df[(df['year']>=year_lo) & (df['year']<year_Q2) & (df['stationlat']<=0) & (df['stationlat']>-60) & (df['stationcountry']!='AUSTRALIA')].groupby(['stationcode']).mean()
    delta_hi_00S60S_noaus = df[(df['year']>=year_Q2) & (df['year']<year_hi) & (df['stationlat']<=0) & (df['stationlat']>-60) & (df['stationcountry']!='AUSTRALIA')].groupby(['stationcode']).mean()
    diff = delta_hi_00S60S_noaus - delta_lo_00S60S_noaus    
    figstr = 'delta_cc_00S60S_noaus.png'
    titlestr = 'Histograms of mean change in anomaly $\mathrm{\pm}$ 30 years of 1885: 0-60°S (-Australia)'
    plot_hist_array(diff, figstr, titlestr)
        
    delta_lo_eu = df[(df['year']>=year_lo) & (df['year']<year_Q2) & (df['stationlat']>=30) & (df['stationlat']<70) & (df['stationlon']>=-10) & (df['stationlon']<50) ].groupby(['stationcode']).mean()
    delta_hi_eu = df[(df['year']>=year_Q2) & (df['year']<year_hi) & (df['stationlat']>=30) & (df['stationlat']<70) & (df['stationlon']>=-10) & (df['stationlon']<50) ].groupby(['stationcode']).mean()
    diff = delta_hi_eu - delta_lo_eu    
    figstr = 'delta_cc_eu.png'
    titlestr = 'Histograms of mean change in anomaly $\mathrm{\pm}$ 30 years of 1885: Europe + ME'
    plot_hist_array(diff, figstr, titlestr)

    delta_lo_eu2 = df[(df['year']>=year_lo) & (df['year']<year_Q2) & (df['stationlat']>=30) & (df['stationlat']<70) & (df['stationlon']>=-10) & (df['stationlon']<30) ].groupby(['stationcode']).mean()
    delta_hi_eu2 = df[(df['year']>=year_Q2) & (df['year']<year_hi) & (df['stationlat']>=30) & (df['stationlat']<70) & (df['stationlon']>=-10) & (df['stationlon']<30) ].groupby(['stationcode']).mean()
    diff = delta_hi_eu2 - delta_lo_eu2    
    figstr = 'delta_cc_eu2.png'
    titlestr = 'Histograms of mean change in anomaly $\mathrm{\pm}$ 30 years of 1885: Europe (-ME)'
    plot_hist_array(diff, figstr, titlestr)

    #for i in range(1,13):
    
    #    counts, edges = np.histogram(diff[str(i)], nbins, range=[-1.0,1.0+0.01], density=False)    
    #    Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
    #    Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    #    Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]
                 
    #    figstr = 'delta_cc_' + "{0:.0f}".format(i) + '.png'
    #    titlestr = 'Histogram of mean change in anomaly $\mathrm{\pm}$ 30 years of 1885: month=' + "{0:.0f}".format(i)
    
    #    fig, ax = plt.subplots(figsize=(15,10))          
    #    plt.fill_between(bins, counts, step="post", facecolor='lightgrey', alpha=0.5)
    #    plt.plot(bins, counts, drawstyle='steps-post', linewidth=2, color='black')    
    #    plt.axvline(x=Q1, color='blue', label='Q1: ' + "{0:.2f}".format(Q1))   
    #    plt.axvline(x=Q2, color='red', label='Q2: ' + "{0:.2f}".format(Q2))    
    #    plt.axvline(x=Q3, color='blue', label='Q3: ' + "{0:.2f}".format(Q3))    
    #    plt.tick_params(labelsize=fontsize)    
    #    plt.legend(loc='best', fontsize=fontsize)
    #    plt.xlabel("$\mathrm{\Delta}$(anomaly), $\mathrm{\degree}$C", fontsize=fontsize)
    #    plt.ylabel("Stations per 0.01$\mathrm{\degree}$C bin", fontsize=fontsize)
    #    plt.title(titlestr, fontsize=fontsize)
    #    plt.savefig(figstr)
    #    plt.close(fig)

#------------------------------------------------------------------------------
# PLOTS
#------------------------------------------------------------------------------

if plot_temporal_coverage == True:
    
    # PLOT: histogram of temporal coverage: to 1900
          
    nbins = 1900 - year_min + 1
    bins = np.linspace(year_min, 1900, nbins) 
    counts, edges = np.histogram(df[df['year']<1901]['year'], nbins, range=[year_min,1900+1], density=False)    
    Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
    Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]
             
    figstr = 'crutem5-histogram-1900.png'
    titlestr = 'Histogram of yearly coverage for CRUTEM5 (to 1900): N=' + "{0:.0f}".format(np.sum(counts))

    fig, ax = plt.subplots(figsize=(15,10))          
#    plt.hist(df[df['year']<1901]['year'], bins=nbins, density=True, facecolor='grey', alpha=0.5, label='KDE')
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

    # PLOT: histogram of temporal coverage: full record

    nbins = year_max - year_min + 1
    bins = np.linspace(year_min, year_max, nbins) 
    counts, edges = np.histogram(df['year'], nbins, range=[year_min,year_max+1], density=False)    
    Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
    Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]
             
    figstr = 'crutem5-histogram-full.png'
    titlestr = 'Histogram of yearly coverage for CRUTEM5 (full): N=' + "{0:.0f}".format(np.sum(counts)) 

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
    
    # PLOT: histogram of latitudinal coverage

    nbins = lat_end - lat_start + 1
    bins = np.linspace(lat_start, lat_end, nbins) 
    counts, edges = np.histogram(df[(df['stationlat']>lat_start) & (df['stationlat']<lat_end)]['stationlat'], nbins, range=[lat_start,lat_end+1], density=False)    
    Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
    Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]

    figstr = 'crutem5-histogram-lat-full.png'
    titlestr = 'Histogram of latitudinal coverage: CRUTEM5 (full)'

    fig, ax = plt.subplots(figsize=(15,10))          
    plt.fill_between(bins, counts, step="post", facecolor='lightgrey', alpha=0.5)
    plt.plot(bins, counts, drawstyle='steps-post', linewidth=2, color='black')    
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

    # PLOT: histogram of longitudinal coverage

    nbins = lon_end - lon_start + 1
    bins = np.linspace(lon_start, lon_end, nbins) 
    counts, edges = np.histogram(df[(df['stationlon']>lon_start) & (df['stationlon']<lon_end)]['stationlon'], nbins, range=[lon_start,lon_end+1], density=False)    
    Q1 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.25).abs().argsort()[:1]].values[0]
    Q2 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.50).abs().argsort()[:1]].values[0]
    Q3 = pd.Series(bins).iloc[(pd.Series(np.cumsum(counts))-np.sum(counts)*0.75).abs().argsort()[:1]].values[0]

    figstr = 'crutem5-histogram-lon-full.png'
    titlestr = 'Histogram of longitudinal coverage: CRUTEM5 (full)'

    fig, ax = plt.subplots(figsize=(15,10))          
    plt.fill_between(bins, counts, step="post", facecolor='lightgrey', alpha=0.5)
    plt.plot(bins, counts, drawstyle='steps-post', linewidth=2, color='black')    
    plt.axvline(x=Q1, color='blue', label='Q1: ' + "{0:.0f}".format(Q1))    
    plt.axvline(x=Q2, color='red', label='Q2: ' + "{0:.0f}".format(Q2))    
    plt.axvline(x=Q3, color='blue', label='Q3: ' + "{0:.0f}".format(Q3))    
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
    
    # PLOT: seasonal cycle per station
        
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
    
    # PLOT: quick plot of stations on world map

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

