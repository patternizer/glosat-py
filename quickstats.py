#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: quickstats.py
#------------------------------------------------------------------------------
# Version 0.2
# 2 June, 2022
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------

fontsize = 16

fix_known_cases = False    # ( default = False ) NB: True --> updates df_temp_qc dataframe
percentile_filter = False  # ( default = False ) NB: True --> updates df_temp_qc dataframe
save_pkl = False           # ( default = False ) NB: True --> overwrites df_temp_qc dataframe
test_station = True        # ( default = False )

base_start, base_end = 1961, 1990
stationname, stationcode = 'CET', '037401'

filename_pkl = 'df_temp_qc.pkl'

#------------------------------------------------------------------------------
# LOAD: dataframe
#------------------------------------------------------------------------------

df_temp = pd.read_pickle( filename_pkl, compression='bz2' )

# STATS: count, mean, std, [5-point stats] 

stats1 = df_temp.describe()
stats2 = df_temp.groupby('stationcode').count().describe() # number of stations 

#------------------------------------------------------------------------------
# ARCHIVE: integrity
#------------------------------------------------------------------------------

stationcodes = df_temp['stationcode'].unique()
stationnames = df_temp['stationname'].unique()
stationlats = df_temp.groupby('stationcode').mean().stationlat.values
stationlons = df_temp.groupby('stationcode').mean().stationlon.values
stationelevations = df_temp.groupby('stationcode').mean().stationelevation.values
stationyears = df_temp.groupby('stationcode').count().year.values
nstations_counts = df_temp.groupby('stationcode').count()
nstations = len(stationcodes)
    
# STATS: missing metadata counts

stationcodes_nodata = stationcodes[ stationyears == 0 ]
stationcodes_nolats = stationcodes[ ~np.isfinite( stationlats ) ]
stationcodes_nolons = stationcodes[ ~np.isfinite( stationlons ) ]
stationcodes_noelevations = stationcodes[ ~np.isfinite( stationelevations ) ]

print('N(stations) =', nstations)
print('stationcodes: missing =', len(stationcodes_nodata), ' :', stationcodes_nodata)
print('stationlats: missing =', len(stationcodes_nolats), ' :',  stationcodes_nolats)
print('stationlons: missing =', len(stationcodes_nolons), ' :',  stationcodes_nolons)
print('stationelevations: missing =', len(stationcodes_noelevations), ' :',  stationcodes_noelevations)

# STATS: year minmax

print('year min = ', df_temp.year.min())
print('year max = ', df_temp.year.max())

# STATS: temp minmax

tmax = []
tmin = []
for j in range(1,13):

    tmax_j = list( np.array( df_temp.copy().dropna(subset=[str(j)]).nlargest(n=1, columns=[str(j)]) ).ravel()[0:19] )
    tmin_j = list( np.array( df_temp.copy().dropna(subset=[str(j)]).nsmallest(n=1, columns=[str(j)]) ).ravel()[0:19] )
    tmax.append( tmax_j )
    tmin.append( tmin_j )

    print('Tmax (month=' + str(j) + '): ', tmax_j )
    print('Tmin (month=' + str(j) + '): ', tmin_j )

# STATS: global tmax and tmin

global_tmax = np.nanmax(np.array(tmax)[:,1:13].astype(float).ravel())
global_tmin = np.nanmin(np.array(tmin)[:,1:13].astype(float).ravel())

print('Global Tmax = ', global_tmax )
print('Global Tmin = ', global_tmin )

#==============================================================================
# OUTLIER: checks
#==============================================================================

# FILTER: outliers

if fix_known_cases == True:
            
    print('fixing known cases ...')
                
    # TMAX - exceptions

    outlier_stationcode = np.array(['897340','897340','986440'])
    outlier_year = np.array([2018,2016,2012])
    outlier_month = np.array(['2','3','12'])

    df_outliers = pd.DataFrame({'stationcode':outlier_stationcode, 'year':outlier_year, 'month':outlier_month})

    for i in range(len(df_outliers)):
                
        row = df_temp[ (df_temp['stationcode'] == df_outliers.stationcode[i]) & (df_temp['year'] == df_outliers.year[i]) ].index[0]
        df_temp.loc[ row, df_outliers.month[i]] = np.nan
    
if percentile_filter == True: 
    
    print('fixing outliers ( <> percentile 5-95% range ) ...')

    # FIND: rows where any months >= 95th percentile or <= 5th percentile
    
    da_lower = np.percentile( df_temp.iloc[:,1:13], axis=1, q=1 )
    da_upper = np.percentile( df_temp.iloc[:,1:13], axis=1, q=99 )
    db = pd.DataFrame(df_temp, columns=['1','2','3','4','5','6','7','8','9','10','11','12'])
    idx = []
    for i in range(len(df_temp)):
        if (db.iloc[i,:].any() >= da_upper[i]) | (db.iloc[i,:].any() <= da_lower[i]):
            idx.append( i )    
    
    # REPLACE: outlier months with NaN
    
    for i in range(len(idx)):
        mask = ( df_temp.iloc[idx[i],1:13] >= da_upper[idx[i]] ) | ( df_temp.iloc[idx[i],1:13] <= da_lower[idx[i]] )
        col = np.arange(1,13)[mask]
        for j in range(len(col)):
            df_temp.loc[idx[i],[str(col[j])]] = np.nan

#==============================================================================
# SAVE: pkl
#==============================================================================

print('saving pkl ...')

if save_pkl == True:
    
    df_temp.to_pickle( filename_pkl, compression='bz2')
                                    
#==============================================================================
# SPOT CHECK: selected station
#==============================================================================

if test_station == True:
    
    print('plotting test station anomaly series ...')
    
    #------------------------------------------------------------------------------
    # PRINT: station metadata
    #------------------------------------------------------------------------------
    
    if len(stationname) == 0:	
    	stationname = df_temp[ df_temp['stationcode'] == stationcode ].stationname.unique()[0]
    else:
    	stationcode = df_temp[ df_temp['stationname'].str.contains( stationname, case=False) ].stationcode.unique()[0]
    stationlat = df_temp[ df_temp['stationcode'] == stationcode ].stationlat.unique()[0]
    stationlon = df_temp[ df_temp['stationcode'] == stationcode ].stationlon.unique()[0]
    print('test station=', stationname)
    print('test stationcode=', stationcode)
    print('test stationlat=', stationlat)
    print('test stationlon=', stationlon)
    print( df_temp[ df_temp['stationname'].str.contains( stationname, case=False) ].describe().year )
    
    #------------------------------------------------------------------------------
    # COMPUTE: station anomaly series
    #------------------------------------------------------------------------------
    
    df_base = df_temp[ (df_temp.year>=base_start) & (df_temp.year<=base_end) ]
    normals = df_base.groupby('stationcode').mean().iloc[:,1:13]
    counts = df_base.groupby('stationcode').count().iloc[:,1:13]
    normals[ counts <= 15 ] = np.nan
    
    df_temp_station = df_temp[ df_temp.stationcode==stationcode ].reset_index(drop=True)
    df_anom_station = df_temp_station.copy()
    
    # TRIM: to 1678 to edge of Pandas
    
    df_anom_station = df_anom_station[ df_anom_station.year >= 1678 ]
    
    # COMPUTE: anomaly timeseries

    normals_station = normals[ normals.index==stationcode].reset_index(drop=True)
    if np.isfinite(normals_station).values.sum() == 0:
        print('Station has no normals')
    else:
        for i in range(1,13): 
            df_anom_station[str(i)] = df_temp_station[str(i)] - normals_station[str(i)][0]
    
        t_station = pd.date_range( start=str(df_anom_station.year.iloc[0]), end=str(df_anom_station.year.iloc[-1]+1), freq='MS')[0:-1]                                                                                                                                          
        ts_station = []    
        for i in range(len(df_anom_station)):            
            monthly = df_anom_station.iloc[i,1:13]
            ts_station = ts_station + monthly.to_list()    
        ts_station = np.array(ts_station)   
    
        # PLOT: anomaly series
    
        fig, ax = plt.subplots()
        plt.plot(t_station, pd.Series(ts_station).rolling(24).mean())  
        plt.ylabel('2m Temperature Anomaly from ' + str(base_start) + '-' + str(base_end) )  
        plt.title(stationcode + ':' + stationname + ' (24m MA)')
        fig.savefig(stationname + '_quickplot.png', dpi=300)
    
#------------------------------------------------------------------------------
print('** END')
