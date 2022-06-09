#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: prepare_app.py
#------------------------------------------------------------------------------
# Version 0.1
# 9 June, 2022
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
from sys import getsizeof

#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------

float64 = False # ( detault = False  )
if float64 == True: 
    precision = 'float64' 
else: 
    precision = 'float32'
    
filename_temp_in = 'df_temp_qc.pkl'
filename_anom_in = 'df_anom_qc.pkl'

filename_temp_out = 'df_temp_app.pkl'
filename_anom_out = 'df_anom_app.pkl'

#------------------------------------------------------------------------------
# LOAD: dataframes
#------------------------------------------------------------------------------

df_temp = pd.read_pickle( filename_temp_in, compression='bz2' )
df_anom = pd.read_pickle( filename_anom_in, compression='bz2' )

#------------------------------------------------------------------------------
# DROP: unused variables
#------------------------------------------------------------------------------

df_temp.drop( columns = ['stationfirstyear','stationlastyear','stationsource','stationfirstreliable'], inplace=True)
df_anom.drop( columns = ['stationfirstyear','stationlastyear','stationsource','stationfirstreliable'], inplace=True )

print('IN (column memory size):')
print('df_temp:', getsizeof( df_temp ) )
print('df_anom:', getsizeof( df_anom ) )

for i in range(len(df_temp.columns)): print( getsizeof( df_temp.iloc[:,i] ) ) 
for i in range(len(df_anom.columns)): print( getsizeof( df_anom.iloc[:,i] ) ) 

#------------------------------------------------------------------------------
# CONVERT: precision
#------------------------------------------------------------------------------

for j in range(1,13): 
    
    df_temp[ str(j) ] = df_temp[ str(j) ].astype( precision )        
    df_anom[ str(j) ] = df_anom[ str(j) ].astype( precision )        

df_temp['stationlat'] = df_temp['stationlat'].astype( precision )
df_temp['stationlon'] = df_temp['stationlon'].astype( precision )
df_temp['stationelevation'] = df_temp['stationelevation'].astype( precision )

df_anom['stationlat'] = df_anom['stationlat'].astype( precision )
df_anom['stationlon'] = df_anom['stationlon'].astype( precision )
df_anom['stationelevation'] = df_anom['stationelevation'].astype( precision )

#------------------------------------------------------------------------------
# MEMORY TEST: float vs str
#------------------------------------------------------------------------------

mem_str = getsizeof(df_temp['stationelevation'].iloc[0].astype(str))    # float32=85, float64=85
mem_num = getsizeof(df_temp['stationelevation'].iloc[0])                # float32=28, float64=32
mem_nan = getsizeof( np.nan )                                           # 24

print('OUT (column memory size):')
print('df_temp:', getsizeof( df_temp ) )
print('df_anom:', getsizeof( df_anom ) )
      
for i in range(len(df_temp.columns)): print( getsizeof( df_temp.iloc[:,i] ) ) 
for i in range(len(df_anom.columns)): print( getsizeof( df_anom.iloc[:,i] ) ) 

#------------------------------------------------------------------------------
# SAVE: dataframes for app
#------------------------------------------------------------------------------

df_temp.to_pickle( filename_temp_out, compression='bz2' )
df_anom.to_pickle( filename_anom_out, compression='bz2' )

#------------------------------------------------------------------------------
print('** END')
