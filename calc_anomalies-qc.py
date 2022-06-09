#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: calc-anomalies-qc.py
#------------------------------------------------------------------------------
# Version 0.1
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
import re
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

save_pkl = True # ( default = True ) 
float64 = True  # ( detault = True ) # False --> float32

if float64 == True: 
    precision = 'float64'
else: 
    precision = 'float32'
    
filename_temp = 'df_temp_qc.pkl'
filename_normals = 'df_normals_qc.pkl'
filename_anomalies = 'df_anom_qc.pkl'

#------------------------------------------------------------------------------
# LOAD: absolute temperatures and normals
#------------------------------------------------------------------------------

print('loading temperatures and normals ...')

df_temp = pd.read_pickle( filename_temp, compression='bz2' )          # fill value = np.nan
df_normals = pd.read_pickle( filename_normals, compression='bz2' )    # fill value = np.nan

#    1 ID
#    2-3 First year, last year
#    4-5 First year for normal, last year for normal
#    6-17 Twelve normals for Jan to Dec, degC (not degC*10), missing normals are np.nan
#    18 Source code (1=missing, 2=Phil's estimate, 3=WMO, 4=calculated here, 5=calculated previously)
#    19-30 Twelve values giving the % data available for computing each normal from the 1961-1990 period
           
#------------------------------------------------------------------------------
# COMPUTE: anomalies
#------------------------------------------------------------------------------

print('calculating anomalies ...')

# COMPUTE: number of years per station

counts = df_temp.groupby('stationcode').count().year

# EXTRACT: normals monthly values per station

normals = np.array(df_normals)[:,6:18]

# CONSTRUCT: normals monthly values array of the same size as df_temp

for i in range(len(counts)):

    if i % 100 == 0: print( i )

    b = np.tile( normals[i,:], [ counts[i], 1 ] )            
    if i == 0:
        normals_repeated = b        
    else:
        a = normals_repeated
        normals_repeated = np.concatenate((normals_repeated, b), axis=0)    

# COMPUTE: anomalies

df_anom = df_temp.copy()
A = np.array( df_anom.iloc[:,1:13] )
B = np.array( normals_repeated ).astype( dtype = precision)
C = np.array( A - B ).astype( dtype = precision)

# STORE: anomalies

df_anom.iloc[:,1:13] = C
for i in range(1,13): df_anom[str(i)] = df_anom[str(i)].astype( dtype = precision)
                     
#------------------------------------------------------------------------------
# SAVE: anomalies
#------------------------------------------------------------------------------

if save_pkl == True:
			
    print('saving anomalies ...')

    df_anom.to_pickle( filename_anomalies, compression='bz2')
    
#------------------------------------------------------------------------------
print('** END')

