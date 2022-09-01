#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: load-sds-qc.py
#------------------------------------------------------------------------------
# Version 0.2
# 1 September, 2022
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
# Dataframe libraries:
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------

base_start, base_end = 1941, 1990

#sds_file = 'sd5.GloSAT.prelim04_FRYuse_ocPLAUS1_iqr3.600reg0.3_19411990_MIN15_OCany_19611990_MIN15_PERDEC00_NManySDreq.txt'
sds_file = 'sd5.GloSAT.prelim04c.1781LEKnorms_FRYuse_ocPLAUS1_iqr3.600reg0.3simpleSD4.0_19411990_MIN15_OCany_19611990_MIN15_PERDEC00_NManySDreq.txt'

load_df_sds = False # ( default = False ) True --> load pre-calculated pkl
check_sds = False   # ( default = False ) True --> compare loaded vs MIN15 data-driven sds
float64 = True      # ( detault = True  ) False --> float32

if float64 == True: 
    precision = 'float64' 
else: 
    precision = 'float32'
    
filename_sds = 'df_sds_qc.pkl'
filename_temp = 'df_temp_qc.pkl'

#------------------------------------------------------------------------------
# LOAD normals
#------------------------------------------------------------------------------

if load_df_sds == True:

    print('loading SDs ...')

    df_sds = pd.read_pickle( filename_sds, compression='bz2' )    

else:

    print('extracting SDs ...')

	#    1 ID
	#    2-3 First year, last year
	#    4-5 First year for SD, last year for SD
	#    6-17 Twelve SDs for Jan to Dec, degC (not degC*10), missing SDs are -999.000
	#    18 Source code (1=missing, 2=Phil's estimate, 3=WMO, 4=calculated here, 5=calculated previously)
	#    19-30 Twelve values giving the % data available for computing each SD from the 1941-1990 period

    f = open(sds_file)
    lines = f.readlines()

    stationcodes = []
    firstyears = []
    lastyears = []
    sdfirstyears = []
    sdlastyears = []
    sourcecodes = []
    sd12s = []
    sdpercentage12s = []
    
    for i in range(len(lines)):
        words = lines[i].split()
        if len(words) > 1:
            stationcode = words[0].zfill(6)
            firstyear = int(words[1])
            lastyear = int(words[2])
            sdfirstyear = int(words[3])
            sdlastyear = int(words[4])
            sourcecode = int(words[17])
            sd12 = words[5:17]
            sdpercentage12 = words[18:31]

            stationcodes.append(stationcode)
            firstyears.append(firstyear)
            lastyears.append(lastyear)
            sdfirstyears.append(sdfirstyear)
            sdlastyears.append(sdlastyear)
            sourcecodes.append(sourcecode)
            sd12s.append(sd12)
            sdpercentage12s.append(sdpercentage12)
    f.close()
    
    sd12s = np.array(sd12s).astype( precision )
    sdpercentage12s = np.array(sdpercentage12s).astype(int)

    df_sds = pd.DataFrame({
        'stationcode':stationcodes,
        'firstyear':firstyears, 
        'lastyear':lastyears, 
        'sdfirstyear':sdfirstyears, 
        'sdlastyear':sdlastyears, 
        'sourcecode':sourcecodes,
        '1':sd12s[:,0],                               
        '2':sd12s[:,1],                               
        '3':sd12s[:,2],                               
        '4':sd12s[:,3],                               
        '5':sd12s[:,4],                               
        '6':sd12s[:,5],                               
        '7':sd12s[:,6],                               
        '8':sd12s[:,7],                               
        '9':sd12s[:,8],                               
        '10':sd12s[:,9],                               
        '11':sd12s[:,10],                               
        '12':sd12s[:,11],                               
        '1pc':sdpercentage12s[:,0],                               
        '2pc':sdpercentage12s[:,1],                               
        '3pc':sdpercentage12s[:,2],                               
        '4pc':sdpercentage12s[:,3],                               
        '5pc':sdpercentage12s[:,4],                               
        '6pc':sdpercentage12s[:,5],                               
        '7pc':sdpercentage12s[:,6],                               
        '8pc':sdpercentage12s[:,7],                               
        '9pc':sdpercentage12s[:,8],                               
        '10pc':sdpercentage12s[:,9],                               
        '11pc':sdpercentage12s[:,10],                               
        '12pc':sdpercentage12s[:,11],                                                              
    })   

#------------------------------------------------------------------------------
# MATCH: df_temp_qc stationcodes
#------------------------------------------------------------------------------

print('matching stationcodes ... ')

df_temp = pd.read_pickle( filename_temp, compression='bz2' )    

stationcodes_df_temp = df_temp.stationcode.unique()
stationcodes_df_sds = df_sds.stationcode
idx = list( set(stationcodes_df_sds) - set(stationcodes_df_temp) )  # stationcodes in df_temp but not in df_sds

for i in range(len(idx)):    
    rows = df_sds[ df_sds.stationcode == idx[i] ].index
    df_sds =  df_sds.drop( rows )
    
#------------------------------------------------------------------------------
# REPLACE: fill value -999 with np.nan
#------------------------------------------------------------------------------

for j in range(1,13): df_sds[ str(j) ].replace(-999, np.nan, inplace=True)

#------------------------------------------------------------------------------
# CHECK: SDs
#------------------------------------------------------------------------------
            
if check_sds == True:
		
    print('checking SDs ...')
            
    # EXTRACT: CRUTEM5 station SDs

    df_sds1 = df_sds.copy()
    df_sds1.index = df_sds1['stationcode']
    df_sds1 = df_sds1[ df_sds1.columns[6:18] ]
    df_sds1 = df_sds1.dropna()
        
    # COMPUTE: SDs from quality-controlled df_temp

    df_sds2 = df_temp.copy()
    df_sds2_means = df_sds2[ (df_sds2['year']>=base_start) & (df_sds2['year']<=base_end)].groupby(['stationcode']).std().iloc[:,1:13] # climatological means
    df_sds2_mask = df_sds2[ (df_sds2['year']>=base_start) & (df_sds2['year']<=base_end)].groupby(['stationcode']).count().iloc[:,1:13] >= 15 # boolean mask for n>=15 in baseline
    df_sds2 = df_sds2_means[df_sds2_mask] # SDs
    df_sds2 = df_sds2.astype( precision )
        
    # DIFF:
    
    sds_diff = df_sds1 - df_sds2
    
    # sds_diff.describe()
    #
    #                 1            2  ...           11           12
    # count  8444.000000  8435.000000  ...  8467.000000  8461.000000
    # mean     -0.004641    -0.004232  ...    -0.004790    -0.006567
    # std       0.069047     0.076144  ...     0.070846     0.116145
    # min      -3.308209    -3.617588  ...    -2.471146    -7.444435
    # 25%      -0.000262    -0.000250  ...    -0.000262    -0.000262
    # 50%      -0.000010     0.000004  ...    -0.000011    -0.000005
    # 75%       0.000246     0.000254  ...     0.000243     0.000253
    # max       0.252880     0.113163  ...     0.276401     1.204783

    sds_diff = sds_diff.reset_index(drop=True)
    threshold = 0.2
    mask = np.abs(sds_diff) >= threshold
    
    fig,ax = plt.subplots(figsize=(15,10))
    plt.plot( sds_diff[mask], 's', alpha=0.2, color='red')        
    plt.plot( sds_diff[~mask], 's', alpha=0.2, color='grey', label='abs(diff) < ' + str(threshold))
    plt.title('CRUTEM5 - data SDs(MIN15): abs(diff) >= ' + str(threshold))
    plt.savefig( 'sds-qc-crutem-MIN15-' + str(threshold) +'.png', dpi=300)
    
#------------------------------------------------------------------------------
# SAVE: SDs_qc
#------------------------------------------------------------------------------

print('saving normals ... ')

df_sds.to_pickle( filename_sds, compression='bz2' )
        
#------------------------------------------------------------------------------
print('** END')

