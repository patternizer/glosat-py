#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: load-normals-qc.py
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

base_start, base_end = 1961, 1990

#normals_file = 'normals5.GloSAT.prelim01_FRYuse_ocPLAUS1_iqr3.600reg0.3_19411990_MIN15_OCany_19611990_MIN15_PERDEC00_NManySDreq.txt'
#normals_file = 'normals5.GloSAT.prelim02_FRYuse_ocPLAUS1_iqr3.600reg0.3_19411990_MIN15_OCany_19611990_MIN15_PERDEC00_NManySDreq.txt'
#normals_file = 'normals5.GloSAT.prelim03_FRYuse_ocPLAUS1_iqr3.600reg0.3_19411990_MIN15_OCany_19611990_MIN15_PERDEC00_NManySDreq.txt'
normals_file = 'normals5.GloSAT.prelim04_FRYuse_ocPLAUS1_iqr3.600reg0.3_19411990_MIN15_OCany_19611990_MIN15_PERDEC00_NManySDreq.txt'

load_df_normals = False # ( default = False ) True --> load pre-calculated pkl
check_normals = False   # ( default = False ) True --> compare loaded vs MIN15 data-driven normals
float64 = True          # ( detault = True  ) False --> float32

if float64 == True: 
    precision = 'float64' 
else: 
    precision = 'float32'
    
filename_normals = 'df_normals_qc.pkl'
filename_temp = 'df_temp_qc.pkl'

#------------------------------------------------------------------------------
# LOAD normals
#------------------------------------------------------------------------------

if load_df_normals == True:

    print('loading normals ...')

    df_normals = pd.read_pickle( filename_normals, compression='bz2' )    

else:

    print('extracting normals ...')

	#    1 ID
	#    2-3 First year, last year
	#    4-5 First year for normal, last year for normal
	#    6-17 Twelve normals for Jan to Dec, degC (not degC*10), missing normals are -999.000
	#    18 Source code (1=missing, 2=Phil's estimate, 3=WMO, 4=calculated here, 5=calculated previously)
	#    19-30 Twelve values giving the % data available for computing each normal from the 1961-1990 period

    f = open(normals_file)
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
    
    normal12s = np.array(normal12s).astype( precision )
    normalpercentage12s = np.array(normalpercentage12s).astype(int)

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

#------------------------------------------------------------------------------
# MATCH: df_temp_qc stationcodes
#------------------------------------------------------------------------------

print('matching stationcodes ... ')

df_temp = pd.read_pickle( filename_temp, compression='bz2' )    

stationcodes_df_temp = df_temp.stationcode.unique()
stationcodes_df_normals = df_normals.stationcode
idx = list( set(stationcodes_df_normals) - set(stationcodes_df_temp) )  # stationcodes in df_temp but not in df_normals

for i in range(len(idx)):    
    rows = df_normals[ df_normals.stationcode == idx[i] ].index
    df_normals =  df_normals.drop( rows )
    
#------------------------------------------------------------------------------
# REPLACE: fill value -999 with np.nan
#------------------------------------------------------------------------------

for j in range(1,13): df_normals[ str(j) ].replace(-999, np.nan, inplace=True)

#------------------------------------------------------------------------------
# CHECK: normals
#------------------------------------------------------------------------------
            
if check_normals == True:
		
    print('checking normals ...')
            
    # EXTRACT: CRUTEM5 station normals

    df_normals1 = df_normals.copy()
    df_normals1.index = df_normals1['stationcode']
    df_normals1 = df_normals1[ df_normals1.columns[6:18] ]
    df_normals1 = df_normals1.dropna()
        
    # COMPUTE: normals from quality-controlled df_temp

    df_normals2 = df_temp.copy()
    df_normals2_means = df_normals2[ (df_normals2['year']>=base_start) & (df_normals2['year']<=base_end)].groupby(['stationcode']).mean().iloc[:,1:13] # climatological means
    df_normals2_mask = df_normals2[ (df_normals2['year']>=base_start) & (df_normals2['year']<=base_end)].groupby(['stationcode']).count().iloc[:,1:13] >= 15 # boolean mask for n>=15 in baseline
    df_normals2 = df_normals2_means[df_normals2_mask] # normals
    df_normals2 = df_normals2.astype( precision )
        
    # DIFF:
    
    normals_diff = df_normals1 - df_normals2
    
    # normals_diff.describe()
    #
    #                  1            2  ...           11           12
    # count  8027.000000  8027.000000  ...  8024.000000  8026.000000
    # mean      0.000530     0.000435  ...     0.000957     0.001367
    # std       0.030335     0.028640  ...     0.035101     0.046110
    # min      -0.636448    -0.758435  ...    -0.963334    -1.020000
    # 25%      -0.000333    -0.000333  ...    -0.000333    -0.000333
    # 50%       0.000000     0.000000  ...     0.000000     0.000000
    # 75%       0.000333     0.000333  ...     0.000333     0.000333
    # max       1.637000     1.276667  ...     1.203333     2.596000

    normals_diff = normals_diff.reset_index(drop=True)
    threshold = 0.2
    mask = np.abs(normals_diff) >= threshold
    
    fig,ax = plt.subplots(figsize=(15,10))
    plt.plot( normals_diff[mask], 's', alpha=0.2, color='red')        
    plt.plot( normals_diff[~mask], 's', alpha=0.2, color='grey', label='abs(diff) < ' + str(threshold))
    plt.title('CRUTEM5 - data normals(MIN15): abs(diff) >= ' + str(threshold))
    plt.savefig( 'normals-qc-crutem-MIN15-' + str(threshold) +'.png', dpi=300)
    
#------------------------------------------------------------------------------
# SAVE: normals_qc
#------------------------------------------------------------------------------

print('saving normals ... ')

df_normals.to_pickle( filename_normals, compression='bz2' )
        
#------------------------------------------------------------------------------
print('** END')

