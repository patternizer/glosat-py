#------------------------------------------------------------------------------
# PROGRAM: make_full.py
#------------------------------------------------------------------------------
# Version 0.1
# 22 June, 2022
# Michael Taylor
# https://patternizer.github.io
# michael DOT a DOT taylor AT uea DOT ac DOT uk
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
# Dataframe libraries:
import numpy as np
import pandas as pd
import pickle
from datetime import timedelta
# Plotting libraries:
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

tstart, tend = 1781, 2022

#input_file = 'df_anom_qc.pkl'
#output_file = 'df_anom_qc_full.pkl'

input_file = 'df_exposure_bias.pkl'
output_file = 'df_exposure_bias_full.pkl'

#------------------------------------------------------------------------------
# METHODS
#------------------------------------------------------------------------------

def make_timeseries( da, tstart, tend  ):
      
#    t_full = np.arange( tstart, tend + 1 )
#    df_full = pd.DataFrame( {'year':t_full} )
#    db = df_full.merge( da, how='left', on='year' )

    t_full = pd.date_range( start = str(tstart), end = str(tend), freq='MS')[0:-1]
    df_full = pd.DataFrame( {'datetime':t_full} )
    db = df_full.merge( da, how='left', on='datetime' )

    return db

#------------------------------------------------------------------------------
# LOAD: input .pkl file
#------------------------------------------------------------------------------

df = pd.read_pickle( input_file, compression='bz2' )
stationcodes = df.stationcode.unique()

#plt.plot(df.groupby('datetime').mean().bias)

#------------------------------------------------------------------------------
# EXTEND: time axis for each station
#------------------------------------------------------------------------------

ds = []
#for k in range( len( stationcodes ) ):
for k in range( 20 ):
                                
    da = df[ df.stationcode == stationcodes[k] ].reset_index( drop=True )
    da['datetime'] = da['datetime'] -timedelta(days=14)
    
    db = make_timeseries( da, tstart, tend )        

    ############################
    # RESHAPE to monthlies here 
    ############################
    
              
    # APPEND: dataframe to mothership

    ds.append( db )

    if k % 10 == 0: print(k)
    
# CONCATENATE:
    
ds = pd.concat(ds, axis=0)      

# REINDEX:

df_anom_full = ds.reset_index( drop=True )
           
#------------------------------------------------------------------------------
# SAVE: output .pkl file
#------------------------------------------------------------------------------

df_anom_full.to_pickle( output_file, compression='bz2' )

#------------------------------------------------------------------------------
print('** END')




    
