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

use_datetime = False 	# ( default = False ) False --> yearly rows
use_monthstart = True 	# ( default = True  ) False --> mid-month datetimes

input_file = 'df_temp_qc.pkl'
output_file = 'df_temp_qc_full.pkl'

#input_file = 'df_anom_qc.pkl'
#output_file = 'df_anom_qc_full.pkl'

#input_file = 'df_temp_ebc.pkl'
#output_file = 'df_temp_ebc_full.pkl'

#input_file = 'df_ebc.pkl'
#output_file = 'df_ebc_full.pkl'

#------------------------------------------------------------------------------
# METHODS
#------------------------------------------------------------------------------

def make_timeseries( da, tstart, tend  ):
      
    if use_datetime == False:
          
        t_full = np.arange( tstart, tend + 1 )
        df_full = pd.DataFrame( {'year':t_full} )
        db = df_full.merge( da, how='left', on='year' )
        
    else:

        t_full = pd.date_range( start = str(tstart), end = str(tend), freq='MS')[0:-1]
        df_full = pd.DataFrame( {'datetime':t_full} )
        db = df_full.merge( da, how='left', on='datetime' )

    return db

#------------------------------------------------------------------------------
# LOAD: input .pkl file
#------------------------------------------------------------------------------

df = pd.read_pickle( input_file, compression='bz2' )
stationcodes = df.stationcode.unique()

#------------------------------------------------------------------------------
# EXTEND: time axis for each station
#------------------------------------------------------------------------------

ds = []
for k in range( len( stationcodes ) ):
                                
    da = df[ df.stationcode == stationcodes[k] ].reset_index( drop=True )        

    if use_datetime == True:
        if use_monthstart == True:    
            da['datetime'] = da['datetime'] - timedelta(days=14)
    
    db = make_timeseries( da, tstart, tend )        
              
    # APPEND: dataframe to mothership

    ds.append( db )

    if k % 10 == 0: print(k)
    
# CONCATENATE:
    
ds = pd.concat(ds, axis=0)      

# REINDEX:

ds_full = ds.reset_index( drop=True )
           
#------------------------------------------------------------------------------
# SAVE: output .pkl file
#------------------------------------------------------------------------------

ds_full.to_pickle( output_file, compression='bz2' )

#------------------------------------------------------------------------------
print('** END')




    
