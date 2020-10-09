#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: quicklook_dat.py
#------------------------------------------------------------------------------
# Version 0.1
# 9 October, 2020
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
# Dataframe libraries:
import numpy as np
import pandas as pd

# Plotting libraries:
import matplotlib.pyplot as plt; plt.close('all')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

fontsize = 20
filename = 'meteo-france-diff-test.dat'
nheader = 1
ncol = 2 # number of columns in legend

#------------------------------------------------------------------------------
# LOAD: .dat file
#------------------------------------------------------------------------------

f = open(filename)
lines = f.readlines()    
dates = []
vals = []
for i in range(nheader,len(lines)):    
    words = lines[i].split()
    if len(words) > 1:                    
        date = float(words[0])
        val = (len(words)-1)*[None]
        for j in range(len(val)):
            try: val[j] = float(words[j+1])
            except: pass
            dates.append(date)
            vals.append(val) 
f.close()    
dates = np.array(dates).astype('int')
vals = np.array(vals)

df = pd.DataFrame(columns=['year','1','2','3','4','5','6','7','8','9','10','11','12'])
df['year'] = dates
for j in range(1,13):   
    df[df.columns[j]] = [ vals[i][j-1] for i in range(len(df)) ]

#------------------------------------------------------------------------------
# PLOT: dataframe
#------------------------------------------------------------------------------

figstr = filename+'_'+'.png'
titlestr = filename
xstr = 'Year'
ystr = 'Difference, °C'
scale_factor = 1/10

fig, ax = plt.subplots(figsize=(15,10))          
for i in range(1,13):    
    plt.step(df['year'], df.iloc[:,i]*scale_factor, label='Month '+str(i))
plt.tick_params(labelsize=fontsize)    
plt.legend(loc='lower left', ncol=ncol, fontsize=12)
plt.xlabel(xstr, fontsize=fontsize)
plt.ylabel(ystr, fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr)
plt.close(fig)
        
#------------------------------------------------------------------------------
print('** END')

