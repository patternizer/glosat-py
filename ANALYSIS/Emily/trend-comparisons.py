#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Piecewise trend comparison: GloSAT v Berkeley-Earth v NTrend
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn import datasets, linear_model

#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------

fontsize=20

#------------------------------------------------------------------------------
# METHODS
#------------------------------------------------------------------------------

def linear_regression_ols(x,y):
    
    regr = linear_model.LinearRegression()
    # regr = TheilSenRegressor(random_state=42)
    # regr = RANSACRegressor(random_state=42)
    
    X = x.values.reshape(len(x),1)
    t = np.linspace(X.min(),X.max(),len(X)) # dummy var spanning [xmin,xmax]        
    regr.fit(X, y)
    ypred = regr.predict(t.reshape(-1, 1))
    slope = regr.coef_
    intercept = regr.intercept_
        
#   return t, ypred, slope, intercept
    return t, ypred

#------------------------------------------------------------------------------
# TREND DATA
#------------------------------------------------------------------------------

df = pd.DataFrame({'midpoint':[], 'glosat':[],'berkeley_qc':[],'berkeley_bc':[],'ntrend':[]})
df['midpoint'] = [1810, 1830, 1850, 1870, 1890, 1910, 1930, 1950, 1970, 1990, 2010]
df['glosat'] = [0.66, -0.23, 0.38, -0.22, 0.21, 0.25, 0.87, -0.07, -0.01, 0.26, 0.46]
df['berkeley_qc'] = [0.66, -0.04, 0.35, -0.21, 0.20, 0.22, 0.90, -0.09, -0.01, 0.27, 0.34]
df['berkeley_bc'] = [0.56, -0.04, 0.35, -0.21, 0.17, 0.17, 0.86, -0.09, -0.20, 0.27, 0.34]
df['ntrend'] = [-0.10, -0.56, 0.13, 0.17, 0.04, 0.33, 0.47, 0.09, 0.14, 0.37, 1.03]

#------------------------------------------------------------------------------
# STATISTICS
#------------------------------------------------------------------------------

# Linear regression

corrcoef1 = scipy.stats.pearsonr(df.glosat,df.berkeley_qc)[0]
corrcoef2 = scipy.stats.pearsonr(df.glosat,df.berkeley_bc)[0]
corrcoef3 = scipy.stats.pearsonr(df.glosat,df.ntrend)[0]

X1, Y1 = linear_regression_ols(df.glosat, df.berkeley_qc)
X2, Y2 = linear_regression_ols(df.glosat, df.berkeley_bc)
X3, Y3 = linear_regression_ols(df.glosat, df.ntrend)

# Mean Absolute Difference (MAD) = PREFERRED

MAD1 = np.mean(np.abs(df.glosat-df.berkeley_qc))   # 0.041
MAD2 = np.mean(np.abs(df.glosat-df.berkeley_bc))   # 0.073
MAD3 = np.mean(np.abs(df.glosat-df.ntrend))        # 0.306

# Root Mean Squared Difference (RMSD)

RMSD1 = np.sqrt(np.mean((df.glosat-df.berkeley_qc)**2))  # 0.070
RMSD2 = np.sqrt(np.mean((df.glosat-df.berkeley_bc)**2))  # 0.098
RMSD3 = np.sqrt(np.mean((df.glosat-df.ntrend)**2))       # 0.367

# mean absolute percentage deviation (MAPD) = sensitive to value of -0.01 in GloSAT trends

MAPD1 = np.mean(np.abs((df.glosat-df.berkeley_qc)/df.glosat))   # 0.158
MAPD2 = np.mean(np.abs((df.glosat-df.berkeley_bc)/df.glosat))   # 1.928
MAPD3 = np.mean(np.abs((df.glosat-df.ntrend)/df.glosat))        # 2.323

#------------------------------------------------------------------------------
# PLOTS
#------------------------------------------------------------------------------

# PLOT: trends

fig,ax = plt.subplots(figsize=(15,10))
plt.axhline(y=0, color='grey', ls='dashed', lw=3)
plt.plot(df.midpoint, df.glosat, marker='.', lw=1, label='glosat')
plt.plot(df.midpoint, df.berkeley_qc,  marker='.', ms=10, lw=1, label='berkeley-qc')
plt.plot(df.midpoint, df.berkeley_bc,  marker='.', ms=10, lw=1, label='berkeley-bc')
plt.plot(df.midpoint, df.ntrend, marker='.', ms=10, lw=1, label='ntrend')
plt.legend(loc='upper left', fontsize=12)
plt.ylabel(r'Trend, $\degree$C / 20 years', fontsize=fontsize)
ax.yaxis.grid(True, which='major')    
ax.tick_params(labelsize=fontsize) 
plt.savefig('trends.png')

# PLOT: trend deltas - not useful here

fig,ax = plt.subplots(figsize=(15,10))
plt.axhline(y=0, color='grey', ls='dashed', lw=3)
plt.plot(df.midpoint, df.glosat.diff(), marker='.', lw=1, label=r'$\Delta$(glosat)')
plt.plot(df.midpoint, df.berkeley_qc.diff(),  marker='.', ms=10, lw=1, label=r'$\Delta$(berkeley-qc)')
plt.plot(df.midpoint, df.berkeley_bc.diff(),  marker='.', ms=10, lw=1, label=r'$\Delta$(berkeley-bc)')
plt.plot(df.midpoint, df.ntrend.diff(), marker='.', ms=10, lw=1, label=r'$\Delta$(ntrend)')
plt.legend(loc='upper left', fontsize=12)
plt.ylabel(r'Sequential change in trend, $\degree$C / 20 years', fontsize=fontsize)
ax.yaxis.grid(True, which='major')    
ax.tick_params(labelsize=fontsize) 
plt.savefig('trend-deltas.png')

# PLOT: differences

fig,ax = plt.subplots(figsize=(15,10))
plt.axhline(y=0, color='grey', ls='dashed', lw=3)
plt.plot(df.midpoint, df.glosat-df.berkeley_qc, marker='.', ms=10, lw=1, label='glosat - berkeley-qc')
plt.plot(df.midpoint, df.glosat-df.berkeley_bc, marker='.', ms=10, lw=1, label='glosat - berkeley-bc')
plt.plot(df.midpoint, df.glosat-df.ntrend, marker='.', ms=10, lw=1, label='glosat - ntrend')
plt.legend(loc='upper left', fontsize=12)
plt.ylabel(r'Trend difference, $\degree$C / 20 years', fontsize=fontsize)
ax.yaxis.grid(True, which='major')    
ax.tick_params(labelsize=fontsize) 
plt.savefig('trend-diffs.png')

# PLOT: correlation

fig,ax = plt.subplots(figsize=(15,10))
xmin = np.min([df.glosat.min(),df.berkeley_qc.min(),df.berkeley_bc.min(),df.ntrend.min()])
xmax = np.max([df.glosat.max(),df.berkeley_qc.max(),df.berkeley_bc.max(),df.ntrend.max()])
plt.plot([xmin,xmax],[xmin,xmax], ls='dashed', lw=3, color='grey')
plt.scatter(df.glosat, df.berkeley_qc, marker='o', s=20, label='glosat v berkeley-qc: corrcoef='+str(np.round(corrcoef1,3)))
plt.scatter(df.glosat, df.berkeley_bc, marker='o', s=20, label='glosat v berkeley-bc: corrcoef='+str(np.round(corrcoef2,3)))
plt.scatter(df.glosat, df.ntrend, marker='o', s=20, label='glosat v ntrend: corrcoef='+str(np.round(corrcoef3,3)))
plt.plot(X1, Y1, lw=1)
plt.plot(X2, Y2, lw=1)
plt.plot(X3, Y3, lw=1)
plt.xlim(xmin,xmax)
plt.ylim(xmin,xmax)
plt.legend(loc='upper left', fontsize=12)
plt.xlabel(r'Trend, $\degree$C / 20 years', fontsize=fontsize)
plt.ylabel(r'Trend, $\degree$C / 20 years', fontsize=fontsize)
ax.xaxis.grid(True, which='major')    
ax.yaxis.grid(True, which='major')    
ax.tick_params(labelsize=fontsize) 
plt.savefig('trend-corr.png')

# PLOT: ratios

fig,ax = plt.subplots(figsize=(15,10))
plt.axhline(y=1, color='grey', ls='dashed', lw=3)
plt.plot(df.midpoint, df.berkeley_qc/df.glosat, marker='.', ms=10, lw=1, label='berkeley-qc / glosat')
plt.plot(df.midpoint, df.berkeley_bc/df.glosat, marker='.', ms=10, lw=1, label='berkeley-bc / glosat')
plt.plot(df.midpoint, df.ntrend/df.glosat, marker='.', ms=10, lw=1, label='ntrend / glosat')
plt.ylim(-2,4)
plt.legend(loc='upper left', fontsize=12)
plt.ylabel(r'Trend ratio', fontsize=fontsize)
ax.yaxis.grid(True, which='major')    
ax.tick_params(labelsize=fontsize) 
plt.savefig('trend-ratios.png')

#------------------------------------------------------------------------------
print('** END')




