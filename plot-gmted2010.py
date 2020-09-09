#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: app.py
#------------------------------------------------------------------------------
# Version 0.2
# 9 September, 2020
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
# Numerics and dataframe libraries:
import numpy as np
import numpy.ma as ma
from scipy.interpolate import griddata
from scipy import spatial
from mod import Mod
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import xarray as xr
import pickle
# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.cm as cm
import matplotlib.colors as c
from matplotlib.colors import Normalize
from matplotlib import colors as mcol
from matplotlib.cm import ScalarMappable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
matplotlib.rcParams['text.usetex'] = False
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.collections import PolyCollection
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.tri as tri
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
# METHODS
#------------------------------------------------------------------------------

# Haversine Function: 

def haversine(lat1, lon1, lat2, lon2):

    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 

    # Radius of earth is 6371 km
    km = 6371* c
    return km

# Find nearest cell in array

def find_nearest(lat, lon):
    distances = df.apply(
        lambda row: haversine(lat, lon, row['lat'], row['lon']), 
        axis=1)
    return df.loc[distances.idxmin(), 'name']

# Extract lat, lon, value from netcdf 2D array

def ExtractVarsFromNetcdf(x_coord, y_coord, ncfile, varnames):
    """   
    @params:
        x_coord    - lon
        y_coord    - lat
        ncfile     - netcdf file
        varnames   - netcdf variables
    """
    with Dataset(ncfile, "r") as nc:

        # Get the nc lat and lon from the point's (x,y)
        lon = nc.variables["lon"][int(round(x_coord))]
        lat = nc.variables["lat"][int(round(y_coord))]

        # Return a np.array with the netcdf data
        nc_data = np.ma.getdata(
            [nc.variables[varname][:, x_coord, y_coord] for varname in varnames]
        )
        return nc_data, lon, lat
        
#------------------------------------------------------------------------------
# IMPORT GloSATp01 and GMTED2010: 
#------------------------------------------------------------------------------

fontsize = 12
use_minmax = False
use_horizontal_colorbar = True
projection = 'equalearth'
cmap = 'viridis'
#cmap = 'coolwarm'

df_temp = pd.read_pickle('df_temp.pkl', compression='bz2')
stationlon = df_temp['stationlon']
stationlat = df_temp['stationlat']
stationelevation = df_temp['stationelevation']

gb = df_temp.groupby(['stationlat','stationlon','stationelevation'])['stationcode'].unique().reset_index()
stationcode = gb['stationcode']
stationlat = gb['stationlat']
stationlon = gb['stationlon']
stationelevation = gb['stationelevation']

filename_gmted2010 = 'GMTED2010_15n015_00625deg.nc'
ds = xr.open_dataset(filename_gmted2010, decode_cf=True) 
lat = np.array(ds.latitude)
lon = np.array(ds.longitude)
elevation = np.array(ds.elevation)

#-------------------------
# STATION HAVERSINE CHECK:
#-------------------------

print('plotting GloSAT haversine distance from (0,0) ...')

x = stationlon
y = stationlat
z = [haversine(stationlon[i],stationlat[i],0,0) for i in range(len(stationcode))]

filestr = "glosat-haversine-check.png"
titlestr = 'GloSATp01: haversine check'
colorbarstr = 'Haversine distance from (0,0) [km]'

fig,ax = plt.subplots(figsize=(15,10))
plt.scatter(x=x, y=y, c=z)
plt.tick_params(labelsize=fontsize)
ax.xaxis.grid(True, which='major')        
ax.yaxis.grid(True, which='major')        
plt.xlabel('Longitude', fontsize=fontsize)
plt.ylabel('Latitude', fontsize=fontsize)
plt.title(titlestr)
if use_horizontal_colorbar == True:
    cb = plt.colorbar(orientation="horizontal", shrink=0.5, extend='both')
    cb.set_label(colorbarstr, labelpad=25, fontsize=fontsize)
else:
    cb = plt.colorbar(shrink=0.5, extend='both')
    cb.set_label(colorbarstr, rotation=270, labelpad=25, fontsize=fontsize)
plt.savefig(filestr)
plt.close('all')

#-------------------------
# STATION ELEVATION CHECK:
#-------------------------

print('plotting GloSAT elevation scatter map ...')

x = stationlon
y = stationlat
z = stationelevation

filestr = "glosat-elevation-scatter.png"
titlestr = 'GloSATp01: station elevations, AMLS [m]'
colorbarstr = 'Elevation, AMSL [m]'

fig,ax = plt.subplots(figsize=(15,10))
plt.scatter(x=x, y=y, c=z)
plt.tick_params(labelsize=fontsize)
ax.xaxis.grid(True, which='major')        
ax.yaxis.grid(True, which='major')        
plt.xlabel('Longitude', fontsize=fontsize)
plt.ylabel('Latitude', fontsize=fontsize)
plt.title(titlestr)
if use_horizontal_colorbar == True:
    cb = plt.colorbar(orientation="horizontal", shrink=0.5, extend='both')
    cb.set_label(colorbarstr, labelpad=25, fontsize=fontsize)
else:
    cb = plt.colorbar(shrink=0.5, extend='both')
    cb.set_label(colorbarstr, rotation=270, labelpad=25, fontsize=fontsize)
plt.savefig(filestr)
plt.close('all')

#------------------------
# FIND CLOSEST GMTED2010:
#------------------------

X,Y = np.meshgrid(lon,lat)
Z = np.array(ds.elevation)
N = len(lon)*len(lat)

# create 1D-arrays from 2D-arrays
x = X.reshape(N)
y = Y.reshape(N)
z = Z.reshape(N)

# Put the GMTED2010 and GloSATp01 lat,lon,elev data into Oandas DataFrames
df = pd.DataFrame({'stationlon':stationlon, 'stationlat':stationlat, 'stationelevation':stationelevation}, index=range(len(stationlat))) 
dg = pd.DataFrame({'lon':x, 'lat':y, 'elevation':z}, index=range(N))

# TEST DIFFERENT METHODS

pt = [51,1]  # <-- the point to find (lat,lon)

# METHOD 1: Find minimum Haversine (accurate but very slow)
def find_nearest(lat, lon):
    distances = dg.apply(lambda row: haversine(lat, lon, row['lat'], row['lon']), axis=1)
    return dg.loc[distances.idxmin(),:]

query = find_nearest(pt[0],pt[1]) # --> TOO SLOW!
lati = query['lat']
loni = query['lon']
elevi = query['elevation']

# METHOD 2: Find closest sorted lat,lon (fast but error prone)
def find_index(x,y):
    xi=np.searchsorted(lat,x)
    yi=np.searchsorted(lon,y)
    return xi,yi

query = find_index(pt[0],pt[1])
lati = lat[query[0]]
loni = lon[query[1]]
elevi = elevation[query[0],query[1]]

# METHOD 3: Find closest using SciPy Spatial Tree (fast once tree built and accurate)
from scipy import spatial
#A = np.random.random((10000000,2))*100
A = list(zip(*map(dg.get, ['lat', 'lon'])))
tree = spatial.KDTree(A)

latn = []
lonn = []
elevationn = []
distancen = []
for i in range(len(df)):
    pt = [df.loc[i]['stationlat'],df.loc[i]['stationlon']]
    distance,index = tree.query(pt)    
    lati = dg.loc[index,:]['lat']
    loni = dg.loc[index,:]['lon']
    elevationi = dg.loc[index,:]['elevation']    
    distancei = distance
    latn.append(lati)
    lonn.append(loni)
    elevationn.append(elevationi)
    distancen.append(distancei)

df['gmtedlon']=lonn
df['gmtedlat']=latn
df['gmtedelevation']=elevationn
df['gmteddistance']=distancen

df.to_pickle('df_nearest.pkl', compression='bz2')
        
#--------------------------------------------------------
# PLOT ELEVATION MAP: GloSATp01: delaunay intepolation
#--------------------------------------------------------

print('plotting GloSAT elevation delaunay map ...')

x = stationlon
y = stationlat
z = stationelevation

#x = np.array(x).astype('float32')
#y = np.array(y).astype('float32')
#z = np.array(z).astype('float32')

xyz = {'x': x, 'y': y, 'z': z}
df = pd.DataFrame(xyz, index=range(len(xyz['x']))) 

# Create 2D-arrays
x1 = np.linspace(df['x'].min(), df['x'].max(), len(df['x'].unique()))
y1 = np.linspace(df['y'].min(), df['y'].max(), len(df['y'].unique()))
x2, y2 = np.meshgrid(x1, y1)
z2 = griddata((df['x'], df['y']), df['z'], (x2, y2), method='cubic')

# Set np.nan to 0.0 for open sea
z2[z2==np.nan] = 0.0

filestr = "glosat-elevation-delaunay.png"
titlestr = 'GloSATp01: delaunay interpolated (cubic) elevations, AMLS [m]'
colorbarstr = 'Elevation, AMSL [m]'

fig,ax  = plt.subplots(figsize=(15,10))
#cntr = plt.contourf(x2, y2, z2, level=100, cmap='viridis')
#cntr = ax.tricontour(x, y, z, levels=5, linewidths=0.5, colors='k')
cntrf = ax.tricontourf(x, y, z, levels=100, cmap="viridis")
ax.plot(x, y, 'ko', ms=0.5)
plt.xlim(-180,180)
plt.ylim(-90,90)
ax.xaxis.grid(True, which='major')        
ax.yaxis.grid(True, which='major')        
plt.tick_params(labelsize=fontsize)
plt.xlabel('Longitude', fontsize=fontsize)
plt.ylabel('Latitude', fontsize=fontsize)
plt.title(titlestr)
cb = plt.colorbar(cntrf, ax=ax, orientation="horizontal", shrink=0.5, extend='both')
cb.set_label(colorbarstr, labelpad=25, fontsize=fontsize)
#if use_horizontal_colorbar == True:
#    cb = fig.colorbar(cntr, ax=ax, orientation="horizontal", shrink=0.5, extend='both')
#    cb.set_label(colorbarstr, labelpad=25, fontsize=fontsize)
#else:
#    cb = fig.colorbar(cntr, ax=ax, shrink=0.5, extend='both')
#    cb.set_label(colorbarstr, rotation=270, labelpad=25, fontsize=fontsize)
plt.savefig(filestr)
plt.close('all')

#--------------------------------------------------------
# PLOT ELEVATION MAP: GloSATp01:
#--------------------------------------------------------

print('plotting GloSAT elevation map ...')
 
x = stationlon
y = stationlat
z = stationelevation

#x = np.array(x).astype('float32')
#y = np.array(y).astype('float32')
#z = np.array(z).astype('float32')

xyz = {'x': x, 'y': y, 'z': z}
df = pd.DataFrame(xyz, index=range(len(xyz['x']))) 

# Create 2D-arrays
x1 = np.linspace(df['x'].min(), df['x'].max(), len(df['x'].unique()))
y1 = np.linspace(df['y'].min(), df['y'].max(), len(df['y'].unique()))
x2, y2 = np.meshgrid(x1, y1)
z2 = griddata((df['x'], df['y']), df['z'], (x2, y2), method='cubic')

x = x2
y = y2
v = z2

filestr = "glosat-elevation-map.png"
titlestr = 'GloSATp01: elevations, AMLS [m]'
colorbarstr = 'Elevation, AMSL [m]'

projection = 'platecarree'

fig  = plt.figure(figsize=(15,10))

if projection == 'platecarree': p = ccrs.PlateCarree(central_longitude=0); threshold = 0
if projection == 'mollweide': p = ccrs.Mollweide(central_longitude=0); threshold = 1e6
if projection == 'robinson': p = ccrs.Robinson(central_longitude=0); threshold = 0
if projection == 'equalearth': p = ccrs.EqualEarth(central_longitude=0); threshold = 0
if projection == 'geostationary': p = ccrs.Geostationary(central_longitude=0); threshold = 0
if projection == 'goodehomolosine': p = ccrs.InterruptedGoodeHomolosine(central_longitude=0); threshold = 0
if projection == 'europp': p = ccrs.EuroPP(); threshold = 0
if projection == 'northpolarstereo': p = ccrs.NorthPolarStereo(); threshold = 0
if projection == 'southpolarstereo': p = ccrs.SouthPolarStereo(); threshold = 0
if projection == 'lambertconformal': p = ccrs.LambertConformal(central_longitude=0); threshold = 0

ax = plt.axes(projection=p)
#ax.stock_img()
ax.coastlines()
#ax.coastlines(resolution='50m')
#ax.add_feature(cf.RIVERS.with_scale('50m'))
#ax.add_feature(cf.BORDERS.with_scale('50m'))
#ax.add_feature(cf.LAKES.with_scale('50m'))
ax.gridlines()
        
g = ccrs.Geodetic()
trans = ax.projection.transform_points(g, x, y)
x0 = trans[:,:,0]
x1 = trans[:,:,1]
    
if projection == 'platecarree':   
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
    for mask in (x0>threshold,x0<=threshold):            
        im = ax.pcolor(ma.masked_where(mask, x), ma.masked_where(mask, y), ma.masked_where(mask, v), vmin=vmin, vmax=vmax, transform=ax.projection, cmap=cmap)
else:
    for mask in (x0>threshold,x0<=threshold):
        im = ax.pcolor(ma.masked_where(mask, x0), ma.masked_where(mask, x1), ma.masked_where(mask, v), vmin=vmin, vmax=vmax, transform=ax.projection, cmap=cmap) 
im.set_clim(vmin,vmax)
if use_horizontal_colorbar == True:
    cb = plt.colorbar(im, orientation="horizontal", shrink=0.5, extend='both')
    cb.set_label(colorbarstr, labelpad=25, fontsize=fontsize)
else:
    cb = plt.colorbar(im, shrink=0.5, extend='both')
    cb.set_label(colorbarstr, rotation=270, labelpad=25, fontsize=fontsize)

plt.title(titlestr, fontsize=fontsize)
plt.savefig(filestr)
plt.close('all')

#--------------------------------------------------------
# PLOT ELEVATION MAP: GMTED2010: 0.0625 degree resolution
#--------------------------------------------------------

print('plotting GMTED2010 elevation map ...')

v = elevation[::10,::10]
x, y = np.meshgrid(lon[::10], lat[::10])
    
vmin = np.min(v)
vmax = np.max(v)

filestr = "GMTED2010_15n015_00625deg.png"
titlestr = 'GMTED2010: 0.0625 [°]'
colorbarstr = 'Elevation, AMSL [m]'
projection = 'platecarree'

fig  = plt.figure(figsize=(15,10))

if projection == 'platecarree': p = ccrs.PlateCarree(central_longitude=0); threshold = 0
if projection == 'mollweide': p = ccrs.Mollweide(central_longitude=0); threshold = 1e6
if projection == 'robinson': p = ccrs.Robinson(central_longitude=0); threshold = 0
if projection == 'equalearth': p = ccrs.EqualEarth(central_longitude=0); threshold = 0
if projection == 'geostationary': p = ccrs.Geostationary(central_longitude=0); threshold = 0
if projection == 'goodehomolosine': p = ccrs.InterruptedGoodeHomolosine(central_longitude=0); threshold = 0
if projection == 'europp': p = ccrs.EuroPP(); threshold = 0
if projection == 'northpolarstereo': p = ccrs.NorthPolarStereo(); threshold = 0
if projection == 'southpolarstereo': p = ccrs.SouthPolarStereo(); threshold = 0
if projection == 'lambertconformal': p = ccrs.LambertConformal(central_longitude=0); threshold = 0

ax = plt.axes(projection=p)
#ax.stock_img()
ax.coastlines()
#ax.coastlines(resolution='50m')
#ax.add_feature(cf.RIVERS.with_scale('50m'))
#ax.add_feature(cf.BORDERS.with_scale('50m'))
#ax.add_feature(cf.LAKES.with_scale('50m'))
ax.gridlines()
        
g = ccrs.Geodetic()
trans = ax.projection.transform_points(g, x, y)
x0 = trans[:,:,0]
x1 = trans[:,:,1]
    
if projection == 'platecarree':   
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
    for mask in (x0>threshold,x0<=threshold):            
        im = ax.pcolor(ma.masked_where(mask, x), ma.masked_where(mask, y), ma.masked_where(mask, v), vmin=vmin, vmax=vmax, transform=ax.projection, cmap=cmap)
else:
    for mask in (x0>threshold,x0<=threshold):
        im = ax.pcolor(ma.masked_where(mask, x0), ma.masked_where(mask, x1), ma.masked_where(mask, v), vmin=vmin, vmax=vmax, transform=ax.projection, cmap=cmap) 
im.set_clim(vmin,vmax)
if use_horizontal_colorbar == True:
    cb = plt.colorbar(im, orientation="horizontal", shrink=0.5, extend='both')
    cb.set_label(colorbarstr, labelpad=25, fontsize=fontsize)
else:
    cb = plt.colorbar(im, shrink=0.5, extend='both')
    cb.set_label(colorbarstr, rotation=270, labelpad=25, fontsize=fontsize)

plt.title(titlestr, fontsize=fontsize)
plt.savefig(filestr)
plt.close('all')
    
#----------------------------------------------------
print('** END')    



