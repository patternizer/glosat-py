#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: load-stations-qc.py
#------------------------------------------------------------------------------
# Version 0.2
# 27 May, 2022
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
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

#filename_txt = 'stat4.CRUTEM5.1prelim01.1721-2019.txt'
#filename_txt = 'stat4.GloSATprelim02.1658-2020.txt'
#filename_txt = 'stat4.GloSATprelim03.1658-2020.txt'
filename_txt = 'stat4.GloSATprelim04.1658-2021.txt'
#filename_txt = 'stat4.postqc.GloSAT.prelim04_FRYuse_ocPLAUS1_iqr3.600reg0.3_19411990_MIN15_OCany_19611990_MIN15_PERDEC00_NManySDreq.txt' # 1658-2021

# WMO extreme values:
# https://wmo.asu.edu/content/world-meteorological-organization-global-weather-climate-extremes-archive
    
global_monthly_min = -89.2 # 21 July 1983: Vostok, Antarctica
global_monthly_max = +56.7 # 10 July 1913: Furnace Creek (Greenland Ranch), CA, USA

base_start, base_end = 1961, 1990

load_df = False               # ( default = False ) --> load pre-calculated pkl
use_pickle = True             # ( default = True  ) False --> CSV

include_cru = False           # ( default = False )
fix_names = True              # ( default = True  ) 
fix_countries = True          # ( default = True  )
fix_unphysical = True         # ( default = True  ) 
require_elevation = False     # ( default = False )
require_geolocation = True    # ( default = True  )

filename_pkl = 'df_temp_qc.pkl'
filename_csv = 'df_temp_qc.csv'

#==============================================================================
# METHODS
#==============================================================================
        
def load_dataframe(filename_txt):
    
    #------------------------------------------------------------------------------
    # I/O: filename_txt
    #------------------------------------------------------------------------------

    # load .txt file (comma separated) into pandas dataframe

    # station header sample:    
    # [067250 464  -78 1538 BLATTEN, LOETSCHENTA SWITZERLAND   20012012  982001    9999]
        
    # station data sample:
    # [1921  -44  -71  -68  -46  -12   17   42   53   27  -20  -21  -40]

    yearlist = []
    monthlist = []
    stationcode = []
    stationlat = []
    stationlon = []
    stationelevation = []
    stationname = []
    stationcountry = []
    stationfirstyear = []
    stationlastyear = []
    stationsource = []
    stationfirstreliable = []
    stationfirstreliable = []
    stationcruindex = []
    stationgridcell = []
    
    with open (filename_txt, 'r', encoding="ISO-8859-1") as f:  
                    
        for line in f:   

            if len(line)>1: # ignore empty lines  
       
                if (len(line.split()[0])>4): # header lines
                    
                    # Station Header File (IDL numbering): 
                    # https://crudata.uea.ac.uk/cru/data/temperature/crutem4/station-data.htm
                    #
                    #    (ch. 1-6) World Meteorological Organization (WMO) Station Number with a single additional character making a field of 6 integers. WMO numbers comprise a 5 digit sub-field, where the first two digits are the country code and the next three digits designated by the National Meteorological Service (NMS). Some country codes are not used. If the additional sixth digit is a zero, then the WMO number is or was an official WMO number. If the sixth digit is not zero then the station does not have an official WMO number and an alternative number has been assigned by CRU. Two examples are given below. Many additional stations are grouped beginning 99****. Station numbers in the blocks 72**** to 75**** are additional stations in the United States.
                    #    (ch. 7-10) Station latitude in degrees and tenths (-999 is missing), with negative values in the Southern Hemisphere
                    #    (ch. 11-15) Station longitude in degrees and tenths (-1999 is missing), with negative values in the Eastern Hemisphere (NB this is opposite to the more usual convention)                    
                    #    (ch. 16-20) Station Elevation in metres (-999 is missing)                    
                    #    (ch. 22-41) Station Name                    
                    #    (ch. 43-55) Country                    
                    #    (ch. 57-60) First year of monthly temperature data                    
                    #    (ch. 61-64) Last year of monthly temperature data                    
                    #    (ch. 67-68) Data Source (see below)                    
                    #    (ch. 69-72) First reliable year (generally the same as the first year)                    
                    #    (ch. 73-76) Unique index number (internal use)                    
                    #    (ch. 77-80) Index into the 5° x 5° gridcells (internal use)  
                    #
                    # NB: char positions differ due to empty space counts difference
                
                    code = line[0:6].strip()                
                    lat = line[6:10].strip()
                    lon = line[10:15].strip()
                    elevation = line[15:20].strip()
                    name = line[21:41].strip()
                    country = line[42:55].strip()
                    firstyear = line[56:60].strip()
                    lastyear = line[60:64].strip()
                    source = line[66:68].strip()
                    firstreliable = line[68:72].strip()
                    cruindex = line[72:76].strip()
                    gridcell = line[76:80].strip()
                                                
                else:           
                    yearlist.append( int( line.strip().split()[0] ) )                                 
                    monthlist.append( np.array( line.strip().split()[1:] ) )     
                            
                    stationcode.append( code )
                    stationlat.append( lat )
                    stationlon.append( lon )
                    stationelevation.append( elevation )
                    stationname.append( name )
                    stationcountry.append( country )
                    stationfirstyear.append( firstyear )
                    stationlastyear.append( lastyear )
                    stationsource.append( source )
                    stationfirstreliable.append( firstreliable )
                    stationcruindex.append( cruindex )
                    stationgridcell.append( gridcell )

            else:                
                continue
    f.close

    # CONSTRUCT: dataframe
    
    df = pd.DataFrame(columns=['year','1','2','3','4','5','6','7','8','9','10','11','12'])
    df['year'] = yearlist    
    df['stationcode'] = stationcode    
    df['stationlat'] = stationlat
    df['stationlon'] = stationlon
    df['stationelevation'] = stationelevation
    df['stationname'] = stationname
    df['stationcountry'] = stationcountry
    df['stationfirstyear'] = stationfirstyear
    df['stationlastyear'] = stationlastyear
    df['stationsource'] = stationsource
    df['stationfirstreliable'] = stationfirstreliable

    if include_cru == True:

        df['stationcruindex'] = stationcruindex 
        df['stationgridcell'] = stationgridcell

    #==============================================================================
    # QUALITY CONTROL
    #==============================================================================

    # MONTHLY DATA: replace monthlists not containing 12 monthly integer values with row of '-999' (for float conversion via NaN)

    print( 'fixing monthlist without 12 values ... ')
        
    a = np.array([ monthlist[i].shape for i in range(len(monthlist)) ]).ravel(); b = np.arange(len(a))
    idx = b[a!=12]
    for i in range(len(idx)):			
        monthlist[ idx[i] ] = np.array(['-999']*12)
#   repair = monthlist[ idx[0] ] 
#   monthlist[ idx[0] ] = np.array( list(repair[0:3].ravel()) + list(['91','-999']) + list(repair[4:].ravel()) )
    
    # CODE: coerce station codes to be 6 digit by adding leading 0 to 5-digit codes

    print( 'fixing station codes without 6-digits ... ')

    a = df['stationcode'].values; b = np.arange(len(a))
    idx = []
    for i in range(len(a)):		
        if len(a[i]) < 5: idx.append( b[i] )
        elif len(a[i]) == 5: df['stationcode'].loc[i] = str(df['stationcode'].loc[i]).zfill(6)
    if len(idx) > 0: df['stationcode'].loc[idx] = '-999'
    
    # LAT: replace missing station lat with '-999' (for float conversion via NaN)

    print( 'replacing missing lats with -999 ... ')

    a = np.array( df.stationlat ); b = np.arange(len(a))
    mask = np.array( [ a[i] == '' for i in range(len(a)) ] )
    idx = b[mask]
    df['stationlat'].loc[idx] = '-999'

    # LON: replace missing station lon with '-9999' (for float conversion via NaN)

    print( 'replacing missing lons with -9999 ... ')

    a = np.array( df.stationlon ); b = np.arange(len(a))
    mask = np.array( [ a[i] == '' for i in range(len(a)) ] )
    idx = b[mask]
    df['stationlon'].loc[idx] = '-9999'
    
    # ELEVATION: replace missing station elevation with '-9999' (for float conversion via NaN)

    print( 'replacing missing elevations with -9999 ... ')

    a = np.array( df.stationelevation ); b = np.arange(len(a))
    mask = np.array( [ a[i] == '' for i in range(len(a)) ] )
    idx = b[mask]
    df['stationelevation'].loc[idx] = '-9999'

    # NAMES: remove odd characters from station names

    if fix_names == True:
        
        print('fixing names ...')

        a = df['stationname']
#       b = pd.Series( [ re.sub('[^A-Za-z0-9 ]+', '', a.values[i]) for i in range(len(a)) ] )
        b = pd.Series( [ re.sub('[^A-Za-z0-9]+', ' ', a.values[i]) for i in range(len(a)) ] )
        df['stationname'] = pd.Series( b )

    # COUNTRIES: remove odd characters from station countries

    if fix_countries == True:
        
        print('fixing countries ...')

        a = df['stationcountry']
        b = pd.Series( [ re.sub('[^A-Za-z]+', ' ', a.values[i]) for i in range(len(a)) ] )
        df['stationcountry'] = b
        
    # FIRSTYEAR: replace missing station firstyear with '-9999'

    print( 'replacing missing firstyear with -9999 ... ')

    a = np.array( df.stationfirstyear ); b = np.arange(len(a))
    mask = np.array( [ a[i] == '' for i in range(len(a)) ] )
    idx = b[mask]
    df['stationfirstyear'].loc[idx] = '-9999'

    # LASTYEAR: replace missing station lastyear with '-9999'

    print( 'replacing missing lastyear with -9999 ... ')

    a = np.array( df.stationlastyear ); b = np.arange(len(a))
    mask = np.array( [ a[i] == '' for i in range(len(a)) ] )
    idx = b[mask]
    df['stationlastyear'].loc[idx] = '-9999'

    # SOURCE: replace missing station source code with '-99'

    print( 'replacing missing stationsource with -99 ... ')

    a = np.array( df.stationsource ); b = np.arange(len(a))
    mask = np.array( [ a[i] == '' for i in range(len(a)) ] )
    idx = b[mask]
    df['stationsource'].loc[idx] = '-99'

    # FIRSTRELIABLEYEAR: replace missing station firstreliable with station firstyear
    
    print( 'replacing missing stationfirstreliable with -9999 ... ')

    a = np.array( df.stationfirstreliable ); b = np.arange(len(a))
    mask = np.array( [ a[i] == '' for i in range(len(a)) ] )
    idx = b[mask]
    df['stationfirstreliable'].loc[idx] = df['stationfirstyear'].loc[idx]

    if include_cru == True:

        # CRUINDEX: set non-numeric station cruindex to -9999
        
        print('replacing missing stationcruindex with -9999 ... ')

        a = np.array( df.stationcruindex ); b = np.arange(len(a))
        mask_inverse = np.array( [ a[i].isdigit() for i in range(len(a)) ] )
        mask = np.invert( mask_inverse )
        idx = b[mask]
        df['stationcruindex'].loc[idx] = '-9999'
        df['stationcruindex'] = df['stationcruindex'].astype(int)
            
        # GRIDCELL: set non-numeric station gridcell to -9999
        
        print('replacing missing stationgridcell with -9999 ... ')
        
        a = np.array( df.stationgridcell ); b = np.arange(len(a))
        mask_inverse = np.array( [ a[i].isdigit() for i in range(len(a)) ] )
        mask = np.invert( mask_inverse )
        idx = b[mask]
        df['stationgridcell'].loc[idx] = '-9999'
        df['stationgridcell'] = df['stationgridcell'].astype(int)

    #------------------------------------------------------------------------------    
    # CONVERT: float variables from list of str to int (needed for NaN replacement of fill Values)   
    #------------------------------------------------------------------------------

    for j in range(1,13): df[ str(j) ] = [ int( monthlist[i][j-1].strip() ) for i in range(len(monthlist)) ]             
    df['stationlat'] = df['stationlat'].astype(int)
    df['stationlon'] = df['stationlon'].astype(int)
    df['stationelevation'] = df['stationelevation'].astype(int)    
    df['stationfirstyear'] = df['stationfirstyear'].astype(int)
    df['stationlastyear'] = df['stationlastyear'].astype(int)
    df['stationsource'] = df['stationsource'].astype(int)
    df['stationfirstreliable'] = df['stationfirstreliable'].astype(int)

    if include_cru == True:
        
        df['stationcruindex'] = df['stationcruindex'].astype(int)
        df['stationgridcell'] = df['stationgridcell'].astype(int) 
        
    #------------------------------------------------------------------------------
    # REPLACE: float variable fill values with NaN:
    #------------------------------------------------------------------------------
           
    print('replacing fill values with np.nan ...')
           
    for j in range(1,13): df[ str(j) ].replace(-999, np.nan, inplace=True)
    df['stationlat'].replace(-999, np.nan, inplace=True) 
    df['stationlon'].replace(-9999, np.nan, inplace=True) 
    df['stationelevation'].replace(-9999, np.nan, inplace=True) 
    df['stationfirstyear'].replace(-9999, np.nan, inplace=True)
    df['stationlastyear'].replace(-9999, np.nan, inplace=True)
    df['stationsource'].replace(-99, np.nan, inplace=True)
    df['stationfirstreliable'].replace(-9999, np.nan, inplace=True)

    if include_cru == True:
        
        df['stationcruindex'].replace(-9999, np.nan, inplace=True)
        df['stationgridcell'].replace(-9999, np.nan, inplace=True)
        
    #------------------------------------------------------------------------------
    # APPLY: scale factors
    #------------------------------------------------------------------------------

    print('applying coordinate scaling ...')

    for j in range(1,13): df[ str(j) ] = df[ str(j) ] / 10.0
    df['stationlat'] = df['stationlat'] / 10.0
    df['stationlon'] = df['stationlon'] / 10.0

    #------------------------------------------------------------------------------
    # CONVERT: longitudes from +W to +E
    #------------------------------------------------------------------------------

    print('replacing +W with +E longitudes ...')
    
    df['stationlon'] = -df['stationlon']

    #------------------------------------------------------------------------------
    # FILTER: out unphysical temperatures (current min=, max=)
    #------------------------------------------------------------------------------
	
    if fix_unphysical == True:

        print('replacing unphysical temperatures with np.nan ...')
        
        for j in range(1,13):

            mask = ( df[ str(j) ] < global_monthly_min ) | (df[ str(j) ] > global_monthly_max )
            df[ str(j) ][mask] = np.nan
            
        # ( see quickstats.py for extraction of monthly minmax
        
    #------------------------------------------------------------------------------
    # DROP: stations without elevation
    #------------------------------------------------------------------------------

    if require_elevation == True: 
		
        print('dropping stations without elevation ...')

        df = df.dropna(subset=['stationelevation'])        

    #------------------------------------------------------------------------------
    # DROP: stations without geolocation
    #------------------------------------------------------------------------------

    if require_geolocation == True: 
		
        print('dropping stations without geolocation ...')

        df = df.dropna(subset=['stationlat', 'stationlon'])        
                        
    return df

#==============================================================================
# LOAD: archive
#==============================================================================

print('loading archive ...')

if load_df == True:

    if use_pickle == True: df = pd.read_pickle( filename_pkl, compression='bz2' )    
    else: df = pd.read_csv( filename_csv, index_col=0 )

else: 
    
    df = load_dataframe(filename_txt)

    #------------------------------------------------------------------------------
    # CONVERT: dtypes for efficient storage
    #------------------------------------------------------------------------------

    df['year'] = df['year'].astype('int16')
    for j in range(1,13): df[ str(j) ] = df[ str(j) ].astype('float32')        
    df['stationlat'] = df['stationlat'].astype('float32')
    df['stationlon'] = df['stationlon'].astype('float32')
    df['stationelevation'] = df['stationelevation'].astype('float32')
    df['stationfirstyear'] = df['stationfirstyear'].astype('int16')
    df['stationlastyear'] = df['stationlastyear'].astype('int16')
    df['stationsource'] = df['stationsource'].astype('int8')
    df['stationfirstreliable'] = df['stationfirstreliable'].astype('int16')

    if include_cru == True:
    
        df['stationcruindex'] = df['stationcruindex'].astype('int16')
        df['stationgridcell'] = df['stationgridcell'].astype('int16')
        
#==============================================================================
# SAVE: dataframe
#==============================================================================

print('saving absolute temperatures ...')

df_temp = df.copy()
if use_pickle == True:
    df_temp.to_pickle( filename_pkl, compression='bz2' )
else:
    df_temp.to_csv( filename_csv )

#------------------------------------------------------------------------------
print('** END')


