#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: load-stations.py
#------------------------------------------------------------------------------
# Version 0.1
# 23 July, 2020
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

fontsize = 20
fillval = -999
lat_start = -90;  lat_end = 90
lon_start = -180; lon_end = 180
station_start=0;  station_end=10

load_df = False

#------------------------------------------------------------------------------
# METHODS
#------------------------------------------------------------------------------

def load_dataframe(filename_txt):
    
    #------------------------------------------------------------------------------
    # I/O: stat4.CRUTEM5.1prelim01.1721-2019.txt (text dump from CDF4)
    #------------------------------------------------------------------------------

    # load .txt file (comma separated) into pandas dataframe

    # filename_txt = 'stat4.CRUTEM5.1prelim01.1721-2019.txt'
    # station header sample:    
    # [067250 464  -78 1538 BLATTEN, LOETSCHENTA SWITZERLAND   20012012  982001    9999]
    # station data sample:
    # [1921  -44  -71  -68  -46  -12   17   42   53   27  -20  -21  -40]

    yearlist = []
    monthlist = []
    stationinfo = []
    stationcode = []
    stationcountry = []
    stationlat = []
    stationlon = []

    with open (filename_txt, 'r', encoding="ISO-8859-1") as f:  
                    
        for line in f:   

            if len(line)>1: # ignore empty lines  
       
                if (len(line.strip().split())!=13) | (len(line.split()[0])>4):   
                    # when line is stationinfo extract stationid and stationname
                    info = line.strip().split()
                    country = info[-4]
                    if len(info[0]) == 6:
                        code = info[0]
                        if info[1][1:].isdigit():
                            lat = int(info[1])                            
                            if info[2][1:].isdigit():
                                lon = int(info[2])
                            else:
                                split = info[2].split('-')
                                if split[0].isdigit():
                                    lon = int(split[0])
                                else:
                                    lon = int('-' + split[1])                            
                        else:
                            split = info[1].split('-')  
                            if len(split) == 2:
                                lat = int(split[0])
                                lon = int('-' + split[1])
                            elif len(split) > 2:
                                if split[0].isdigit():                                    
                                    lat = int(split[0])                                 
                                    lon = int('-' + split[1])                                                                  
                                else:
                                    lat = int('-' + split[1])                                 
                                    lon = int('-' + split[2])  
                               
                    else:

                        code = info[0][0:6]   
                        split = info[0].split('-')
                        if len(split) == 2:
                            lat = int(split[1])
                            if info[1].isdigit():
                                lon = int(info[1])                                
                            else:
                                split = info[1].split('-')                                
                                if split[0].isdigit():
                                    lon = int(split[0])
                                else:
                                    lon = int('-' + split[1])                                                            
                        elif len(split) == 3:
                            lat = int('-' + split[1])                            
                            lon = int('-' + split[2])                            
                        elif len(split) == 4:
                            lat = int('-' + split[1])                            
                            lon = int('-' + split[2])                                                        
                        else:
                            print(line)
                            lat = np.nan
                            lon = np.nan
                            
                    if lat == -999: lat = np.nan
                    if lon == -9999: lon = np.nan
                            
                else:           
                    yearlist.append(int(line.strip().split()[0]))                                 
                    monthlist.append(np.array(line.strip().split()[1:]).astype('int'))                                 
                    stationinfo.append(info) # store for flagging
                    stationcode.append(code)
                    stationcountry.append(country)
                    stationlat.append(lat)
                    stationlon.append(lon)
            else:
                continue
    f.close

    # construct dataframe
    df = pd.DataFrame(columns=['year','1','2','3','4','5','6','7','8','9','10','11','12'])
    df['year'] = yearlist

    for j in range(1,12+1):

        df[df.columns[j]] = [ monthlist[i][j-1] for i in range(len(monthlist)) ]

    df.replace(fillval, np.nan, inplace=True) 
    df['stationinfo'] = stationinfo
    df['stationcode'] = stationcode
    df['stationcountry'] = stationcountry
    df['stationlat'] = stationlat
    df['stationlon'] = stationlon
                
    df.to_csv('df.csv')

    return df

#------------------------------------------------------------------------------
# LOAD DATAFRAME
#------------------------------------------------------------------------------

if load_df == True:

    df = pd.read_csv('df.csv', index_col=0)

else:    

    filename_txt = 'stat4.CRUTEM5.1prelim01.1721-2019.txt'
    df = load_dataframe(filename_txt)

#------------------------------------------------------------------------------
# APPLY SCALE FACTORS
#------------------------------------------------------------------------------

df = pd.read_csv('df.csv', index_col=0)
df['stationlat'] = df['stationlat']/10.0
df['stationlon'] = df['stationlon']/10.0
for j in range(1,12+1):
    df[df.columns[j]] = df[df.columns[j]]/100.0

#------------------------------------------------------------------------------
# SAVE SCALED DATAFRAME
#------------------------------------------------------------------------------

df.to_csv('df.csv')

#------------------------------------------------------------------------------
print('** END')
