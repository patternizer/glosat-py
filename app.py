#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: app.py
#------------------------------------------------------------------------------
# Version 0.6
# 19 August, 2020
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
from mod import Mod
import pandas as pd
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
import cmocean
# Plotly libraries
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# App Deployment Libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from flask import Flask
import random
from random import randint
from random import randrange
import os
import os.path
from pathlib import Path
import sys
import subprocess
from subprocess import Popen
import time

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# EXTRACT TARBALL IF df_temp.csv AND df_anom.csv IS COMPRESSED:
#------------------------------------------------------------------------------

#filename = Path("df_temp.csv")
#if not filename.is_file():
#    print('Uncompressing df_temp.csv from tarball ...')
#    filename = "df_temp.tar.bz2"
#    subprocess.Popen(['tar', '-xjvf', filename])
#    time.sleep(5) # pause 5 seconds to give tar extract time to complete prior to attempting pandas read_csv

#filename = Path("df_anom.csv")
#if not filename.is_file():
#    print('Uncompressing df_anom.csv from tarball ...')
#    filename = "df_anom.tar.bz2"
#    subprocess.Popen(['tar', '-xjvf', filename])
#    time.sleep(5) # pause 5 seconds to give tar extract time to complete prior to attempting pandas read_csv

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------
fontsize = 12

#df_temp = pd.read_csv('df_temp.csv', index_col=0)
#df_anom = pd.read_csv('df_anom.csv', index_col=0)
df_temp = pd.read_pickle('df_temp.pkl', compression='bz2')
df_anom = pd.read_pickle('df_anom.pkl', compression='bz2')
time.sleep(5) # pause 5 seconds to extract dataframe
stationlon = df_temp['stationlon']
stationlat = df_temp['stationlat']
stationcode = df_temp['stationcode'].str.zfill(6).unique()

opts = [{'label' : stationcode[i], 'value' : i} for i in range(len(stationcode))]

# ========================================================================
# Start the App
# ========================================================================

server = Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)
app.config.suppress_callback_exceptions = True

app.layout = html.Div(children=[
            
# ------------
    html.H1(children='GloSAT-py',            
            style={'padding' : '10px', 'width': '100%', 'display': 'inline-block'},
    ),
# ------------
            
# ------------
    html.Div([
        dbc.Row([
            # ------------
            dbc.Col(html.Div([                    
                dcc.Dropdown(
                    id = "station",
                    options = opts,           
                    value = 7939, # Death Valley
                    style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'},
                ),                                    
            ]), 
            width={'size':2}, 
            ),         
               
            dbc.Col(

            html.Div([
                dcc.Graph(id="station-info"),
                html.Br(),
#                html.Label(['Dataset: ', html.A('CRUTEM5.1 v=prelim01', href='https://catalogue.ceda.ac.uk/uuid/eeabb5e1ff2140f48e76ea1ffda6bb48'), ' by ', html.A('UEA-CRU, UEA-NCAS, MO-HC', href='https://crudata.uea.ac.uk/cru/data/temperature/')]),            
#                html.Label(['Dataviz: ', html.A('Github', href='https://github.com/patternizer/glosat-py'), ' by ', html.A('patternizer', href='https://patternizer.github.io')]),            
                html.Label(['Dataset: GloSATp01']),
                html.Br(),
                html.Label(['Dataviz: ', html.A('Github', href='https://github.com/patternizer/glosat-py')]),            
            ],
            style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),    

            width={'size':4}, 
            ),

            dbc.Col(html.Div([
                dcc.Graph(id="plot-worldmap", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                 
            ]), 
            width={'size':6}, 
            ),            
  
        ]),
    ]),
# ------------

# ------------
    html.Div([
        dbc.Row([
            # ------------
            dbc.Col(html.Div([                    
                dcc.Graph(id="plot-timeseries", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                      
            ]), 
            width={'size':6}, 
            ),                        
            dbc.Col(html.Div([
                dcc.Graph(id="plot-climatology", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                 
            ]), 
            width={'size':6}, 
            ),            
        ]),
    ]),
# ------------

# ------------
    html.Div([
        dbc.Row([
            # ------------
            dbc.Col(html.Div([                    
                dcc.Graph(id="plot-stripes", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                      
            ]), 
            width={'size':6}, 
            ),                        
            dbc.Col(html.Div([
                dcc.Graph(id="plot-spiral", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                 
            ]), 
            width={'size':6}, 
            ),            
        ]),
    ]),
# ------------

])

# ========================================================================
# Callbacks
# ========================================================================

@app.callback(
    Output(component_id='station-info', component_property='figure'),
    [Input(component_id='station', component_property='value')],    
    )
    
def update_station_info(value):
    
    """
    Display station info
    """

#    code = df_temp[df_temp['stationcode']==stationcode[value]]['stationcode'].iloc[0]
    lat = df_temp[df_temp['stationcode']==stationcode[value]]['stationlat'].iloc[0]
    lon = df_temp[df_temp['stationcode']==stationcode[value]]['stationlon'].iloc[0]
    station = df_temp[df_temp['stationcode']==stationcode[value]]['stationname'].iloc[0]
    country = df_temp[df_temp['stationcode']==stationcode[value]]['stationcountry'].iloc[0]
                                  
    data = [
        go.Table(
            header=dict(values=['Lat','Lon','Station','Country'],
                line_color='darkslategray',
                fill_color='lightgrey',
                align='left'),
            cells=dict(values=[
                    [str(lat)], 
                    [str(lon)],
                    [station], 
                    [country], 
                ],
                line_color='darkslategray',
                fill_color='white',
                align='left')
        ),
    ]
    layout = go.Layout(  height=140, width=400, margin=dict(r=10, l=0, b=0, t=0))

    return {'data': data, 'layout':layout} 

@app.callback(
    Output(component_id='plot-timeseries', component_property='figure'),
    [Input(component_id='station', component_property='value')],    
    )
    
def update_plot_timeseries(value):
    
    """
    Plot station timeseries
    """

    da = df_anom[df_anom['stationcode']==stationcode[value]].iloc[:,range(0,13)]

    ts_monthly = []    
    for i in range(len(da)):            
        monthly = da.iloc[i,1:]
        ts_monthly = ts_monthly + monthly.to_list()    
    ts_monthly = np.array(ts_monthly)                
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')     

    ts_yearly = []    
    ts_yearly_sd = []    
    for i in range(len(da)):            
        if da.iloc[i,1:].isnull().any():
            yearly = np.nan
            yearly_sd = np.nan
        else:
            yearly = np.nanmean(da.iloc[i,1:])
            yearly_sd = np.nanstd(da.iloc[i,1:])
        ts_yearly.append(yearly)    
        ts_yearly_sd.append(yearly_sd)    
    ts_yearly_sd = np.array(ts_yearly_sd)                    
    t_yearly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='A')   

    n = len(t_yearly)
    colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
    hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]
                                  
    data=[
        go.Scatter(x=t_monthly, y=ts_monthly, 
            mode='lines+markers', 
            legendgroup="a",
            line=dict(width=1.0, color='lightgrey'),
            marker=dict(size=5, opacity=0.5, color='grey'),
            name='Monthly',
            yaxis='y1',
            error_y=dict(
                type='constant',
                array=ts_yearly_sd,
                visible=False),
        ),
        go.Scatter(x=t_yearly, y=ts_yearly, 
            mode='markers', 
            legendgroup="a",
            marker=dict(size=10, opacity=1.0, color=hexcolors),
            name='Yearly',
            yaxis='y1',
        )
    ]   
    
    fig = go.Figure(data)
    fig.update_layout(
#        xaxis = dict(range=[t_yearly[np.isfinite(ts_yearly)][0], t_yearly[np.isfinite(ts_yearly)][-1]]),       
        xaxis = dict(range=[t_yearly[0],t_yearly[-1]]),       
        xaxis_title = {'text': 'Year'},
        yaxis_title = {'text': 'Temperature anomaly, °C'},
    )
    fig.update_layout(height=300, width=700, margin={"r":10,"t":10,"l":10,"b":10})    

    return fig

@app.callback(
    Output(component_id='plot-stripes', component_property='figure'),
    [Input(component_id='station', component_property='value')],    
    )

def update_plot_stripes(value):
    
    """
    Plot station stripes
    https://showyourstripes.info/
    """

    da = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]].iloc[:,range(0,13)]
    ts_yearly = []    
    for i in range(len(da)):            
        if da.iloc[i,1:].isnull().any():
            yearly = np.nan
        else:
            yearly = np.nanmean(da.iloc[i,1:])
        ts_yearly.append(yearly)  
    ts_yearly = np.array(ts_yearly)        
    t_yearly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='A')   

    # Calculate 1961-1990 monthly mean and full timeseries yearly s.d.
    # color range +/- 2.6 standard deviations

#    da = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]
#    ts_monthly = np.array(da.iloc[:,1:13]).ravel()             
#    ts_yearly = (da.groupby('year').mean().iloc[:,0:12]).mean(axis=1)    
#    ts_yearly = np.nanmean(np.array(da.groupby('year').mean().iloc[:,0:12]),axis=1)                
#    t_yearly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='A')   
    mu = np.nanmean(np.array(da[(da['year']>=1961) & (da['year']<=1990)].iloc[:,1:13]).ravel())
#   sigma = np.nanstd(ts_yearly[(ts_yearly.index>=1900)&(ts_yearly.index<=2000)])
    sigma = np.nanstd(ts_yearly)    
#    maxcolor = +2.6 * sigma
#    mincolor = -2.6 * sigma
#    stripe = (ts_yearly-mu)*0.0+1.0
    n = len(ts_yearly)            

    #--------------------------------------------------------------------------
    # Mod of Zeke Hausfather's colorscale mapping that caters also for NaN
    #--------------------------------------------------------------------------
    mask = np.isfinite(ts_yearly)
    ts_yearly_min = ts_yearly[mask].min()    
    ts_yearly_max = ts_yearly[mask].max()    
    ts_yearly_ptp = ts_yearly[mask].ptp()
#    ts_yearly_normed = ((ts_yearly - ts_yearly_min) / ts_yearly_ptp) * (len(ts_yearly[mask]) - 1)             
    ts_yearly_normed = ((ts_yearly - ts_yearly_min) / ts_yearly_ptp) * (n-1)             
#    temps_normed = ((temps - temps_min) / temps_ptp) * (len(temps[mask]) - 1)            
#    temps_normed = ((temps - temps.min(0)) / temps.ptp(0)) * (len(temps) - 1)
#    x_lbls = np.arange(n)
#    t_yearly = ts_yearly.index
    ts_yearly_normed = ts_yearly_normed / (n-1)
    ts_ones = np.full(n,1)
    #--------------------------------------------------------------------------

#    my_cmap = plt.cm.RdBu_r #choose colormap to use for bars
#    norm = Normalize(vmin=0, vmax=elements - 1)

#    def colorval(num):
#        return my_cmap(norm(num))

#    fig=plt.figure(figsize=(6,3))
#    plt.axis('off')
#    plt.axis('tight')
#    plt.bar(t_yearly, ts_ones, color = list(map(colorval, ts_yearly_normed)), width=1.0)
#    plt.plot(t_yearly, ts_yearly_normed - 0.002, color='black', linewidth=1)
#    plt.ylim(0,1)
#    fig.subplots_adjust(bottom = 0)
#    fig.subplots_adjust(top = 1)
#    fig.subplots_adjust(right = 1.005)
#    fig.subplots_adjust(left = 0)
#    fig.savefig('1.png', dpi=300)
        
    data=[
        go.Bar(y=ts_ones, x=t_yearly, 
            marker = dict(color = ts_yearly_normed, colorscale='RdBu_r'),                              
            name = 'NaN',
            hoverinfo='none',
        ),            
        go.Scatter(x=t_yearly, y=ts_yearly_normed, 
            mode='lines', 
            line=dict(width=2, color='black'),
            name='Anomaly',
            hoverinfo='none',                                                                               
        ),
    ]   
    
    fig = go.Figure(data)
    fig.update_layout(
        xaxis_title = {'text': 'Year'},
        yaxis_title = {'text': 'Annual anomaly (from 1961-1990), °C'},        
        xaxis = dict(
            range = [t_yearly[0], t_yearly[-1]],
            showgrid = False, # thin lines in the background
            zeroline = False, # thick line at x=0
            visible = True,  # numbers below
        ), 
        yaxis = dict(
            range = [0, 1],
            showgrid = False, # thin lines in the background
            zeroline = False, # thick line at x=0
            visible = True,  # numbers below
        ), 
        showlegend = True,    
    )
    fig.update_yaxes(showticklabels = False) # hide all the xticks        
    fig.update_layout(height=300, width=700, margin={"r":10,"t":10,"l":50,"b":10})    
    
    return fig

@app.callback(
    Output(component_id='plot-climatology', component_property='figure'),
    [Input(component_id='station', component_property='value')],    
    )

def update_plot_climatology(value):
    
    """
    Plot station climatology
    """

    X = df_temp[df_temp['stationcode']==stationcode[value]].iloc[:,0]
    Y = df_temp[df_temp['stationcode']==stationcode[value]].iloc[:,range(1,13)].T

    n = len(Y.T)            
    colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
    hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]

    data = []
    for k in range(len(Y.T)):
        trace=[go.Scatter(              
            x=np.arange(1,13), y=Y.iloc[:,k], 
            mode='lines+markers', 
            line=dict(width=1.5, color=hexcolors[k]),
            marker=dict(size=5, opacity=0.5),
            name=str(X.iloc[k]))]
        data = data + trace

    fig = go.Figure(data)
    fig.update_layout(
        xaxis_title = {'text': 'Month'},
        yaxis_title = {'text': 'Monthly temperature, °C'},
#        title = {'text': 'Seasonal cycle', 'x':0.5, 'y':0.925, 'xanchor': 'center', 'yanchor': 'top'}
    )
    fig.update_layout(height=300, width=600, margin={"r":10,"t":10,"l":10,"b":10})
    
    return fig

@app.callback(
    Output(component_id='plot-spiral', component_property='figure'),
    [Input(component_id='station', component_property='value')],    
    )

def update_plot_spiral(value):
    
    """
    Plot station climate spiral of monthly or yearly mean anomaly from min:
    # http://www.climate-lab-book.ac.uk/spirals/    
    """
    
    da = df_anom[df_anom['stationcode']==df_anom['stationcode'].unique()[value]].iloc[:,range(0,13)]
    baseline = np.nanmean(np.array(da[(da['year']>=1850)&(da['year']<=1900)].groupby('year').mean()).ravel())    
    ts_monthly = np.array(da.iloc[:,1:13]).ravel() - baseline             
    mask = np.isfinite(ts_monthly)
    ts_monthly_min = ts_monthly[mask].min()    
    ts_monthly = ts_monthly - ts_monthly_min    

    ts_yearly = []    
    for i in range(len(da)):            
        if da.iloc[i,1:].isnull().any():
            yearly = np.nan
        else:
            yearly = np.nanmean(da.iloc[i,1:])
        ts_yearly.append(yearly)   
    t_yearly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='A')   
    mask = np.isfinite(ts_yearly)
    ts_yearly_min = np.nanmin(np.array(ts_yearly)[mask])
    ts_yearly = ts_yearly - ts_yearly_min
            
    n = len(da)
    colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
    hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]
    
    data = []
    for k in range(len(da)):
        trace=[go.Scatterpolar(              
#            r = np.array(da[da['year']==da.iloc[k][0].astype('int')].iloc[:,1:13]).ravel() - baseline - ts_monthly_min,            
            r = np.tile(ts_yearly[k],12),      
#            theta = np.linspace(0, 2*np.pi, 12),
            theta = np.linspace(0, 360, 12),
            mode = 'lines', 
            line = dict(width=1, color=hexcolors[k]),
            name = str(da.iloc[k][0].astype('int')),
#            fill = 'toself',
#            fillcolor = hexcolors[k],
        )]
        data = data + trace

    fig = go.Figure(data)
    
    fig.update_layout(
        title = "Yearly anomaly from minimum ("+str(da.iloc[0][0].astype('int'))+"-"+str(da.iloc[-1][0].astype('int'))+")",        
#        template = "plotly_dark",
        template = None,
        showlegend = True,
        polar = dict(
#            radialaxis = dict(range=[0, 15], showticklabels=True, ticks=''),
            radialaxis = dict(range=[0, 3], showticklabels=True, ticks=''),
            angularaxis = dict(showticklabels=False, ticks=''),
        ),
#        annotations=[dict(x=0, y=0, text=str(da.iloc[0][0].astype('int')))],        
    )
    fig.update_layout(height=300, width=600, margin={"r":50,"t":30,"l":50,"b":50})    
    
    return fig
    
@app.callback(
    Output(component_id='plot-worldmap', component_property='figure'),
    [Input(component_id='station', component_property='value')])

def update_plot_worldmap(value):
    
    """
    Plot station location on world map
    """

    Y = df_temp[df_temp['stationcode']==stationcode[value]].iloc[:,range(1,13)].T
    n = len(Y.T)            
    
    colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
    hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]
    cmap = hexcolors

    lat = [df_temp[df_temp['stationcode']==stationcode[value]]['stationlat'].iloc[0]]
    lon = [df_temp[df_temp['stationcode']==stationcode[value]]['stationlon'].iloc[0]]
    var = []
    station = df_temp[df_temp['stationcode']==stationcode[value]]['stationcode'].iloc[0]
    data = df_temp[df_temp['stationcode']==stationcode[value]].iloc[0]
    
#    fig = go.Figure(go.Densitymapbox(lat=lat, lon=lon, z=var, radius=10))
    fig = go.Figure(px.scatter_mapbox(lat=lat, lon=lon, color_discrete_sequence=["darkred"], zoom=1))
    fig.update_layout(mapbox_style="carto-positron", mapbox_center_lat=lat[0], mapbox_center_lon=lon[0]) 
#    fig.update_layout(mapbox_style="stamen-watercolor", mapbox_center_lat=lat[0], mapbox_center_lon=lon[0]) 
#    fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lat=lat[0], mapbox_center_lon=lon[0]) 
#    fig.update_layout(title={'text': 'Location', 'x':0.5, 'y':0.925, 'xanchor': 'center', 'yanchor': 'top'})
    fig.update_layout(height=200, width=600, margin={"r":80,"t":10,"l":40,"b":10})
    
    return fig

##################################################################################################
# Run the dash app
##################################################################################################

if __name__ == "__main__":
    app.run_server(debug=False)
#    app.run(debug=False, use_reloader=False)    
    
