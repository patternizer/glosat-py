#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: app.py
#------------------------------------------------------------------------------
# Version 0.4
# 29 July, 2020
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
import numpy as np
import numpy.ma as ma
from mod import Mod
import pandas as pd
# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.cm as cm
import matplotlib.colors as c
from matplotlib import colors as mcol
from matplotlib.cm import ScalarMappable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
matplotlib.rcParams['text.usetex'] = True
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
# EXTRACT TARBALL IF df.csv IS COMPRESSED:
#------------------------------------------------------------------------------

filename = Path("df.csv")
if not filename.is_file():
    print('Uncompressing df.csv from tarball ...')
    #tar -xzvf df.tar.gz
    #tar -xjvf df.tar.bz2
    #filename = "df.tar.gz"
    #subprocess.Popen(['tar', '-xzvf', filename])
    filename = "df.tar.bz2"
    subprocess.Popen(['tar', '-xjvf', filename])
    time.sleep(5) # pause 5 seconds to give tar extract time to complete prior to attempting pandas read_csv

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------
fontsize = 12

df = pd.read_csv('df.csv', index_col=0)
stationlon = df['stationlon']
stationlat = df['stationlat']
stationcode = df['stationcode'].unique()

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
    html.H1(children='CRUTEM5-py',            
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
                    value = 10080,
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
                html.Label(['Dataset: ', html.A('CRUTEM5.1 prelim01 ', href='https://catalogue.ceda.ac.uk/uuid/eeabb5e1ff2140f48e76ea1ffda6bb48')]),
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

    code = df[df['stationcode']==stationcode[value]]['stationcode'].iloc[0]
    lat = df[df['stationcode']==stationcode[value]]['stationlat'].iloc[0]
    lon = df[df['stationcode']==stationcode[value]]['stationlon'].iloc[0]
    station = df[df['stationcode']==stationcode[value]]['stationname'].iloc[0]
    country = df[df['stationcode']==stationcode[value]]['stationcountry'].iloc[0]
                                  
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
    layout = go.Layout(  height=140, width=380, margin=dict(r=10, l=0, b=0, t=0))

    return {'data': data, 'layout':layout} 

@app.callback(
    Output(component_id='plot-timeseries', component_property='figure'),
    [Input(component_id='station', component_property='value')],    
    )
    
def update_plot_timeseries(value):
    
    """
    Plot station timeseries
    """

    da = df[df['stationcode']==stationcode[value]].iloc[:,range(0,13)]
    ts = []    
    for i in range(len(da)):            
        monthly = da.iloc[i,1:]
        ts = ts + monthly.to_list()    
    ts = np.array(ts)                
    ts_yearly = []    
    ts_yearly_sd = []    
    for i in range(len(da)):            
        yearly = np.nanmean(da.iloc[i,1:13])
        yearly_sd = np.nanstd(da.iloc[i,1:13])
        ts_yearly.append(yearly)    
        ts_yearly_sd.append(yearly_sd)    
    ts_yearly_sd = np.array(ts_yearly_sd)                    
    t = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts), freq='M')     
    t_yearly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='A')   

    Y = df[df['stationcode']==stationcode[value]].iloc[:,range(1,13)].T
    n = len(Y.T)            
    colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
    hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]
                                  
    data=[
        go.Scatter(x=t, y=ts, 
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
#            line=dict(width=2.0, color=hexcolors),
            marker=dict(size=10, opacity=1.0, color=hexcolors),
            name='Yearly',
            yaxis='y1',
        )
    ]   
    
    fig = go.Figure(data)
    fig.update_layout(
        xaxis_title = {'text': 'Year'},
        yaxis_title = {'text': 'Temperature anomaly, °C'},
#        title = {'text': 'Seasonal cycle', 'x':0.5, 'y':0.925, 'xanchor': 'center', 'yanchor': 'top'}
    )
    fig.update_layout(height=300, width=700, margin={"r":0,"t":0,"l":10,"b":0})    

    return fig

@app.callback(
    Output(component_id='plot-climatology', component_property='figure'),
    [Input(component_id='station', component_property='value')],    
    )

def update_plot_climatology(value):
    
    """
    Plot station climatology
    """

    X = df[df['stationcode']==stationcode[value]].iloc[:,0]
    Y = df[df['stationcode']==stationcode[value]].iloc[:,range(1,13)].T

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
        yaxis_title = {'text': 'Monthly temperature anomaly, °C'},
#        title = {'text': 'Seasonal cycle', 'x':0.5, 'y':0.925, 'xanchor': 'center', 'yanchor': 'top'}
    )
    fig.update_layout(height=300, width=600, margin={"r":0,"t":0,"l":10,"b":0})
    
    return fig

@app.callback(
    Output(component_id='plot-worldmap', component_property='figure'),
    [Input(component_id='station', component_property='value')])
    
def update_plot_worldmap(value):
    
    """
    Plot station location on world map
    """

    Y = df[df['stationcode']==stationcode[value]].iloc[:,range(1,13)].T
    n = len(Y.T)            
    
    colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
    hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]
    cmap = hexcolors

    lat = [df[df['stationcode']==stationcode[value]]['stationlat'].iloc[0]]
    lon = [df[df['stationcode']==stationcode[value]]['stationlon'].iloc[0]]
    var = []
    station = df[df['stationcode']==stationcode[value]]['stationcode'].iloc[0]
    data = df[df['stationcode']==stationcode[value]].iloc[0]
    
#    fig = go.Figure(go.Densitymapbox(lat=lat, lon=lon, z=var, radius=10))
    fig = go.Figure(px.scatter_mapbox(lat=lat, lon=lon, color_discrete_sequence=["darkred"], zoom=1))
    fig.update_layout(mapbox_style="carto-positron", mapbox_center_lat=lat[0], mapbox_center_lon=lon[0]) 
#    fig.update_layout(mapbox_style="stamen-watercolor", mapbox_center_lat=lat[0], mapbox_center_lon=lon[0]) 
#    fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lat=lat[0], mapbox_center_lon=lon[0]) 
#    fig.update_layout(title={'text': 'Location', 'x':0.5, 'y':0.925, 'xanchor': 'center', 'yanchor': 'top'})
    fig.update_layout(height=200, width=600, margin={"r":80,"t":0,"l":40,"b":0})
    
    return fig

##################################################################################################
# Run the dash app
##################################################################################################

if __name__ == "__main__":
    app.run_server(debug=True)
    
