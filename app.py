#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: app.py
#------------------------------------------------------------------------------
# Version 0.2
# 22 July, 2020
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
import xarray as xr
# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.cm as cm
import matplotlib.colors as c
from matplotlib import colors as mcol
from matplotlib.cm import ScalarMappable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.collections import PolyCollection
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
#import seaborn as sns
import cmocean
# Mapping libraries:
import cartopy
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cf
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
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
import os
import random
from random import randint
from random import randrange


# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------
fontsize = 12

df = pd.read_csv('df.csv', index_col=0)
stationlon = -df['stationlon']
stationlat = df['stationlat']
stationcode = df['stationcode'].unique()

projection = 'platecarree'

#monthstr = ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
value = '10080'
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
                    value = 0,
                    style = {'padding' : '10px', 'width': '60%', 'display': 'inline-block'},
                ),                                    
            ]), 
            width={'size':8}, 
            ),                        
            dbc.Col(html.Div([
                dcc.RadioItems(
                    id = "colormap",
                    options=[
                        {'label': ' Viridis', 'value': 'Viridis'},
                        {'label': ' Cividis', 'value': 'Cividis'},
                        {'label': ' Plotly3', 'value': 'Plotly3'},
                        {'label': ' Magma', 'value': 'Magma'},
                        {'label': ' Coolwarm', 'value': 'Coolwarm'}                        
                    ],
                    value = 'Coolwarm',
                    labelStyle={'padding' : '5px', 'display': 'inline-block'},
                ),
            ]), 
            width={'size':4}, 
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
            width={'size':4}, 
            ),                        
            dbc.Col(html.Div([
                dcc.Graph(id="plot-climatology", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                 
            ]), 
            width={'size':4}, 
            ),            
            dbc.Col(html.Div([
                dcc.Graph(id="plot-worldmap", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                 
            ]), 
            width={'size':4}, 
            ),            
        ]),
    ]),
# ------------

# ------------
    html.Div([   
        dbc.Row([

            dbc.Col(html.Div([                    

                html.P([
                    html.H3(children='About'),
                    html.Label('A visual tool for inspecting the CRUTEM5.1 dataset'),
                    html.Br(),    
                    html.Label(['Codebase: ', html.A('Github', href='https://github.com/patternizer/glosat-py')]),       
                    html.Br(),                  
                    html.Label(['Created using Plotly Python by ', html.A('Michael Taylor', href='https://patternizer.github.io')]),            
                ],                        
                style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                      
            ]), 
            width={'size':12}, 
            ),                        

        ]),                        
    ]), 
# ------------

])

# ========================================================================
# Callbacks
# ========================================================================

@app.callback(
    Output(component_id='plot-worldmap', component_property='figure'),
    [Input(component_id='station', component_property='value'), 
    Input(component_id='colormap', component_property='value')],    
    )
    
def update_plot_worldmap(value, colors):
    
    """
    Plot station location on world map
    """
    if colors == 'Viridis': cmap = px.colors.sequential.Viridis_r
    elif colors == 'Cividis': cmap = px.colors.sequential.Cividis_r
    elif colors == 'Plotly3': cmap = px.colors.sequential.Plotly3_r
    elif colors == 'Magma': cmap = px.colors.sequential.Magma_r
    elif colors == 'Coolwarm': cmap = px.colors.sequential.RdBu_r
    
    Y = df[df['stationcode']==stationcode[value]].iloc[:,range(1,13)].T
    n = len(Y.T)            
    colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) # array([[0.13774403, 0.17485525, 0.41001322, 1.], [...])
    hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]
    cmap = hexcolors

    lat = [df[df['stationcode']==stationcode[value]]['stationlat'].iloc[0]]
    lon = [-df[df['stationcode']==stationcode[value]]['stationlon'].iloc[0]]
    var = [df[df['stationcode']==stationcode[value]].iloc[:,1:13].mean().mean()]
    station = df[df['stationcode']==stationcode[value]]['stationinfo'].iloc[0]
    
#    fig = go.Figure(go.Densitymapbox(lat=lat, lon=lon, z=var, zmin=-2, zmax=2, radius=10, colorscale=cmap, autocolorscale = False))
    fig = go.Figure(go.Densitymapbox(lat=lat, lon=lon, z=var, radius=10, colorscale=cmap, autocolorscale = False))
    fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=0) 
    fig.update_layout(title={'text': str(station), 'x':0.5, 'y':0.95, 'xanchor': 'left', 'yanchor': 'top'})
    fig.update_layout(margin={"r":0,"t":50,"l":10,"b":0})
    
    return fig

@app.callback(
    Output(component_id='plot-timeseries', component_property='figure'),
    [Input(component_id='station', component_property='value'), 
    Input(component_id='colormap', component_property='value')],    
    )
    
def update_plot_timeseries(value, colors):
    
    """
    Plot station timeseries
    """
    if colors == 'Viridis': cmap = px.colors.sequential.Viridis_r
    elif colors == 'Cividis': cmap = px.colors.sequential.Cividis_r
    elif colors == 'Plotly3': cmap = px.colors.sequential.Plotly3_r
    elif colors == 'Magma': cmap = px.colors.sequential.Magma_r
    elif colors == 'Coolwarm': cmap = px.colors.sequential.RdBu_r

    da = df[df['stationcode']==stationcode[value]].iloc[:,range(0,13)]
    ts = []    
    for i in range(len(da)):            
        monthly = da.iloc[i,1:]
        ts = ts + monthly.to_list()    
    ts = np.array(ts)                    
    t = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts), freq='M')            
    ts_yearly = da.groupby(da['year']).mean().values
    t_yearly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='Y')    
                                  
    fig = go.Figure(
        go.Scatter(x=t, y=ts, 
            mode='lines+markers', 
            line=dict(width=1.0, color='navy'),
            marker=dict(size=5, opacity=0.5),
        )
    )   
    fig.update_layout(
#        xlabel("Year"),
#        ylabel("Monthly temperature anomaly, $\mathrm{\degree}C$"),
        title={'text': 'Monthly temperature anomaly timeseries', 'x':0.5, 'y':0.95, 'xanchor': 'center', 'yanchor': 'top'})
    fig.update_layout(margin={"r":0,"t":50,"l":10,"b":0})

    return fig

@app.callback(
    Output(component_id='plot-climatology', component_property='figure'),
    [Input(component_id='station', component_property='value'), 
    Input(component_id='colormap', component_property='value')],    
    )
def update_plot_climatology(value, colors):
    
    """
    Plot station climatology
    """
    if colors == 'Viridis': cmap = px.colors.sequential.Viridis_r
    elif colors == 'Cividis': cmap = px.colors.sequential.Cividis_r
    elif colors == 'Plotly3': cmap = px.colors.sequential.Plotly3_r
    elif colors == 'Magma': cmap = px.colors.sequential.Magma_r
    elif colors == 'Coolwarm': cmap = px.colors.sequential.RdBu_r

    X = df[df['stationcode']==stationcode[value]].iloc[:,0]
    Y = df[df['stationcode']==stationcode[value]].iloc[:,range(1,13)].T

    n = len(Y.T)            
    colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) # array([[0.13774403, 0.17485525, 0.41001322, 1.], [...])
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
#        xlabel("Month"),
#        ylabel("Monthly temperature anomaly, $\mathrm{\degree}C$")        
        title={'text': 'Monthly temperature anomaly seasonal cycle', 'x':0.5, 'y':0.95, 'xanchor': 'center', 'yanchor': 'top'})
    fig.update_layout(margin={"r":0,"t":50,"l":10,"b":0})
    
    return fig

##################################################################################################
# Run the dash app
##################################################################################################

if __name__ == "__main__":
    app.run_server(debug=True)
    
