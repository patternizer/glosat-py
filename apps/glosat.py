#------------------------------------------------------------------------------
# PROGRAM: glosat.py
#------------------------------------------------------------------------------
# Version 0.14 (from standalone app versioning)
# 25 October, 2020
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

# Numerics and dataframe libraries:
import numpy as np
import numpy.ma as ma
from mod import Mod
import scipy
import scipy.stats as stats    
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
import cmocean
# Plotly libraries
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# App Deployment Libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from app import app

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------
fontsize = 12

#df_temp = pd.read_csv('df_temp.csv', index_col=0)
#df_anom = pd.read_csv('df_anom.csv', index_col=0)
df_temp = pd.read_pickle('df_temp.pkl', compression='bz2')
df_anom = pd.read_pickle('df_anom.pkl', compression='bz2')

gb = df_temp.groupby(['stationcode'])['stationname'].unique().reset_index()
stationcodestr = gb['stationcode']
stationnamestr = gb['stationname'].apply(', '.join).str.lower()
stationstr = stationcodestr + ': ' + stationnamestr
opts = [{'label' : stationstr[i], 'value' : i} for i in range(len(stationstr))]

#------------------------------------------------------------------------------
# GloSAT APP LAYOUT
#------------------------------------------------------------------------------

layout = html.Div([
    dbc.Container([
                
        dbc.Row([
            dbc.Col(html.Div([   
                dcc.Dropdown(
                    id = "station",
                    options = opts,           
                    value = 0,
                    style = {"color": "black", 'padding' : '10px', 'width': '100%', 'display': 'inline-block'},
                ),                                    
            ], className="dash-bootstrap"), 
            width=4, 
            ),             
            dbc.Col(html.Div([
                html.Br(),
                dcc.RadioItems(
                    id = "radio-fry",
                    options=[
                        {'label': ' FRY', 'value': 'On'},
                        {'label': ' Raw', 'value': 'Off'},
                    ],
                    value = 'On',
                    labelStyle={'padding' : '5px', 'display': 'inline-block'},
                ),
            ]), 
            width=2, 
            ),      
            dbc.Col(
                dcc.Graph(id="station-info"), 
            width=4, 
            ),                        
        ]),

        dbc.Row([
            dbc.Col(html.Div([                    
                dcc.Graph(id="plot-stripes", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                     
            ]), 
            width={'size':6}, 
            ),                        
            dbc.Col(html.Div([
                dcc.Graph(id="plot-worldmap", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                    
            ]), 
            width={'size':6}, 
            ),            
        ]),

        dbc.Row([
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

        dbc.Row([
            dbc.Col(html.Div([
                dcc.Graph(id="plot-ranks", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                 
            ]), 
            width={'size':6}, 
            ),
            dbc.Col(html.Div([     
                dcc.Graph(id="plot-spiral", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                 
            ]), 
            width={'size':6}, 
            ),
#            dbc.Col(html.Div([     
#                html.Br(),
#                html.Label(['Status: Experimental']),
#                html.Br(),
#                html.Label(['Dataset: GloSATp02']),
#                html.Br(),
#                html.Label(['Dataviz: ', html.A('Github', href='https://github.com/patternizer/glosat-py'), ' (dev)']),                
#            ],
#            style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),    
#            width={'size':6}, 
#            ),
        ]),
            
    ]),
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

    lat = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationlat'].iloc[0]
    lon = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationlon'].iloc[0]
    elevation = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationelevation'].iloc[0]
    station = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationname'].iloc[0].lower()
    country = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationcountry'].iloc[0]
                                  
    data = [
        go.Table(
            header=dict(values=['Latitude [°N]','Longitude [°E]','Elev. AMSL [m]','Station','Country'],
                line_color='darkslategray',
                fill_color='lightgrey',
                font = dict(color='Black'),
                align='left'),
            cells=dict(values=[
                    [str(lat)], 
                    [str(lon)],
                    [str(elevation)],
                    [station], 
                    [country], 
                ],
#               line_color='darkslategray',
#               fill_color='white',
                line_color='slategray',
                fill_color='black',
                font = dict(color='white'),
                align='left')
        ),
    ]
    layout = go.Layout(template = "plotly_dark", height=100, width=600, margin=dict(r=10, l=10, b=10, t=10))

    return {'data': data, 'layout':layout} 

@app.callback(
    Output(component_id='plot-worldmap', component_property='figure'),
    [Input(component_id='station', component_property='value')],                   
    )

def update_plot_worldmap(value):
    
    """
    Plot station location on world map
    """

    da = df_temp[ df_temp['stationcode']==df_temp['stationcode'].unique()[value] ]
    n = len(da)            
    
    colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
    hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]
    cmap = hexcolors

    lat = [df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationlat'].iloc[0]]
    lon = [df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationlon'].iloc[0]]
    station = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationcode'].iloc[0]
    
    fig = go.Figure(
        px.scatter_mapbox(lat=lat, lon=lon, color_discrete_sequence=["darkred"], zoom=1))
    fig.update_layout(
        template = "plotly_dark",
#       template = None,
        xaxis_title = {'text': 'Longitude, °E'},
        yaxis_title = {'text': 'Latitude, °N'},
    )
    fig.update_layout(mapbox_style="carto-positron", mapbox_center_lat=lat[0], mapbox_center_lon=lon[0]) 
#   fig.update_layout(mapbox_style="stamen-watercolor", mapbox_center_lat=lat, mapbox_center_lon=lon) 
#   fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lat=lat[0], mapbox_center_lon=lon 
#   fig.update_layout(title={'text': 'Location', 'x':0.5, 'y':0.925, 'xanchor': 'center', 'yanchor': 'top'})    
    fig.update_layout(height=300, width=600, margin={"r":50,"t":10,"l":50,"b":60})
    
    return fig

@app.callback(
    Output(component_id='plot-stripes', component_property='figure'),
    [Input(component_id='station', component_property='value'),
    Input(component_id='radio-fry', component_property='value')],         
    )

def update_plot_stripes(value,trim):
    
    """
    Plot station stripes
    https://showyourstripes.info/
    """

    # value = np.where(df_temp['stationcode'].unique()=='545110')[0][0] # Beijing
    # value = np.where(df_anom['stationcode'].unique()=='024580')[0][0] # Uppsala-Flygplats

    if trim == 'On':
        fry = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationfirstreliable'].unique()
        da = df_temp[ (df_temp['year']>=fry[0]) & (df_temp['stationcode']==df_temp['stationcode'].unique()[value]) ].iloc[:,range(0,13)]
    elif trim == 'Off':   
        da = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]].iloc[:,range(0,13)]

    ts_yearly = np.mean(np.array(da.groupby('year').mean().iloc[:,0:12]),axis=1) 
#   ts_yearly = np.nanmean(np.array(da.groupby('year').mean().iloc[:,0:12]),axis=1) 
    # Solve Y1677-Y2262 Pandas bug with Xarray:        
    # t_yearly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='A')   
    t_yearly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='A', calendar='noleap')     
    n = len(ts_yearly)            

    #--------------------------------------------------------------------------
    # Mod of Zeke Hausfather's colorscale mapping that caters also for NaN
    #--------------------------------------------------------------------------
    mask = np.isfinite(ts_yearly)
    ts_yearly_min = ts_yearly[mask].min()    
    ts_yearly_max = ts_yearly[mask].max()    
    ts_yearly_ptp = ts_yearly[mask].ptp()
    ts_yearly_normed = ((ts_yearly - ts_yearly_min) / ts_yearly_ptp)             
    ts_ones = np.full(n,1)

    # TEST: df_temp ts_yearly_normed versus df_anom ts_yearly_normed
    
#    da2 = df_anom[df_anom['stationcode']==df_anom['stationcode'].unique()[value]].iloc[:,range(0,13)]
#    ts_yearly2 = np.mean(np.array(da2.groupby('year').mean().iloc[:,0:12]),axis=1)
#    mask2 = np.isfinite(ts_yearly2)
#    ts_yearly_min2 = ts_yearly2[mask].min()    
#    ts_yearly_max2 = ts_yearly2[mask].max()    
#    ts_yearly_ptp2 = ts_yearly2[mask].ptp()
#    ts_yearly_normed2 = ((ts_yearly2 - ts_yearly_min2) / ts_yearly_ptp2)             

#    plt.plot(t_yearly,ts_yearly)
#    plt.plot(t_yearly,ts_yearly2)
#    plt.plot(t_yearly,ts_yearly_normed)
#    plt.scatter(x=np.array(t_yearly),y=ts_yearly_normed2,marker='.', color='orange')

    #--------------------------------------------------------------------------
        
    data=[
        go.Bar(y=ts_ones, x=t_yearly, 
            marker = dict(color = ts_yearly_normed, colorscale='RdBu_r', line_width=0),  
            name = 'NaN',
            showlegend=False,
            hoverinfo='none',
        ),            
        go.Scatter(x=t_yearly, y=ts_yearly_normed, 
            mode='lines', 
            line=dict(width=2, color='black'),
            name='Anomaly',
            showlegend=False,
            hoverinfo='none',                                                                               
        ),
    ]   
    
    fig = go.Figure(data)
    fig.update_layout(
        template = "plotly_dark",
#       template = None,
        xaxis_title = {'text': 'Year'},
        yaxis_title = {'text': 'Annual anomaly (from 1961-1990), °C'},        
        xaxis = dict(
            range = [t_yearly[0], t_yearly[-1]],
            showgrid = False, # thin lines in the background
            zeroline = False, # thick line at x=0
            visible = True,   # numbers below
        ), 
        yaxis = dict(
            range = [0, 1],
            showgrid = False, # thin lines in the background
            zeroline = False, # thick line at x=0
            visible = True,   # numbers below
        ), 
        showlegend = True,    
    )
    fig.update_yaxes(showticklabels = False) # hide all the xticks        
    fig.update_layout(legend=dict(
        orientation='v',
        yanchor="top",
        y=0.3,
        xanchor="left",
        x=0.8),
    )
    fig.update_layout(height=300, width=600, margin={"r":10,"t":10,"l":50,"b":10})    
    
    return fig

@app.callback(
    Output(component_id='plot-timeseries', component_property='figure'),
    [Input(component_id='station', component_property='value'),    
    Input(component_id='radio-fry', component_property='value')],    
    )
    
def update_plot_timeseries(value,trim):
    
    """
    Plot station timeseries
    """

    # value = np.where(df_temp['stationcode'].unique()=='545110')[0][0] # Beijing
    # value = np.where(df_anom['stationcode'].unique()=='024580')[0][0] # Uppsala-Flygplats

    if trim == 'On':
        fry = df_anom[df_anom['stationcode']==df_anom['stationcode'].unique()[value]]['stationfirstreliable'].unique()
        da = df_anom[ (df_anom['year']>=fry[0]) & (df_anom['stationcode']==df_anom['stationcode'].unique()[value]) ].iloc[:,range(0,13)]
    elif trim == 'Off':   
        da = df_anom[df_anom['stationcode']==df_anom['stationcode'].unique()[value]].iloc[:,range(0,13)]

    ts_monthly = []    
    for i in range(len(da)):            
        monthly = da.iloc[i,1:]
        ts_monthly = ts_monthly + monthly.to_list()    
    ts_monthly = np.array(ts_monthly)   

    # Solve Y1677-Y2262 Pandas bug with Xarray:        
    # t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')                  
    t_monthly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M', calendar='noleap')     

    ts_yearly = np.mean(np.array(da.groupby('year').mean().iloc[:,0:12]),axis=1) 
    ts_yearly_sd = np.std(np.array(da.groupby('year').mean().iloc[:,0:12]),axis=1)                    
#   ts_yearly = np.nanmean(np.array(da.groupby('year').mean().iloc[:,0:12]),axis=1) 
#   ts_yearly_sd = np.nanstd(np.array(da.groupby('year').mean().iloc[:,0:12]),axis=1)                    
    # Solve Y1677-Y2262 Pandas bug with Xarray:       
    # t_yearly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='A')   
    t_yearly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='A', calendar='noleap')   

    n = len(ts_yearly)
    colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
    hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]

    # Climate Stripes Colourmap

    mask = np.isfinite(ts_yearly)
    if mask.sum() == 0:
        ts_yearly_normed = np.ones(n)*np.nan

        data=[
            go.Scatter(x=t_monthly, y=[], 
                mode='lines+markers', 
                legendgroup="a",
                line=dict(width=1.0, color='lightgrey'),
                marker=dict(size=5, opacity=0.5, color='grey'),
                name='Monthly',
                yaxis='y1',
                error_y=dict(
                    type='constant',
                    array=[],
                    visible=False),
            ),
            go.Scatter(x=t_yearly, y=[], 
                mode='lines+markers', 
                legendgroup="a",
                line=dict(width=1.0, color='black'),
                marker=dict(size=7, symbol='square', opacity=1.0, color = ts_yearly_normed, colorscale='RdBu_r', line_width=1),                  
                name='Yearly',
                yaxis='y1',
            )
        ]   
        
    else:        
        ts_yearly_min = ts_yearly[mask].min()    
        ts_yearly_max = ts_yearly[mask].max()    
        ts_yearly_ptp = ts_yearly[mask].ptp()
        ts_yearly_normed = ((ts_yearly - ts_yearly_min) / ts_yearly_ptp)             

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
                mode='lines+markers', 
                legendgroup="a",
                line=dict(width=1.0, color='black'),
                marker=dict(size=7, symbol='square', opacity=1.0, color = ts_yearly_normed, colorscale='RdBu_r', line_width=1),                  
                name='Yearly',
                yaxis='y1',
            )
        ]   
                                      
    fig = go.Figure(data)
    fig.update_layout(
        template = "plotly_dark",
#       template = None,
        xaxis = dict(range=[t_yearly[0],t_yearly[-1]]),       
        xaxis_title = {'text': 'Year'},
        yaxis_title = {'text': 'Anomaly (from 1961-1990), °C'},
    )

    if mask.sum() == 0:
        fig.update_layout(
            annotations=[
                dict(
                    x=t_yearly[np.floor(len(t_yearly)/2).astype(int)],
                    y=0,
                    xref="x",
                    yref="y",
                    text="No baseline anomaly",
                    showarrow=False,
                    font=dict(
                        family="Courier New, monospace",
                        size=16,
                        color="#ffffff"
                        ),                    
                )
            ]
        )    
    fig.update_layout(legend=dict(
        orientation='v',
        yanchor="top",
        y=0.3,
        xanchor="left",
        x=0.8),
    )    
    fig.update_layout(height=300, width=600, margin={"r":10,"t":10,"l":70,"b":10})    

    return fig

@app.callback(
    Output(component_id='plot-climatology', component_property='figure'),
    [Input(component_id='station', component_property='value'),
    Input(component_id='radio-fry', component_property='value')],              
    )

def update_plot_climatology(value,trim):
    
    """
    Plot station climatology
    """

    # find drop-down index for Beijing
    # value = np.where(df_temp['stationcode'].unique()=='545110')[0][0]

    if trim == 'On':
        fry = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationfirstreliable'].unique()
        da = df_temp[ (df_temp['year']>=fry[0]) & (df_temp['stationcode']==df_temp['stationcode'].unique()[value]) ].iloc[:,range(0,13)]
    elif trim == 'Off':   
        da = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]].iloc[:,range(0,13)]

    X = da.iloc[:,0]
    Y = da.iloc[:,range(1,13)]

    # Climate Stripes Colourmap

    ts_yearly = np.mean(np.array(da.groupby('year').mean().iloc[:,0:12]),axis=1) 
#   ts_yearly = np.nanmean(np.array(da.groupby('year').mean().iloc[:,0:12]),axis=1) 
    # Solve Y1677-Y2262 Pandas bug with Xarray:       
    # t_yearly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='A')   
    t_yearly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='A', calendar='noleap')   

    # Climate Stripes Colourmap

    mask = np.isfinite(ts_yearly)
    ts_yearly_min = ts_yearly[mask].min()    
    ts_yearly_max = ts_yearly[mask].max()    
    ts_yearly_ptp = ts_yearly[mask].ptp()
#   ts_yearly_normed = ((ts_yearly - ts_yearly_min) / ts_yearly_ptp)       
    ts_yearly_normed = ((ts_yearly[mask] - ts_yearly_min) / ts_yearly_ptp)             
    ts_yearly = ts_yearly[mask]
    t_yearly = da['year'][mask]    
    X = X[mask]
    Y = Y[mask]

    # Add miniscule (1e-6) white noise to fix duplicates in colour mapping

    n = np.isfinite(ts_yearly).sum()
    noise = np.random.normal(0,1,n)/1e6
    ts_yearly_normed_whitened = ts_yearly_normed + noise

    # fig,ax = plt.subplots()
    # plt.plot(t_yearly, ts_yearly_normed)
    # plt.scatter(np.array(t_yearly), ts_yearly_normed_whitened, c=ts_yearly_normed_whitened, cmap='RdBu_r', marker='o')

#   colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
    colors = cmocean.cm.rain(np.linspace(0.05,0.95,n)) 
    hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]
    mapidx = ts_yearly_normed_whitened.argsort()
    hexcolors_mapped = [ hexcolors[mapidx[i]] for i in range(len(mapidx)) ]

    # Calculate gamma distribution fit and extract percentile bounding curves

    df = pd.DataFrame(columns=['min','max','normal','p5','p10','p90','p95'], index=np.arange(1,13))
    for j in range(1,13):

        y = da[str(j)]
    #   z = (y-y.mean())/y.std()
        z = y[np.isfinite(y)]
        disttype = 'gamma'
        dist = getattr(scipy.stats, disttype)
        param = dist.fit(z)
        gamma_median = dist.median(param[0], loc=param[1], scale=param[2])       # q50
        gamma_10_90 = dist.interval(0.8, param[0], loc=param[1], scale=param[2]) # q10 and q90
        gamma_5_95 = dist.interval(0.9, param[0], loc=param[1], scale=param[2])  # q5 and q95	
        gamma = dist.rvs(param[0], loc=param[1], scale=param[2], size=len(z))
        gamma_sorted = np.sort(gamma)
        gamma_bins = np.percentile(gamma_sorted, range(0,101))
        z_min = np.min(z)
        z_max = np.max(z)
        
        df['min'][j] = z_min
        df['max'][j] = z_max
        df['normal'][j] = gamma_median
        df['p5'][j] = gamma_5_95[0]
        df['p95'][j] = gamma_5_95[1]
        df['p10'][j] = gamma_10_90[0]
        df['p90'][j] = gamma_10_90[1]
    
        # p10extremes = z < gamma_bins[10]
        # p90extremes = z > gamma_bins[90]
        # p10extremes_frac = p10extremes.sum()/n
        # p90extremes_frac = p90extremes.sum()/n

    data = []
    trace_9_95 = [        
            go.Scatter(x=np.arange(1,13), y=np.array(df['p95'].astype(float)), 
                       mode='lines', 
                       fill=None,
                       line=dict(width=1.0, color='lightgrey'),
                       name='95% centile',      
                       showlegend=False),                       
            go.Scatter(x=np.arange(1,13), y=np.array(df['p5'].astype(float)), 
                       mode='lines', 
                       fill='tonexty',
                       line=dict(width=1.0, color='lightgrey'),
                       name='5-95% range',      
                       showlegend=True),
            go.Scatter(x=np.arange(1,13), y=np.array(df['p5'].astype(float)), 
                       mode='lines', 
                       fill=None,
                       line=dict(width=1.0, color='lightgrey'),
                       name='5% centile',      
                       showlegend=False)
        ] 
    data = data + trace_9_95
    trace_max=[go.Scatter(                      
        x=np.arange(1,13), y=df['max'], mode='lines', 
        line=dict(width=3, color='pink'),
        name='Highest in record')]
    data = data + trace_max
    trace_min=[go.Scatter(                      
        x=np.arange(1,13), y=df['min'], mode='lines', 
        line=dict(width=3, color='cyan'),
        name='Lowest in record')]
    data = data + trace_min
    
    for k in range(n):
#        if Y.iloc[:,k].isnull().any():
#        if Y.iloc[k,:].isnull().all():
#            yearly = np.nan,
#        else:
            trace=[go.Scatter(                      
                x=np.arange(1,13), y=np.array(Y)[k,:], 
                mode='lines', 
#               mode='lines+markers', 
#               line=dict(width=1.5),
                line=dict(width=1, color=hexcolors[k]),
#               marker=dict(size=3, symbol='square', opacity=1.0, color=hexcolors[k], line_width=1),                  
#               marker=dict(size=3, symbol='square', opacity=0.5, color=hexcolors_mapped[k]),
#               marker=dict(size=3, symbol='square', opacity=0.5, color = ts_yearly_normed[k], colorscale='RdBu_r'),                                
                name=str(np.array(X)[k]),
                showlegend=False,
                )
            ]
            data = data + trace

    trace_median=[go.Scatter(                      
        x=np.arange(1,13), y=df['normal'], mode='lines', 
        line=dict(width=3, color='white'),
        name='1961-1990 normal')]
    data = data + trace_median
    trace_latest=[go.Scatter(                      
        x=np.arange(1,13), y=np.array(Y)[n-1,:], 
        mode='lines+markers', 
        line=dict(width=3, color=hexcolors[n-1]),
        marker=dict(size=7, symbol='square', opacity=1.0, color='orange', line_width=1, line_color='black'),                  
        name=str(np.array(X)[n-1]),
        showlegend=True),
    ]
    data = data + trace_latest

    fig = go.Figure(data)
    fig.update_layout(
        template = "plotly_dark",
#       template = None,
        xaxis_title = {'text': 'Month'},
        yaxis_title = {'text': 'Monthly temperature, °C'},
    )
    fig.update_layout(height=300, width=600, margin={"r":10,"t":10,"l":70,"b":10})
    
    return fig

@app.callback(
    Output(component_id='plot-ranks', component_property='figure'),
    [Input(component_id='station', component_property='value'),
    Input(component_id='radio-fry', component_property='value')],              
    )

def update_plot_ranks(value,trim):
    
    """
    Plot station year rank anomaly distribution
    """

    # value = np.where(df_temp['stationcode'].unique()=='545110')[0][0] # Beijing
    # value = np.where(df_anom['stationcode'].unique()=='024580')[0][0] # Uppsala-Flygplats
    
    if trim == 'On':
        fry = df_anom[df_anom['stationcode']==df_anom['stationcode'].unique()[value]]['stationfirstreliable'].unique()
        da = df_anom[ (df_anom['year']>=fry[0]) & (df_anom['stationcode']==df_anom['stationcode'].unique()[value]) ].iloc[:,range(0,13)]
    elif trim == 'Off':   
        da = df_anom[df_anom['stationcode']==df_anom['stationcode'].unique()[value]].iloc[:,range(0,13)]
#       da = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]].iloc[:,range(0,13)]

    # Climate Stripes Colourmap

    ts_yearly = np.mean(np.array(da.groupby('year').mean().iloc[:,0:12]),axis=1) 
    ts_yearly_sd = np.std(np.array(da.groupby('year').mean().iloc[:,0:12]),axis=1) 
    ts_yearly_Q1 = np.percentile(np.array(da.groupby('year').mean().iloc[:,0:12]),25, axis=1)         
    ts_yearly_Q3 = np.percentile(np.array(da.groupby('year').mean().iloc[:,0:12]),75, axis=1)         
    ts_yearly_iqr = ts_yearly_Q3-ts_yearly_Q1                    
    # Solve Y1677-Y2262 Pandas bug with Xarray:       
    # t_yearly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='A')   
    t_yearly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='A', calendar='noleap')   

    # Climate Stripes Colourmap

#    mask = np.isfinite(ts_yearly)
#    ts_yearly_min = ts_yearly[mask].min()    
#    ts_yearly_max = ts_yearly[mask].max()    
#    ts_yearly_ptp = ts_yearly[mask].ptp()
#    ts_yearly_normed = ((ts_yearly[mask] - ts_yearly_min) / ts_yearly_ptp)             
#    ts_yearly = ts_yearly[mask]
#    ts_yearly_sd = ts_yearly_sd[mask]
#    t_yearly = da['year'][mask]    

    # Add miniscule (1e-6) white noise to fix duplicates in colour mapping

#    n = np.isfinite(ts_yearly).sum()
#    noise = np.random.normal(0,1,n)/1e6
#    ts_yearly_normed_whitened = ts_yearly_normed + noise

#    colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
#    hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]
#    mapidx = ts_yearly_normed_whitened.argsort()
#    hexcolors_mapped = [ hexcolors[mapidx[i]] for i in range(len(mapidx)) ]

    # Calculate rank yearly anomaly distribution and re-order labels
    
#   df = pd.DataFrame({'t_yearly':t_yearly, 'ts_yearly':ts_yearly, 'ts_yearly_sd':ts_yearly_sd})
    df = pd.DataFrame({'t_yearly':t_yearly, 'ts_yearly':ts_yearly, 'ts_yearly_sd':ts_yearly_sd, 'ts_yearly_Q1':ts_yearly_Q1, 'ts_yearly_Q3':ts_yearly_Q3, 'ts_yearly_iqr':ts_yearly_iqr})
    df_sorted = df.sort_values('ts_yearly',ascending=False)
    dates_ranked = [ str(df_sorted['t_yearly'][df_sorted.index[i]]) for i in range(len(df_sorted)) ]
    x = np.arange(len(t_yearly))     
    y = df_sorted['ts_yearly']
    e = df_sorted['ts_yearly_sd']
    e_Q1 = df_sorted['ts_yearly_Q1']
    e_Q3 = df_sorted['ts_yearly_Q3']
    e_iqr = df_sorted['ts_yearly_iqr']

    # Climate Stripes Colourmap

    mask = np.isfinite(y)
    if mask.sum() == 0: # --> no baseline --> no amonalies
        ts_yearly_normed = np.ones(len(y))*np.nan
        data=[
            go.Bar(y=[], x=dates_ranked, base=[],
                   marker = dict(color = ts_yearly_normed, colorscale='RdBu_r', line_width=0),  
                   name = 'Yearly SD',  
            ),
            go.Scatter(                      
                x=dates_ranked, y=[], 
                mode='lines+markers', 
                line=dict(width=1, color='black'),
                marker=dict(size=2, symbol='square', opacity=1.0, color=ts_yearly_normed, colorscale='RdBu_r', line_width=1, line_color='black'),                  
                name='Yearly mean',
            )
        ]
    else:
        ts_yearly_min = np.array(y[mask]).min()    
        ts_yearly_max = np.array(y[mask]).max()    
        ts_yearly_ptp = np.array(y[mask]).ptp()
        ts_yearly_normed = ((y[mask] - ts_yearly_min) / ts_yearly_ptp)             
    
        data=[
            go.Bar(y=2*e, x=dates_ranked, base=y-e,
    #       go.Bar(y=e_iqr, x=dates_ranked, base=e_Q1,
                   marker = dict(color = ts_yearly_normed, colorscale='RdBu_r', line_width=0),  
                   name = 'Yearly SD',  
            ),
            go.Scatter(                      
                x=dates_ranked, y=y, 
                mode='lines+markers', 
                line=dict(width=1, color='black'),
                marker=dict(size=2, symbol='square', opacity=1.0, color=ts_yearly_normed, colorscale='RdBu_r', line_width=1, line_color='black'),                  
                name='Yearly mean',
            )
        ]
                                      
    fig = go.Figure(data)
    fig.update_layout(
        template = "plotly_dark",
#       template = None,
        xaxis=dict(title='Rank', type='category'),         
        yaxis_title = {'text': 'Anomaly (from 1961-1990), °C'},
    )
    fig.update_xaxes(showticklabels = False) # hide all the xticks        
    fig.update_layout(legend=dict(
        orientation='v',
        yanchor="top",
        y=0.3,
        xanchor="left",
        x=0.1),
    )
    if mask.sum() == 0:
        fig.update_layout(
            annotations=[
                dict(
                    x=t_yearly[np.floor(len(t_yearly)/2).astype(int)],
                    y=0,
                    xref="x",
                    yref="y",
                    text="No baseline anomaly",
                    showarrow=False,
                    font=dict(
                        family="Courier New, monospace",
                        size=16,
                        color="#ffffff"
                        ),                    
                )
            ]
        )        
    fig.update_layout(height=300, width=600, margin={"r":10,"t":10,"l":70,"b":10})    

    return fig

@app.callback(
    Output(component_id='plot-spiral', component_property='figure'),
    [Input(component_id='station', component_property='value'),    
    Input(component_id='radio-fry', component_property='value')],              
    )

def update_plot_spiral(value,trim):
    
    """
    Plot station climate spiral of monthly or yearly mean anomaly from min:
    # http://www.climate-lab-book.ac.uk/spirals/    
    """
    
    if trim == 'On':
        fry = df_anom[df_anom['stationcode']==df_anom['stationcode'].unique()[value]]['stationfirstreliable'].unique()
        da = df_anom[ (df_anom['year']>=fry[0]) & (df_anom['stationcode']==df_anom['stationcode'].unique()[value]) ].iloc[:,range(0,13)]
    elif trim == 'Off':   
        da = df_anom[df_anom['stationcode']==df_anom['stationcode'].unique()[value]].iloc[:,range(0,13)]

    ts_yearly = np.mean(np.array(da.groupby('year').mean().iloc[:,0:12]),axis=1) 

    # Solve Y1677-Y2262 Pandas bug with Xarray:       
    # t_yearly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='A')   
    t_yearly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_yearly), freq='A', calendar='noleap')

    n = np.isfinite(ts_yearly).sum()
    mask = np.isfinite(ts_yearly)
    ts_yearly_min = ts_yearly[mask].min()    
    ts_yearly_max = ts_yearly[mask].max()    
    ts_yearly_ptp = ts_yearly[mask].ptp()
    ts_yearly_normed = ((ts_yearly[mask] - ts_yearly_min) / ts_yearly_ptp)             
    ts_yearly = ts_yearly[mask] - ts_yearly_min    
    t_yearly = t_yearly[mask]
                
    # Find year of yearly minimum
    df = pd.DataFrame({'t_yearly':t_yearly, 'ts_yearly':ts_yearly})
    df['mindiff']=df['ts_yearly']-df['ts_yearly'].min()        
    minidx = np.where(df['ts_yearly']==df['mindiff'].min())[0]
    minyear = df['t_yearly'][minidx].values[0]
    
    df['noise'] = np.random.normal(0,1,n)/1e6
    df['ts_yearly_whitened'] = df['ts_yearly']+df['noise']

    df_sorted = df.sort_values('ts_yearly_whitened',ascending=True)
    dates_ranked = [ str(df_sorted['t_yearly'][df_sorted.index[i]]) for i in range(len(df_sorted)) ]

    X = dates_ranked
    Y = df_sorted['ts_yearly']
    
    colors = cmocean.cm.balance(np.linspace(0.05,0.95,n)) 
    hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]

    
    data = []
#   for k in range(n-1):
    for k in range(n):

        trace=[go.Scatterpolar(              
            r = np.tile(Y[k],12),      
#           r = np.tile(ts_yearly[k],12),      
            # theta = np.linspace(0, 2*np.pi, 12),
            theta = np.linspace(0, 360, 12),
            mode = 'lines', 
            line = dict(width=1, color=hexcolors[k]),
            name = X[k],
#           name = str(t_yearly[k]),
#           fill = 'toself',
#           fillcolor = hexcolors[k],
            showlegend=False,
            ),
        ]
        data = data + trace

    fig = go.Figure(data)
    
    fig.update_layout(
#       title = "Monthly anomaly from minimum ("+str(da.iloc[0][0].astype('int'))+"-"+str(da.iloc[-1][0].astype('int'))+")",        
#       title = {'text': 'Seasonal cycle', 'x':0.5, 'y':0.925, 'xanchor': 'center', 'yanchor': 'top'}
        title = {'text': " Difference from yearly minimum: "+str(minyear.year), 'x':0.5, 'y':0.925, 'xanchor':'center', 'yanchor': 'top'},
        template = "plotly_dark",
#       showlegend = True,
        polar = dict(
#           radialaxis = dict(range=[0, 15], showticklabels=True, ticks=''),
#           radialaxis = dict(range=[0, 3], showticklabels=True, ticks=''),
            angularaxis = dict(showticklabels=False, ticks=''),
        ),
#       annotations=[dict(x=0, y=0, text=str(da.iloc[0][0].astype('int')))],        
    )
    fig.update_layout(height=300, width=600, margin={"r":80,"t":50,"l":70,"b":60})
    
    return fig

##################################################################################################
# Run the dash app
##################################################################################################

#if __name__ == "__main__":
#    app.run_server(debug=False)
    
