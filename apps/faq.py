#------------------------------------------------------------------------------
# PROGRAM: faq.py
#------------------------------------------------------------------------------
# Version 0.1
# 22 August, 2021
# Michael Taylor
# https://patternizer.github.io
# michael DOT a DOT taylor AT uea DOT ac DOT uk
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

layout = html.Div([
    dbc.Container([

        html.H5(children='FAQ #1: First Reliable Years', className="text-left"),
        html.Br(),
        html.P(['In the app, there is a radio button for switching between the raw station monthly mean temperatures and data trimmed to the first reliable year (FRY) where expert opinion judges them to be reliably homogenised by the relevant national meteorological agency. In the majority of cases this coincides with the start of the series of processed observations. The landing page of this app displays the global distribution of stations available in the current version of the GloSAT.p0x data archive (many thanks to ', html.A('Emily Wallis', style={'color':'cyan'}), ' for kindly preparing this plot).']),                                        
        html.Br(),

        html.H5(children='FAQ #2: Climate Stripes', className="text-left"),
        html.Br(),
        html.P(['These are calculated in accordance with the prescription designed by Professor Ed Hawkins at the University of Reading at ', html.A('showyourstripes.info', href='https://showyourstripes.info'), ' but using land surface air temperatures being assimilated during the GloSAT project. In addition, the average temperature in 1961-1990 is set as the boundary between blue and red colours, and the colour scale varies from +/- 2.6 standard deviations of the annual average temperatures between 1901-2000. The stripes are shown for the available data where considered robust. The yearly anomaly timeseries is overlaid in black (many thanks to ', html.A('Zeke Hausfather', style={'color':'cyan'}), ' for kindly providing this code snippet and design idea).']),                                        
        html.Br(),

        html.H5(children='FAQ #3: Anomaly Timeseries', className="text-left"),
        html.P(['For each station, the series of monthly mean temperature anomalies is shown together with the resampled yearly average. The monthly values are quality controlled observations provided by national meteorological agencies and, depending on the station, are typically calculated from sub-series such as daily means calculated from hourly or daily minmax component timeseries. The anomalies are calculated relative to the mean for each month in the baseline period 1961-1990 assuming a minimum of 15 out of a possible 30 values. In the app we colour-code the yearly data according to the climate stripes colormap.']),        
        html.Br(),

        html.H5(children='FAQ #4: Seasonal Series', className="text-left"),
        html.P(['From the monthly anomaly timeseries at each station, we calculate the yearly seasonal average (monthly triplets: DJF, MAM, JJA and SON). These are then smoothed at the decadal timescale using a purpose-built discrete Fourier transform filter. More details on this filter are available at its repo on ', html.A('Github', href='https://github.com/patternizer/glosat-dft-filter'), '.']),        
        html.Br(),

        html.H5(children='FAQ #5: Year Ranks', className="text-left"),
        html.P(['The yearly averaged anomalies are also ranked in descending order and plotted in the form of a rank-frequency plot to allow analysis of when hottest or coldest years occurred. The rank plot is color-coded according to the climate stripes colormap. For each year an indicative spread is calculated from the standard deviation of the monthly anaomalies during that year. The rank-frequency plot follows a classical Zipf law.']),        
        html.Br(),

        html.H5(children='FAQ #6: Climatology', className="text-left"),
        html.P(['In order to place recent warming in context, we overlay the available monthly values of the last year in a station timeseries on top of the median monthly values and the climatology of all preceding years bounded by min and max values. An annual cycle is typically observable.']),        
        html.Br(),

        dbc.Row([
            dbc.Card(
                children=[
                    html.A(
                    dbc.CardBody(
                        dbc.Button("Station Viewer App", href="/glosat", color="primary"), 
                        className="text-center",
                    )),
                ], 
            body=False, color="dark", outline=True), 
        ], justify="center"),

        html.Br(),

        dbc.Row([
            dbc.Col(
                dbc.Card(
                    children=[
                        html.H5(children='GloSAT Dataset', className="text-center"),
                        dbc.Row([
                            dbc.Col(
                                dbc.Button("Station Data", href="http://crudata.uea.ac.uk/cru/data/temperature/crutem4/station-data.htm", color="primary"),
                                className="mt-2", align="center"),
                            dbc.Col(dbc.Button("Temperatures", href="http://crudata.uea.ac.uk/cru/data/temperature/", color="primary"),
                                className="mt-2", align="center"),
                        ], justify="center")
                    ],
                body=True, color="dark", outline=True),
            width=4, className="mb-4"),

            dbc.Col(
                dbc.Card(
                    children=[
                        html.H5(children='Project Website', className="text-center"),
                        dbc.Button("GloSAT", href="https://www.glosat.org", color="primary", className="mt-2"),
                    ],
                body=True, color="dark", outline=True),
            width=4, className="mb-2"),

            dbc.Col(
                dbc.Card(
                    children=[
                        html.H5(children='Plotly Code', className="text-center"),
                        dbc.Button("GitHub", href="https://github.com/patternizer/glosat-py", color="primary", className="mt-2"),
                    ],
                body=True, color="dark", outline=True),
            width=4, className="mb-2"),
            
        ], className="mb-2"),

        html.P(['GloSAT Station Viewer is brought to you by the ', html.A('Climatic Research Unit', href='http://www.cru.uea.ac.uk'), ' in the School of Environmental Sciences, University of East Anglia']),

	])
])

