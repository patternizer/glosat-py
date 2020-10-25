#------------------------------------------------------------------------------
# PROGRAM: home.py
#------------------------------------------------------------------------------
# Version 0.1
# 25 October, 2020
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

image = 'url(/assets/emily-stations-fry.png)'
layout = html.Div([

    dbc.Container([

        dbc.Row([
            dbc.Card(
                children=[
                    dbc.CardImg(src='assets/emily-stations-fry.png', top=False, style={'height':'50vh'}),
                    dbc.CardBody(
                        dbc.Button("App", 
                            href="/glosat", 
                            color="primary"), className="text-center",
                    ),
                ], 
            body=False, color="dark", outline=True), 
        ], justify="center"),
        html.Br(),

        dbc.Row([
            dbc.Col(dbc.Card(children=[html.H3(children='GloSATp02 Dataset',
                                               className="text-center"),
                                       dbc.Row([dbc.Col(dbc.Button("GloSATp02", href="http://crudata.uea.ac.uk/cru/data/temperature/crutem4/station-data.htm",
                                                                   color="primary"),
                                                        className="mt-2"),
                                                dbc.Col(dbc.Button("Descriptor", href="http://crudata.uea.ac.uk/cru/data/temperature/",
                                                                   color="primary"),
                                                        className="mt-2")], justify="center")
                                       ],
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-4"),

            dbc.Col(dbc.Card(children=[html.H3(children='GloSAT project',
                                               className="text-center"),
                                       dbc.Button("GloSAT",
                                                  href="https://www.glosat.org",
                                                  color="primary",
                                                  className="mt-2"),

                                       ],
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-2"),

            dbc.Col(dbc.Card(children=[html.H3(children='Plotly Code',
                                               className="text-center"),
                                       dbc.Button("GitHub",
                                                  href="https://github.com/patternizer/glosat-py",
                                                  color="primary",
                                                  className="mt-2"),
                                       ],
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-2"),

        ], className="mb-2"),

        dbc.Row([
        html.A("GloSAT Station Viewer is brought to you by the Climatic Research Unit at the University of East Anglia",
               href="http://www.cru.uea.ac.uk/")
        ], className="mb-8"),

    ])
])

