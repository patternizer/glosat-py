#------------------------------------------------------------------------------
# PROGRAM: about.py
#------------------------------------------------------------------------------
# Version 0.1
# 30 October, 2020
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

plot_fry = 'url(/assets/emily-stations-fry.png)'
plot_stripes = 'url(/assets/plot-stripes.png)'
plot_location = 'url(/assets/plot-location.png)'
plot_timeseries = 'url(/assets/plot-timeseries.png)'
plot_seasonal = 'url(/assets/plot-seasonal.png)'
plot_ranks = 'url(/assets/plot-ranks.png)'
plot_climatology = 'url(/assets/plot-climatology.png)'
plot_spiral = 'url(/assets/plot-spiral.png)'


faq_1 = [
    dbc.CardHeader("FAQ #1"),
    dbc.CardImg(src='assets/emily-stations-fry.png', top=True, style={'height':'10vh'}),
    dbc.CardBody(
        [html.H5("First Reliable Years", className="card-title"),
        html.P(
        "How are they calculated?",
        className="card-text")],
    )]

faq_2 = [
    dbc.CardHeader("FAQ #2"),
    dbc.CardImg(src='assets/plot-stripes.png', top=True, style={'height':'10vh'}),
    dbc.CardBody(
        [html.H5("Climate Stripes", className="card-title"),
        html.P(
        "How are they calculated?",
        className="card-text")],
    )]

faq_3 = [
    dbc.CardHeader("FAQ #3"),
    dbc.CardImg(src='assets/plot-timeseries.png', top=True, style={'height':'10vh'}),
    dbc.CardBody(
        [html.H5("Anomaly Timeseries", className="card-title"),
        html.P(
        "How are they calculated?",
        className="card-text")],
    )]

faq_4 = [
    dbc.CardHeader("FAQ #4"),
    dbc.CardImg(src='assets/plot-seasonal.png', top=True, style={'height':'10vh'}),
    dbc.CardBody(
        [html.H5("Seasonal Series", className="card-title"),
        html.P(
        "How are they calculated?",
        className="card-text")],
    )]

faq_5 = [
    dbc.CardHeader("FAQ #5"),
    dbc.CardImg(src='assets/plot-ranks.png', top=True, style={'height':'10vh'}),
    dbc.CardBody(
        [html.H5("Rank Distribution", className="card-title"),
        html.P(
        "How is it calculated?",
        className="card-text")],
    )]

faq_6 = [
    dbc.CardHeader("FAQ #6"),
    dbc.CardImg(src='assets/plot-climatology.png', top=True, style={'height':'10vh'}),
    dbc.CardBody(
        [html.H5("Monthly Climatology", className="card-title"),
        html.P(
        "How is it calculated?",
        className="card-text")],
    )]

#faq_7 = [
#    dbc.CardHeader("FAQ #7"),
#    dbc.CardImg(src='assets/plot-spiral.png', top=True, style={'height':'10vh'}),
#    dbc.CardBody(
#        [html.H5("Warming Spiral", className="card-title"),
#        html.P(
#        "How is it calculated?",
#        className="card-text")],
#    )]

layout = html.Div([
    dbc.Container([

        dbc.Card(                    
            children=[
                html.P([
                    html.H3(children=dbc.Button("Station Viewer App", href="/glosat", color="primary", className="mt-2"), className="text-center"),                              
                ], className="text-center"),
                html.Label(['Michael Taylor¹, Phil Jones¹, Tim Osborn¹, David Lister¹, Ian Harris¹, Emily Wallis¹'], className="text-center"),
                html.P([html.A('¹Climatic Research Unit', href='http://www.cru.uea.ac.uk'), ', School of Environmental Sciences, University of East Anglia'], className="text-center"),
                html.P(['This research was partly funded by the ', html.A('GloSAT Project', href='https://www.glosat.org/'), '.'], className="text-center"),  
            ], body=True, color="dark", outline=True), 
            
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
                            dbc.Col(
                                dbc.Button("Temperatures", href="http://crudata.uea.ac.uk/cru/data/temperature/", color="primary"),
                               className="mt-2", align="center"),
                        ], justify="center")
                    ],
                body=True, color="dark", outline=True),
            ),
            dbc.Col(
                dbc.Card(
                    children=[
                        html.H5(children='Project Website', className="text-center"),
                        dbc.Button("GloSAT", href="https://www.glosat.org", color="primary", className="mt-2"),
                    ],
                body=True, color="dark", outline=True),
            ),
            dbc.Col(
                dbc.Card(
                    children=[
                        html.H5(children='Plotly Code', className="text-center"),
                        dbc.Button("GitHub", href="https://github.com/patternizer/glosat-py", color="primary", className="mt-2"),
                    ],
                body=True, color="dark", outline=True),
            ),            
        ], className="mb-4"),
        
        html.Div(children=
            [
            dbc.Row(
                [
                dbc.Col(dbc.Card(faq_1, color="secondary", inverse=True)),
                dbc.Col(dbc.Card(faq_2, color="light", inverse=True)),
                dbc.Col(dbc.Card(faq_3, color="danger", inverse=True)),
                dbc.Col(dbc.Card(faq_4, color="info", inverse=True)),
                dbc.Col(dbc.Card(faq_5, color="success", inverse=True)),
                dbc.Col(dbc.Card(faq_6, color="warning", inverse=True)),
#               dbc.Col(dbc.Card(card_content, color="primary")),
#               dbc.Col(dbc.Card(card_content, color="dark", inverse=True)),
                ], className="mb-4"),
            ]
        ),

        dbc.CardDeck(
            [             
            dbc.Card(children=[html.H4(children='License', className="text-center"), 
                html.P(html.Br()),                               
                html.P("© UEA all rights reserved"),
                html.Label(['The results, maps and figures shown on this website are licenced under an ', html.A('Open Government License', href='http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/'), '.']),                                                    
                html.P('This website has been created for the GloSAT Project and does not use privately hosted web-analytics tools, does not store cookies and respects do not track settings.'),
                html.P('For further information please refer to the UEA privacy policy.'),                    
                dbc.Button("UEA Privacy Policy", href="https://www.uea.ac.uk/about/university-information/statutory-legal-policies", color="primary", className="mt-2"),
                ], body=True, color="dark", outline=True),                 
            dbc.Card(children=[html.H4(children='Contact', className="text-center"),
                html.P(html.Br()),                               
                html.P('For any enquiries please contact:'),
                html.A('Climatic Research Unit (CRU)'),
                html.A('School of Environmental Sciences'),
                html.A('Faculty of Science'),
                html.A('University of East Anglia'),
                html.A('Norwich Research Park'),
                html.A('Norwich NR4 7TJ, UK.'),
                html.P(html.Br()),
                html.A('Telephone: +44 (0) 1603 592542'),
                html.A('Email: cru@uea.ac.uk'),
                ], body=True, color="dark", outline=True),                 
            ], className="mb-2"),

    ], className="mb-2"),
])

