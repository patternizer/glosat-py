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

layout = html.Div([
    dbc.Container([
        dbc.Row([

            dbc.Col(
                dbc.Card(
                    
                    children=[html.H3(children='The App', className="text-center"),                        
                    html.P("Developed by:"),
                    html.Label(['Phil Jones¹, Tim Osborn¹, David Lister¹, Ian Harris¹, Emily Wallis¹, Michael Taylor¹']),
                    html.P('¹Climatic Research Unit, School of Environmental Sciences, University of East Anglia'),
                    html.P('This research was partly funded by the GloSAT Project.'),
                    html.H4(children='License', className="text-left"),
                    html.Label(['The results, maps and figures shown on this website are licenced under an ', html.A('Open Government License', href='http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/'), '.']),                            
                ], body=True, color="dark", outline=True), width=4, className="mb-4"),
             
            dbc.Col(
                dbc.Card(children=[html.H3(children='Privacy Policy', className="text-center"),                                   
                    html.P("© UEA All rights reserved"),
                    html.P('This website has been created for the GloSAT Project and does not use privately hosted web-analytics tools, does not store cookies and respects do not track settings.'),
                    html.P('For further information please refer to the UEA privacy policy.'),                    
                    dbc.Button("Policy", href="https://www.uea.ac.uk/about/university-information/statutory-legal-policies", color="primary", className="mt-2"),
                ], body=True, color="dark", outline=True), width=4, className="mb-2"),

            dbc.Col(
                dbc.Card(children=[html.H3(children='Contact', className="text-center"),
                    html.P('For any enquiries please contact:'),
                    html.A('Climatic Research Unit (CRU)'),
                    html.A('School of Environmental Sciences'),
                    html.A('Faculty of Science'),
                    html.A('University of East Anglia'),
                    html.A('Norwich Research Park'),
                    html.A('Norwich NR4 7TJ, UK.'),
                    html.P(html.Br()),
                    html.A('Telephone: +44 (0) 1603 592542'),
                    html.A('Fax: +44 (0) 1603 591327'),    
                    dbc.Button("Email", href="cru@uea.ac.uk", color="primary", className="mt-2"),
                ], body=True, color="dark", outline=True), width=4, className="mb-2"),

        ], className="mb-2"),
    ])
])

