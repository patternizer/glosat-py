#------------------------------------------------------------------------------
# PROGRAM: app.py (multipage)
#------------------------------------------------------------------------------
# Version 0.1
# 25 October, 2020
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

import flask
import dash
from dash import dcc
from dash import html
#import dash_core_components as dcc
#import dash_html_components as html
import dash_bootstrap_components as dbc

external_stylesheets=[dbc.themes.DARKLY]
#app = dash.Dash(__name__, title='GloSAT Station Viewer', update_title='Loading...', external_stylesheets=external_stylesheets)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
#app = dash.Dash(__name__, server=server)
#server = flask.Flask(__name__)
app.config.suppress_callback_exceptions = True



