#------------------------------------------------------------------------------
# PROGRAM: app.py
#------------------------------------------------------------------------------
# Version 0.1
# 25 October, 2020
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

import dash
import dash_bootstrap_components as dbc
import dash_html_components as html

external_stylesheets=[dbc.themes.DARKLY]
app = dash.Dash(__name__, title='GloSAT Station Viewer', update_title='Loading...', external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True



