#------------------------------------------------------------------------------
# PROGRAM: app.py (multipage)
#------------------------------------------------------------------------------
# Version 0.2
# 18 June, 2022
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import dash
import dash_bootstrap_components as dbc

external_stylesheets=[dbc.themes.DARKLY]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True

