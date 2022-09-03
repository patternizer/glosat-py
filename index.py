#------------------------------------------------------------------------------
# PROGRAM: index.py
#------------------------------------------------------------------------------
# Version 0.1
# 25 October, 2020
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

from dash import Dash, callback, html, dcc, dash_table, Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
from app import server
from app import app

# import all pages in the app
from apps import home, glosat, about, faq

# building the navigation bar
# https://github.com/facultyai/dash-bootstrap-components/blob/master/examples/advanced-component-usage/Navbars.py
dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Home", href="/home"),
        dbc.DropdownMenuItem("About", href="/about"),
        dbc.DropdownMenuItem("FAQ", href="/faq"),
        dbc.DropdownMenuItem("App", href="/glosat"),
    ],
    nav = True,
    in_navbar = True,
    label = "Explore",
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="/assets/logo-glosat-dark.png", height="70px")),
                        dbc.Col(dbc.NavbarBrand("Station Viewer", className="ml-2")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="/home",
            ),

            dbc.NavbarToggler(id="navbar-toggler2"),
            dbc.Collapse(
                dbc.Nav(
                    [dropdown], className="ml-auto", navbar=True
                ),
                id="navbar-collapse2",
                navbar=True,
            ),

        ]
    ),
    color="#1c1f2b",
    dark=True,
    className="mb-4",
)

def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

for i in [3]:
    app.callback(
        Output(f"navbar-collapse{i}", "is_open"),
        [Input(f"navbar-toggler{i}", "n_clicks")],
        [State(f"navbar-collapse{i}", "is_open")],
    )(toggle_navbar_collapse)

# embedding the navigation bar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):

    if pathname == '/glosat':
        return glosat.layout
    elif pathname == '/about':
        return about.layout
    elif pathname == '/faq':
        return faq.layout
    else:
        return home.layout

if __name__ == '__main__':
    app.run_server(debug=True)

