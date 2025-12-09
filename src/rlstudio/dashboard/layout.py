from dash import html, dcc
import dash_bootstrap_components as dbc
from .app import app

# Navbar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Experiments", href="/experiments")),
        dbc.NavItem(dbc.NavLink("Pipeline Viz", href="/pipeline-viz")),
    ],
    brand="RLStudio",
    brand_href="/",
    color="dark",
    dark=True,
)

# Main Container
layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        navbar,
        dbc.Container(id="page-content", className="pt-4"),
    ]
)
