from dash import html, dcc
import dash_bootstrap_components as dbc
from .app import app

# Sidebar
sidebar = html.Div(
    [
        html.H2("RLStudio", className="sidebar-header"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.I(className="fas fa-home me-2"), "Home"],
                    href="/",
                    active="exact",
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-flask me-2"), "Experiments"],
                    href="/experiments",
                    active="exact",
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-rocket me-2"), "Launcher"],
                    href="/launcher",
                    active="exact",
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-project-diagram me-2"), "Pipeline Viz"],
                    href="/pipeline-viz",
                    active="exact",
                ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar",
)

# Main Container
layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        sidebar,
        html.Div(id="page-content", className="content"),
    ]
)
