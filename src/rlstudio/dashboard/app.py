import dash
import dash_bootstrap_components as dbc

# Initialize Dash App
# Suppress callback exceptions for multi-page app architecture
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

app.title = "RLStudio Dashboard"

server = app.server
