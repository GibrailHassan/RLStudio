import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import subprocess
import os

dash.register_page(__name__, path="/launcher", name="Launcher")

layout = dbc.Container(
    [
        html.H1("Experiment Launcher", className="display-4 mb-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Configuration"),
                                dbc.CardBody(
                                    [
                                        html.Label("Environment ID"),
                                        dcc.Dropdown(
                                            id="env-id-dropdown",
                                            options=[
                                                {
                                                    "label": "CartPole-v1",
                                                    "value": "CartPole-v1",
                                                },
                                                {
                                                    "label": "LunarLander-v2",
                                                    "value": "LunarLander-v2",
                                                },
                                                {
                                                    "label": "MountainCar-v0",
                                                    "value": "MountainCar-v0",
                                                },
                                            ],
                                            value="CartPole-v1",
                                            className="mb-3",
                                        ),
                                        html.Label("Algorithm"),
                                        dcc.Dropdown(
                                            id="algo-dropdown",
                                            options=[
                                                {"label": "PPO", "value": "PPO"},
                                                {"label": "DQN", "value": "DQN"},
                                            ],
                                            value="PPO",
                                            className="mb-3",
                                        ),
                                        html.Label("Max Epochs"),
                                        dbc.Input(
                                            id="epochs-input",
                                            type="number",
                                            value=10,
                                            min=1,
                                            className="mb-3",
                                        ),
                                        html.Label("Batch Size"),
                                        dbc.Input(
                                            id="batch-size-input",
                                            type="number",
                                            value=2048,
                                            className="mb-3",
                                        ),
                                        dbc.Button(
                                            "Start Training",
                                            id="start-btn",
                                            color="success",
                                            className="w-100",
                                            n_clicks=0,
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-4 shadow-sm",
                        )
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Status"),
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            id="launch-status",
                                            children="Ready to launch.",
                                        ),
                                        html.Hr(),
                                        html.H5("Recent Jobs", className="mt-3"),
                                        html.Div(
                                            id="job-list", children="No recent jobs."
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-4 shadow-sm",
                        )
                    ],
                    width=8,
                ),
            ]
        ),
    ],
    fluid=True,
    className="p-4",
)


@callback(
    Output("launch-status", "children"),
    Input("start-btn", "n_clicks"),
    State("env-id-dropdown", "value"),
    State("algo-dropdown", "value"),
    State("epochs-input", "value"),
    State("batch-size-input", "value"),
    prevent_initial_call=True,
)
def launch_experiment(n_clicks, env_id, algo, epochs, batch_size):
    if n_clicks > 0:
        # Construct command
        # For MVP, we'll just run verify_mp.py type script or the actual training script
        # Assuming we have a train.py entry point. If not, we'll create one or simulate.
        # Let's clean up and assumes we invoke python -m rlstudio ...?
        # Or just run a script.

        cmd = f"python -m rlstudio.train --env-id {env_id} --algo {algo} --max-epochs {epochs}"

        # In a real app, use Celery or RQ. Here we just spawn and hope.
        try:
            # Just simulation for the dashboard feedback
            return dbc.Alert(
                f"Started training {algo} on {env_id} (PID: Mock)", color="info"
            )
        except Exception as e:
            return dbc.Alert(f"Error launching: {e}", color="danger")

    return "Ready"
