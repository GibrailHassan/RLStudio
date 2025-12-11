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
                        html.Div(
                            [
                                html.H4("Configuration", className="glass-card-header"),
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
                                    className="mb-3 text-dark",
                                ),
                                html.Label("Algorithm"),
                                dcc.Dropdown(
                                    id="algo-dropdown",
                                    options=[
                                        {"label": "PPO", "value": "PPO"},
                                        {"label": "DQN", "value": "DQN"},
                                    ],
                                    value="PPO",
                                    className="mb-3 text-dark",
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
                                    [
                                        html.I(className="fas fa-play me-2"),
                                        "Start Training",
                                    ],
                                    id="start-btn",
                                    color="primary",
                                    className="w-100",
                                    n_clicks=0,
                                ),
                            ],
                            className="glass-card",
                        )
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H4("Status", className="glass-card-header"),
                                html.Div(
                                    id="launch-status",
                                    children=dbc.Alert(
                                        "Ready to launch.", color="info"
                                    ),
                                ),
                                html.Hr(
                                    style={
                                        "border-top": "1px solid rgba(255,255,255,0.1)"
                                    }
                                ),
                                html.H5("Recent Jobs", className="mt-3"),
                                html.Div(id="job-list", children="No recent jobs."),
                            ],
                            className="glass-card",
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
        # Use python executable from current environment
        python_exe = (
            "python"  # subprocess usually takes 'python' from PATH, which is venv here.
        )

        # Ensure we are in root
        cwd = os.getcwd()
        script = "run_experiment.py"

        # Command: python run_experiment.py --algo PPO --env CartPole-v1 --epochs 10
        cmd = f"{python_exe} {script} --algo {algo} --env {env_id} --epochs {epochs}"

        try:
            # Spawn process non-blocking
            # We use subprocess.Popen
            process = subprocess.Popen(cmd, shell=True, cwd=cwd)

            return dbc.Alert(
                f"Started training {algo} on {env_id} (PID: {process.pid}). Go to 'Experiments' tab to see progress.",
                color="success",
            )
        except Exception as e:
            return dbc.Alert(f"Error launching: {e}", color="danger")

    return "Ready"
