from dash import html, dcc, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc
import mlflow
import pandas as pd

import plotly.express as px

# Define Layout
layout = html.Div(
    [
        html.H2("Experiments", className="mb-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H4(
                                            "Recent Runs", className="glass-card-header"
                                        ),
                                        dbc.Button(
                                            [
                                                html.I(
                                                    className="fas fa-sync-alt me-2"
                                                ),
                                                "Refresh",
                                            ],
                                            id="refresh-btn",
                                            color="primary",
                                            className="mb-3",
                                        ),
                                        dash_table.DataTable(
                                            id={
                                                "type": "experiments-data-table",
                                                "index": "main",
                                            },
                                            columns=[
                                                {"name": col, "id": col}
                                                for col in [
                                                    "run_id",
                                                    "experiment_name",
                                                    "status",
                                                    "start_time",
                                                ]
                                            ],
                                            data=[],
                                            style_table={"overflowX": "auto"},
                                            style_cell={
                                                "backgroundColor": "#1e293b",
                                                "color": "#f8fafc",
                                                "border": "1px solid rgba(255,255,255,0.1)",
                                                "textAlign": "left",
                                                "fontFamily": "Inter, sans-serif",
                                                "padding": "10px",
                                            },
                                            style_header={
                                                "backgroundColor": "#0f172a",
                                                "color": "#f8fafc",
                                                "fontWeight": "bold",
                                                "borderBottom": "1px solid rgba(255,255,255,0.2)",
                                                "textTransform": "uppercase",
                                            },
                                            style_data_conditional=[
                                                {
                                                    "if": {"state": "active"},
                                                    "backgroundColor": "rgba(59, 130, 246, 0.2)",
                                                    "border": "1px solid #3b82f6",
                                                },
                                                {
                                                    "if": {"state": "selected"},
                                                    "backgroundColor": "rgba(59, 130, 246, 0.2)",
                                                    "border": "1px solid #3b82f6",
                                                },
                                            ],
                                            page_size=10,
                                        ),
                                    ],
                                    className="glass-card",
                                )
                            ]
                        )
                    ],
                    lg=7,
                    md=12,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H4(
                                    "Live Training Metrics",
                                    className="glass-card-header",
                                ),
                                dcc.Graph(
                                    id="live-plot",
                                    config={
                                        "displayModeBar": False,
                                        "scrollZoom": False,
                                    },
                                    style={"height": "400px"},
                                ),
                                dcc.Interval(
                                    id="interval-component",
                                    interval=2000,
                                    n_intervals=0,
                                ),
                            ],
                            className="glass-card",
                        )
                    ],
                    lg=5,
                    md=12,
                ),
            ]
        ),
    ]
)


@callback(
    [
        Output({"type": "experiments-data-table", "index": "main"}, "data"),
        Output({"type": "experiments-data-table", "index": "main"}, "columns"),
        Output("live-plot", "figure"),
    ],
    [
        Input("refresh-btn", "n_clicks"),
        Input("interval-component", "n_intervals"),
        Input({"type": "experiments-data-table", "index": "main"}, "active_cell"),
    ],
)
def update_metrics(n_clicks, n_intervals, active_cell):
    # Default outputs
    data = []
    columns = [
        {"name": col, "id": col}
        for col in ["run_id", "experiment_name", "status", "start_time"]
    ]

    # Empty figure
    fig = px.line()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )

    try:
        experiments = mlflow.search_experiments()
        all_runs = []
        for exp in experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            if not runs.empty:
                runs["experiment_name"] = exp.name
                all_runs.append(runs)

        if all_runs:
            df = pd.concat(all_runs)
            # Select columns
            base_cols = ["run_id", "experiment_name", "status", "start_time"]
            metric_cols = [c for c in df.columns if c.startswith("metrics.")]

            # Prepare dataframe for display (limit to 10 for performance, or more)
            display_df = df[base_cols + metric_cols].sort_values(
                "start_time", ascending=False
            )

            data = display_df.to_dict("records")
            columns = [{"name": i, "id": i} for i in base_cols + metric_cols]

            # Determine which run to plot
            # Default to top of the list (latest) if nothing selected
            target_run_id = display_df.iloc[0]["run_id"]

            # If user clicked a cell, use that row
            # Since we disabled frontend sorting, row index maps 1:1 to display_df
            if active_cell:
                row_idx = active_cell["row"]
                # Ensure index is within bounds (in case data changed)
                if row_idx < len(display_df):
                    target_run_id = display_df.iloc[row_idx]["run_id"]

            # Update Plot
            client = mlflow.tracking.MlflowClient()
            metrics_to_try = [
                "loss",
                "policy_loss",
                "mean_q",
                "reward",
                "entropy",
                "value_loss",
            ]
            history = None
            metric_name = "loss"

            for m in metrics_to_try:
                history = client.get_metric_history(target_run_id, m)
                if history:
                    metric_name = m
                    break

            if history:
                steps = [m.step for m in history]
                values = [m.value for m in history]

                # Retrieve run info for title
                run_rows = display_df[display_df["run_id"] == target_run_id]
                run_name = target_run_id
                if not run_rows.empty:
                    run_name = (
                        f"{run_rows.iloc[0]['experiment_name']} ({target_run_id})"
                    )

                fig = px.line(
                    x=steps,
                    y=values,
                    template="plotly_dark",
                )
                fig.update_layout(
                    title={
                        "text": f"Metric: {metric_name} | {run_name}",
                        "font": {"size": 12},
                    },
                    xaxis_title="Epoch",
                    yaxis_title=metric_name.replace("_", " ").title(),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#f8fafc"),
                    margin=dict(l=40, r=20, t=40, b=40),
                    autosize=True,
                )
            else:
                fig.update_layout(
                    title=f"No metrics for {target_run_id}",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#f8fafc"),
                )

    except Exception:
        # On error, just return empty/safes
        pass

    return data, columns, fig
