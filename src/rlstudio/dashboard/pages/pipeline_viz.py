from dash import html
import dash_cytoscape as cyto

# Example Pipeline Data (Hardcoded for MVP)
elements = [
    # Nodes
    {
        "data": {"id": "node1", "label": "DataModule (Source)"},
        "position": {"x": 100, "y": 200},
    },
    {
        "data": {"id": "node2", "label": "RLModule (Actor)"},
        "position": {"x": 300, "y": 100},
    },
    {
        "data": {"id": "node3", "label": "RLModule (Critic)"},
        "position": {"x": 300, "y": 300},
    },
    {
        "data": {"id": "node4", "label": "ReplayBuffer"},
        "position": {"x": 500, "y": 200},
    },
    {"data": {"id": "node5", "label": "Trainer"}, "position": {"x": 700, "y": 200}},
    # Edges
    {"data": {"source": "node1", "target": "node4"}},
    {"data": {"source": "node1", "target": "node2"}},
    {"data": {"source": "node1", "target": "node3"}},
    {"data": {"source": "node2", "target": "node4"}},
    {"data": {"source": "node3", "target": "node4"}},
    {"data": {"source": "node4", "target": "node5"}},
]

layout = html.Div(
    [
        html.H2("Pipeline Visualization"),
        html.P("Interactive DAG of the training pipeline."),
        cyto.Cytoscape(
            id="cytoscape-pipeline",
            elements=elements,
            style={"width": "100%", "height": "600px"},
            layout={"name": "preset"},
            stylesheet=[
                {
                    "selector": "node",
                    "style": {
                        "content": "data(label)",
                        "text-valign": "center",
                        "color": "white",
                        "text-outline-width": 2,
                        "text-outline-color": "#007bff",
                        "background-color": "#007bff",
                    },
                },
                {
                    "selector": "edge",
                    "style": {
                        "curve-style": "bezier",
                        "target-arrow-shape": "triangle",
                    },
                },
            ],
        ),
    ]
)
