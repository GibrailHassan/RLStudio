from dash import Input, Output, html
from .app import app
from .layout import layout
from .pages import experiments, pipeline_viz

app.layout = layout


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/experiments":
        return experiments.layout
    elif pathname == "/pipeline-viz":
        return pipeline_viz.layout
    elif pathname == "/launcher":
        return launcher.layout
    else:
        return html.H1("Welcome to RLStudio", className="display-3")


def main():
    app.run(debug=True)


if __name__ == "__main__":
    main()
