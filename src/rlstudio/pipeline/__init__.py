from .node import Node
from .pipeline import Pipeline


def node(func, inputs=None, outputs=None, name=None):
    return Node(func, inputs, outputs, name)


def pipeline(nodes):
    return Pipeline(nodes)
