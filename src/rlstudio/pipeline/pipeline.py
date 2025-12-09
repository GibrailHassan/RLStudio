from typing import List, Dict, Any
from .node import Node
from rlstudio.io import DataCatalog


class Pipeline:
    """
    A Pipeline is a sequence of Nodes.
    """

    def __init__(self, nodes: List[Node]):
        self.nodes = nodes

    def run(self, catalog: DataCatalog = None) -> Dict[str, Any]:
        """
        Executes the pipeline sequentially.
        """
        # In a real framework, this would build a DAG and execute topologically.
        # For MVP, we assume the list order is the execution order.

        # Use a local memory store for intermediate results if catalog is missing
        # If catalog is present, we should leverage it?
        # For now, let's keep a memory dict context.

        context = {}

        # Pre-load initial data from catalog if possible?
        # A full implementation would inspect node inputs and fetch from catalog.

        for node in self.nodes:
            # Prepare inputs for the node
            node_inputs = {}

            # Resolve inputs from:
            # 1. Current execution context (outputs of previous nodes)
            # 2. DataCatalog (if configured and input exists)

            # Normalize inputs to list of strings
            input_names = node.inputs
            if isinstance(input_names, str):
                input_names = [input_names]
            elif isinstance(input_names, dict):
                input_names = list(input_names.values())

            for name in input_names:
                if name in context:
                    node_inputs[name] = context[name]
                elif catalog and catalog.exists(name):
                    node_inputs[name] = catalog.load(name)
                else:
                    # Missing input logic (could error or pass None)
                    pass

            # Run node
            outputs = node.run(node_inputs)

            # Update context
            context.update(outputs)

            # Save outputs to catalog if they are defined in catalog?
            # Basic implementation: just keep in memory for pipeline flow
            if catalog:
                for out_name, out_value in outputs.items():
                    # Check if catalog has an entry for this dataset to save it?
                    # Kedro catalog.save usually requires the dataset to be defined.
                    try:
                        catalog.save(out_name, out_value)
                    except Exception:
                        # Dataset might not be defined in catalog.yml, mostly intermediate
                        pass

        return context

    def __add__(self, other: "Pipeline") -> "Pipeline":
        return Pipeline(self.nodes + other.nodes)
