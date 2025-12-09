from typing import Callable, List, Union, Any, Dict
import inspect


class Node:
    """
    A Node wraps a pure function with defined inputs and outputs.
    """

    def __init__(
        self,
        func: Callable,
        inputs: Union[str, List[str], Dict[str, str], None] = None,
        outputs: Union[str, List[str], Dict[str, str], None] = None,
        name: str = None,
    ):
        self.func = func
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.name = name or func.__name__

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the node's function with the provided inputs.
        """
        # Simple implementation for MVP: Assumes inputs is a dict of kwargs matching function signature
        # In a full Kedro implementation, this logic is more complex (positional vs keyword matching)

        # Filter inputs to only pass what the function expects (basic safety)
        sig = inspect.signature(self.func)
        func_kwargs = {k: v for k, v in inputs.items() if k in sig.parameters}

        # If inputs was defined as a list map, we might need translation, but keeping it simple for now
        # Assuming inputs mapping happens at Pipeline level or Catalog level for this MVP

        result = self.func(**func_kwargs)

        # Handle outputs
        output_dict = {}
        if isinstance(self.outputs, str):
            output_dict[self.outputs] = result
        elif isinstance(self.outputs, list):
            if len(self.outputs) == 1:
                output_dict[self.outputs[0]] = result
            else:
                # Assume result is a tuple if multiple outputs
                for i, out_name in enumerate(self.outputs):
                    output_dict[out_name] = result[i]

        return output_dict

    def __repr__(self):
        return f"Node({self.name})"
