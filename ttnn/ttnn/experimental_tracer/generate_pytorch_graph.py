import networkx as nx
from tracer_backend import OperationGraph
import os
from pytorch_graph_utils import format_file_with_black

class PytorchGraph:
    def __init__(self, operation_graph: OperationGraph):
        self.graph = operation_graph.graph

    def generate_pytorch_code(self) -> str:
        """Generate PyTorch code from the graph."""
        code_lines = ["import torch", ""]
        for node_id in list(nx.topological_sort(self.graph))[::-1]:
            operation = self.graph.nodes[node_id].get("operation")
            if operation:
                code_lines.append(operation.generate_code())
        return "\n".join(code_lines)

 

    def dump_to_python_file(self, file_path: str, format_code: bool = False):
        """Dump the generated PyTorch code into a Python file."""
        pytorch_code = self.generate_pytorch_code()
        with open(file_path, "w") as f:
            f.write("# Auto-generated PyTorch code\n")
            f.write("if __name__ == '__main__':\n")
            for line in pytorch_code.splitlines():
                f.write(f"    {line}\n")
            f.write(f"    print('RAN MODEL SUCCESSFULLY')\n")
        print(f"Generated torch graph code dumped to {os.path.abspath(file_path)}")

        # Run black on the generated file
        if format_code:
            format_file_with_black(file_path)
  