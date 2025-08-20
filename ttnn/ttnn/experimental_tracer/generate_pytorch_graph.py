# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import networkx as nx
from tracer_backend import OperationGraph
import os
from pytorch_graph_utils import format_file_with_black
from typing import Dict, List, Tuple


class PytorchGraph:
    def __init__(self, operation_graph: OperationGraph):
        self.graph = operation_graph.graph

    def get_imports_and_code_lines(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        imports: Dict[str, List[str]] = {}
        code_lines: Dict[str, List[str]] = {}
        for node_id in list(nx.topological_sort(self.graph)):
            operation = self.graph.nodes[node_id].get("operation")
            if operation:
                imports[node_id] = operation.generate_import_code()
                code_lines[node_id] = operation.generate_code() + " # " + operation.unique_name
        return imports, code_lines

    def generate_pytorch_code(self) -> str:
        """Generate PyTorch code from the graph."""
        imports, code_lines = self.get_imports_and_code_lines()
        total_imports = []
        total_code_lines = []
        for node_id, node_imports in imports.items():
            for imp in node_imports:
                if imp not in total_imports:
                    total_imports.append(imp)
            total_code_lines.append(code_lines[node_id])
        return "\n".join(total_imports), "\n".join(total_code_lines)

    def dump_to_python_file(self, file_path: str, format_code: bool = False):
        """Dump the generated PyTorch code into a Python file."""
        imports, code_lines = self.generate_pytorch_code()
        with open(file_path, "w") as f:
            f.write("# Auto-generated PyTorch code\n")
            for imp in imports:
                f.write(f"{imp}")
            f.write("\n")
            f.write("if __name__ == '__main__':\n")
            for line in code_lines.splitlines():
                f.write(f"    {line}\n")
            f.write(f"    print('RAN MODEL SUCCESSFULLY')\n")
        print(f"Generated torch graph code dumped to {os.path.abspath(file_path)}")

        # Run black on the generated file
        if format_code:
            format_file_with_black(file_path)
