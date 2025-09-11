# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import networkx as nx
from tracer_backend_utils import ConstantTensor, InputOp
from tracer_backend import OperationGraph
import os
from pytorch_graph_utils import format_file_with_black
from typing import Dict, List, Tuple
from find_repeated_subgraphs import CompositeOperation
import gzip
import torch
import json


class PytorchGraph:
    def __init__(self, operation_graph: OperationGraph):
        self.graph = operation_graph
        assert (
            not operation_graph.tracer.fake_original_tensor
        ), f"Make sure LazyParams(fake=False) when tracing the model."

    def get_imports_and_code_lines(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        imports: Dict[str, List[str]] = {}
        code_lines: Dict[str, List[str]] = {}
        for node_id in list(nx.topological_sort(self.graph.graph)):
            operation = self.graph.graph.nodes[node_id].get("operation")
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


class CompositePytorchGraph(PytorchGraph):
    """
    A specialized PytorchGraph that supports composite operations.
    It overrides the `find_repeated_subgraphs` method to use the custom implementation.
    """

    def __init__(self, graph: OperationGraph, clustered_graph=False):
        super().__init__(graph)
        self.clustered_graph = clustered_graph

    def get_imports_and_code_lines(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        self.orig_config = ConstantTensor.ConstantTensorFromModel
        ConstantTensor.ConstantTensorFromModel = True
        imports: Dict[str, List[str]] = {}
        code_lines: Dict[str, List[str]] = {}
        main_op = CompositeOperation(
            id="main",
            unique_name="OUTPUT",
            sub_operations=list(
                self.graph.graph.nodes[node_id]["operation"] for node_id in nx.topological_sort(self.graph.graph)
            ),
            function_call_name="composite",
            args=[],
            kwargs={},
        )
        imports["main"] = [
            "import gzip",
            "import torch",
            "import json",
            "from tracer_backend import trace_torch_model",
            "from find_repeated_subgraphs import PatternObjFactory as POFactory, find_repeated_subgraphs",
            "from generate_pytorch_graph import CompositePytorchGraph",
            "from utils import LazyParams",
        ] + main_op.generate_import_code()
        if not self.clustered_graph:
            const_meta = {
                k: {
                    "shape": list(v.value.shape),
                    "dtype": str(v.value.dtype),
                    "min_max": [v.value.min().item(), v.value.max().item()],
                }
                for k, v in CompositeOperation.ALL_CONSTANTS.items()
            }
            # dump const meta
            with open("const_meta.json", "w") as f:
                json.dump(const_meta, f, indent=2)
        code_lines["main"] = ""
        main_op_code = main_op.generate_code()
        input_ops = [
            self.graph.graph.nodes[node_id]["operation"]
            for node_id in self.graph.graph.nodes
            if isinstance(self.graph.graph.nodes[node_id]["operation"], InputOp)
        ]
        for const in CompositeOperation.ALL_CONSTANTS:
            main_op_code = main_op_code.replace(f"{const},", f"params['{const}'],")
            main_op_code = main_op_code.replace(f"{const})", f"params['{const}'])")
            main_op_code = main_op_code.replace(f"({const})", f"(params['{const}'])")
            main_op_code = main_op_code.replace(f'"{const}",', f"params['{const}'],")
            main_op_code = main_op_code.replace(f'"{const}")', f"params['{const}'])")
            main_op_code = main_op_code.replace(f'("{const}")', f"(params['{const}'])")
        code_lines[
            "main"
        ] += f"""
class CustomModel(torch.nn.Module):
    def __init__(self, params):
        self.params = params
        super().__init__()

    def state_dict(self):
        return self.params.to_dict()

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.params.from_dict(state_dict)

    def forward(self,{','.join([inp.input_identifier for inp in input_ops])}):
        params = self.params
        {main_op_code}
        return OUTPUT
        """
        code_lines[
            "main"
        ] += """\n# Set fake=False for real tensors
fake = True
params = LazyParams(
    meta_path="const_meta.json",
    data_path="graph.pth",
    fake=fake
)\n"""
        input_shapes = [tuple(input_op.args[0]) for input_op in input_ops]
        input_args = [f"torch.ones({shape})" for shape in input_shapes]

        code_lines["main"] += f"\ntorch_model = CustomModel(params)"
        if self.clustered_graph:
            code_lines["main"] += f"\n_tensor_io_log.clear()"
            code_lines["main"] += f"\ntorch_model({','.join(input_args)})"
            code_lines["main"] += (
                "\nwith open('tensor_io_log.json', 'w') as f:\n" + "    json.dump(_tensor_io_log, f, indent=2)"
            )
        else:
            code_lines[
                "main"
            ] += f"\noperation_graph = trace_torch_model(torch_model, {input_shapes}, dump_visualization=True, save_original_tensors=not fake)"
            code_lines["main"] += f"\nclustered_graph, composite_ops = find_repeated_subgraphs(operation_graph)"
            code_lines[
                "main"
            ] += """\nif not fake:
    pytorch_graph = CompositePytorchGraph(clustered_graph, clustered_graph=True)
    pytorch_graph.dump_to_python_file('clustered_graph.py', True)
"""
            tensor_dict = {k: v.value.detach() for k, v in CompositeOperation.ALL_CONSTANTS.items()}
            with open("graph.pth", "wb") as f:
                torch.save(tensor_dict, f)
        ConstantTensor.ConstantTensorFromModel = self.orig_config
        return imports, code_lines
