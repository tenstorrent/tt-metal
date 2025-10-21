# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch

import torch
from torch.utils._pytree import tree_map

from typing import Iterator
import contextlib
from os.path import commonprefix
from itertools import dropwhile
import json
from typing import Optional, List, Dict, Any
import os
import networkx as nx
from tracer_backend_utils import (
    get_operation_class,
    Operation,
    OperationMetadata,
    PlaceholderTensor,
    ConstantTensor,
    Parameter,
    TupleOp,
    InputOp,
)
from collections import OrderedDict, deque, defaultdict


# TODO: move this into library proper
@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
    guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        del guard


class TracerData:
    id: int = 0

    def __init__(self, save_original_tensors=False, track_params=True):
        self.graph_output_to_input: Dict[str, List[TrackableTensorArgument]] = {}
        self.graph_output_to_node: Dict[str, Dict[str, Any]] = {}
        self.post_run_output_to_input: Dict[str, List[TrackableTensorArgument]] = {}
        self.post_run_output_to_node: Dict[str, Dict[str, Any]] = {}
        self.constants: Dict[str, ConstantTensor] = {}
        self.save_original_tensors = save_original_tensors
        ConstantTensor.ConstantTensorFromModel = save_original_tensors
        self.track_params = track_params

    def get_next_id(self):
        current_id = TracerData.id
        TracerData.id += 1
        return current_id

    def topological_sort_reorder(self):
        """
        Reorders the graph dictionaries topologically.

        Args:
            graph_output_to_node (dict): Mapping of node IDs to their attributes.
            outputs_to_inputs (dict): Mapping of node IDs to node IDs of their inputs.

        Returns:
            tuple: (reordered_graph_output_to_node, reordered_outputs_to_inputs)
                Dictionaries reordered topologically.  Returns None if a cycle is detected.
        """
        graph = nx.DiGraph()
        # Add nodes and edges to the graph
        for node_id, inputs in self.graph_output_to_input.items():
            graph.add_node(node_id)
            for input_node_id in inputs:
                graph.add_edge(input_node_id.tensor.id, node_id)

        try:
            # Perform topological sort
            topological_order = list(nx.topological_sort(graph))

            # Reorder the dictionaries based on topological order
            self.graph_output_to_node = {
                node_id: self.graph_output_to_node[node_id]
                for node_id in topological_order
                if node_id in self.graph_output_to_node
            }
            self.graph_output_to_input = {
                node_id: self.graph_output_to_input[node_id]
                for node_id in topological_order
                if node_id in self.graph_output_to_input
            }
        except nx.NetworkXUnfeasible:
            print("The graph has a cycle, so topological sort is not possible.")
            return None, None

    def finalize(self):
        """
        Finalize the tracer data by removing None values and fixing in-place operations.
        """
        self.graph_output_to_input.update(self.post_run_output_to_input)
        self.graph_output_to_node.update(self.post_run_output_to_node)
        self.topological_sort_reorder()

    def get_node_statistics(self):
        """
        Get statistics about the nodes in the graph.

        Returns:
            dict: A dictionary containing statistics about the nodes.
        """

        types_of_nodes = {}
        for node in self.graph_output_to_node.values():
            op_type = node["op_type"]
            if op_type not in types_of_nodes:
                types_of_nodes[op_type] = 0
            types_of_nodes[op_type] += 1
        stats = {
            "total_nodes": len(self.graph_output_to_node),
            "node_histogram": {
                k: types_of_nodes[k] for k in sorted(types_of_nodes, key=lambda x: types_of_nodes[x], reverse=True)
            },
            "total_constants": len(self.constants),
        }
        return json.dumps(stats, indent=2)

    # Remove all instances of Trackable_Tensor from tracer data structures
    def remove_trackable_tensors(self):
        for key, value in self.graph_output_to_input.items():
            self.graph_output_to_input[key] = [
                arg.to_frozen() if isinstance(arg, (Trackable_Tensor, TrackableTensorArgument)) else arg
                for arg in value
            ]
        for key, value in self.graph_output_to_node.items():
            if isinstance(value["args"], (list, tuple)):
                value["args"] = tree_map(
                    lambda arg: arg.to_frozen() if isinstance(arg, Trackable_Tensor) else arg,
                    value["args"],
                )

            if isinstance(value["res"], Trackable_Tensor):
                self.graph_output_to_node[key]["res"] = value["res"].to_frozen()
            if isinstance(value["res"], (list, tuple)):
                value["res"] = tree_map(
                    lambda res: res.to_frozen() if isinstance(res, Trackable_Tensor) else res,
                    value["res"],
                )

    def remove_None_and_fix_inplace(self):
        """
        Remove None values from the input tensors and fix in-place operations.

        Args:
            graph_output_to_input (dict): Mapping of node IDs to their input tensors.
            graph_output_to_node (dict): Mapping of node IDs to their attributes.

        Returns:
            outputs_to_inputs (dict): Mapping of node IDs to node IDS of their inputs.
        """
        indirection = {}
        for elem in self.graph_output_to_node:
            assert elem in self.graph_output_to_input, f"Could not find inputs for node {elem}"
            data = [tensor.tensor.id for tensor in self.graph_output_to_input[elem]]
            for index, parent in enumerate(self.graph_output_to_input[elem]):
                if parent.tensor.id in indirection:
                    if parent.index2 is not None:
                        self.graph_output_to_node[elem]["args"][parent.index][parent.index2].id = indirection[
                            parent.tensor.id
                        ]
                    else:
                        self.graph_output_to_node[elem]["args"][parent.index].id = indirection[parent.tensor.id]
                    self.graph_output_to_input[elem][index].tensor.id = indirection[parent.tensor.id]
            if (
                self.graph_output_to_node[elem]["op_type"][-1] == "_"
                or self.graph_output_to_node[elem]["op_type"].split(".Tensor")[0][-1] == "_"
                or self.graph_output_to_node[elem]["op_type"].split(".src")[0][-1] == "_"
            ):
                for parent in data:
                    indirection[parent] = elem

    def remove_tuple_get_item_detach_clone_with_no_consumers(self):
        """
        Iteratively remove TUPLE_GET_ITEM, aten::detach, and aten::clone nodes
        that end up with no consumers, using a reference-count approach.

        Args:
            graph_output_to_input (dict): Mapping of node IDs to their input tensors.
            graph_output_to_node (dict): Mapping of node IDs to their attributes.
        """
        removable_ops = {"TUPLE_GET_ITEM", "aten::detach", "aten::clone"}

        # Step 1: Build consumer counts
        consumer_count = defaultdict(int)
        for inputs in self.graph_output_to_input.values():
            for inp_ts in inputs:
                consumer_count[inp_ts.tensor.id] += 1

        # Step 2: Initialize queue with removable nodes that have no consumers
        queue = deque(
            node_id
            for node_id, node_data in self.graph_output_to_node.items()
            if node_data["op_type"] in removable_ops and consumer_count[node_id] == 0
        )

        # Step 3: Process queue
        while queue:
            node_id = queue.popleft()

            # Skip if already removed
            if node_id not in self.graph_output_to_node:
                continue

            # Remove the node
            node_inputs = self.graph_output_to_input.pop(node_id, [])
            self.graph_output_to_node.pop(node_id, None)

            # Decrement consumer count for its inputs
            for inp_ts in node_inputs:
                producer_id = inp_ts.tensor.id
                consumer_count[producer_id] -= 1
                if (
                    consumer_count[producer_id] == 0
                    and producer_id in self.graph_output_to_node
                    and self.graph_output_to_node[producer_id]["op_type"] in removable_ops
                ):
                    queue.append(producer_id)

    def propagate_module_name(self):
        """
        Post-process the graph to propagate module names and append operation types.

        Args:
            graph_output_to_input (dict): Mapping of node IDs to their input nodes.
            graph_output_to_node (dict): Mapping of node IDs to their attributes.

        Returns:
            common_prefix: The prefix in the name that is common to all nodes.
        """
        for node_id, node_data in self.graph_output_to_node.items():
            # Get the operation type
            operation_type = node_data["op_type"]

            # Get the module name from the inputs
            input_module_names = [
                self.graph_output_to_node[input_t.tensor.id]["name"]
                for input_t in self.graph_output_to_input.get(node_id, [])
                if input_t.tensor.id in self.graph_output_to_node
                and self.graph_output_to_node[input_t.tensor.id]["name"] is not None
            ]
            # Determine the base module name (use the longest one for specificity)
            base_module_name = min(input_module_names, key=lambda x: x.count("."), default=None)
            op = None
            if isinstance(node_data["res"], FrozenTrackableTensor) and node_data["res"].module_name is not None:
                op = node_data["res"].module_name
            elif isinstance(node_data["res"], (tuple, list)) and node_data["res"][0].module_name is not None:
                op = node_data["res"][0].module_name

            # Append the operation type as a postfix
            if op is not None:
                node_data["name"] = op
            elif base_module_name:
                prefix = base_module_name
                suffix = ""
                if base_module_name.count("/") == 1:
                    prefix = base_module_name.split("/")[0]
                    suffix = "/" + base_module_name.split("/")[1]
                operation_type_clean = operation_type.replace("aten::", "").replace(".Tensor", "")
                if len(prefix) > 0:
                    prefix += "."
                node_data["name"] = f"{prefix}{operation_type_clean}" + suffix
        # Remove the largest common prefix from all names
        all_names = [
            node_data["name"]
            for node_data in self.graph_output_to_node.values()
            if "name" in node_data and node_data["name"]
        ]
        common_prefix = commonprefix(all_names)

        for node_data in self.graph_output_to_node.values():
            assert "name" in node_data, "Node data must contain 'name' key"
            if node_data["name"] is None:
                continue
            if common_prefix and common_prefix in node_data["name"]:
                node_data["name"] = node_data["name"].removeprefix(common_prefix)
            node_data["name"] = ".".join(node_data["name"].split(".")[:20])  # Limit to first 20 parts

        # Ensure all names are unique by appending a unique identifier
        seen_counts = defaultdict(int)
        for node_data in self.graph_output_to_node.values():
            name = (
                node_data["name"] if node_data["name"] is not None and len(node_data["name"]) > 0 else "autogenerated"
            )
            assert name.count("/") < 2, "Name should contain at most one '/'"
            name = name.split("/")[0]
            name = name.replace("_", "")
            name = name if name is not None and len(name) > 0 else "autogenerated"
            count = seen_counts[name]
            if count > 0:
                unique_name = f"{name}/{count}"
            else:
                unique_name = name
            seen_counts[name] += 1
            node_data["name"] = unique_name
        return common_prefix

    def finalize_graph(self):
        print(f"Raw tracer statistics:\n{self.get_node_statistics()}")
        self.remove_trackable_tensors()
        self.remove_None_and_fix_inplace()
        self.finalize()
        self.remove_trackable_tensors()
        self.remove_tuple_get_item_detach_clone_with_no_consumers()
        self.propagate_module_name()


class TrackableTensorArgument:
    def __init__(self, tensor, index, index2=None):
        assert isinstance(tensor, Trackable_Tensor), "tensor must be an instance of Trackable_Tensor"
        self.tensor = tensor
        self.index = index
        self.index2 = index2

    def __repr__(self):
        shape = []
        try:
            shape = self.tensor.trackable_shape
        except:
            pass
        return (
            f"TrackableTensorArgument(pid={self.tensor.id}, tensor={shape}, index={self.index}, index2={self.index2})"
        )

    def to_frozen(self):
        """
        Convert the TrackableTensorArgument to a FrozenTrackableTensor.
        This is useful for freezing the tensor state without further modifications.
        """
        if not isinstance(self.tensor, FrozenTrackableTensor):
            self.tensor = self.tensor.to_frozen()
        return self


class FrozenTrackableTensor:
    def __init__(self, tensor):
        assert isinstance(tensor, Trackable_Tensor), "tensor must be an instance of Trackable_Tensor"
        self.id = tensor.id
        self.tensor = tensor.elem  # Store the actual tensor data
        self.module_name = tensor.module_name  # Store the module name if it exists
        self.shape = tensor.shape
        self.graph_output_index = None
        self.const_name = tensor.const_name
        if "graph_output_index" in tensor.__dict__:
            self.graph_output_index = tensor.graph_output_index

    def __repr__(self):
        return f"FrozenTrackableTensor(pid={self.id}, tensor={self.shape})"

    @property
    def is_graph_input(self):
        assert self.id != "-1", "Trackable_Tensor must have a valid id. Got -1."
        return "-" in self.id and self.const_name is None


class Trackable_Tensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ["elem"]

    # Static variable to hold the tracer instance
    tracer_data: Optional[TracerData] = None

    @staticmethod
    def set_tracer_data(tracer_instance):
        """Set the tracer instance for the Trackable_Tensor class."""
        Trackable_Tensor.tracer_data = tracer_instance

    @staticmethod
    def get_tracer_data():
        """Get the tracer data instance for the Trackable_Tensor class."""
        if Trackable_Tensor.tracer_data is None:
            raise ValueError("Tracer Data has not been set for Trackable_Tensor.")
        return Trackable_Tensor.tracer_data

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # The wrapping tensor (Trackable_Tensor) shouldn't hold any
        # memory for the class in question, but it should still
        # advertise the same device as before
        output_shape = elem.size()
        strides = elem.stride()
        output_dtype = elem.dtype
        if "output_shape" in kwargs:
            output_shape = kwargs.get("output_shape", elem.size())
            del kwargs["output_shape"]
        if "output_dtype" in kwargs:
            output_dtype = kwargs.get("output_dtype", elem.dtype)
            del kwargs["output_dtype"]
        if output_shape != elem.size() or output_dtype != elem.dtype:
            strides = torch.ones(output_shape, dtype=output_dtype).stride()
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            output_shape,
            strides=strides,
            storage_offset=elem.storage_offset(),
            # TODO: clone storage aliasing
            dtype=output_dtype,
            layout=elem.layout,
            device=elem.device if Trackable_Tensor.tracer_data.save_original_tensors else "meta",
            requires_grad=elem.requires_grad,
        )
        # ...the real tensor is held as an element on the tensor.
        r.module_name = None  # Initialize module_name
        r.elem = elem if Trackable_Tensor.tracer_data.save_original_tensors else None
        r.trackable_shape = elem.shape
        r.trackable_dtype = elem.dtype
        r.id = "-1"
        r.const_name = None
        return r

    def __repr__(self):
        elem = self.elem if "elem" in self.__dict__ else None
        module_name = self.module_name if "module_name" in self.__dict__ else None
        return f"Trackable_Tensor({elem}, module_name={module_name})"

    def set_id(self, id):
        self.id = id

    def set_module_name(self, module_name):
        """Set the module name for this tensor."""
        assert module_name is not None, "Module name cannot be None"
        if self.module_name is None:
            self.module_name = module_name
        else:
            if self.module_name in module_name and len(module_name) > len(self.module_name):
                self.module_name = module_name

    @classmethod
    def initialize_tracking_info(cls, args, kwargs, rs, func_name, func_module):
        if "_." in func_name or func_name.endswith("_"):
            print(
                f"Warning: found inplace operation {func_name}. Connections may be incorrect. Please rewrite to avoid inplace operation."
            )
        tracer = cls.get_tracer_data()  # Use the tracer data instance
        id, graph_output_to_input, graph_output_to_node = (
            tracer.get_next_id(),
            tracer.graph_output_to_input,
            tracer.graph_output_to_node,
        )
        local_id = str(id)
        graph_output_to_input[local_id] = []
        input_shapes = []
        input_dtypes = []
        # ONLY TWO LEVELS OF NESTING ARE SUPPORTED IN ARGS
        # EXAMPLE: (a, b, (c, d)) -> a, b, c, d
        # NOT SUPPORTED: (a, b, (c, d, (e, f)))
        for index, argument in enumerate(args):
            if isinstance(argument, Trackable_Tensor):
                graph_output_to_input[local_id].append(TrackableTensorArgument(argument, index))
                input_shapes.append(argument.trackable_shape)
                input_dtypes.append(str(argument.trackable_dtype))
            elif isinstance(argument, (list, tuple)):
                for index2, arg in enumerate(argument):
                    if isinstance(arg, Trackable_Tensor):
                        graph_output_to_input[local_id].append(TrackableTensorArgument(arg, index, index2))
                        input_shapes.append(arg.trackable_shape)
                        input_dtypes.append(str(arg.trackable_dtype))
        output_shapes = []
        output_dtypes = []
        if isinstance(rs, Trackable_Tensor):
            rs.set_id(local_id)
            output_shapes.append(rs.trackable_shape)
            output_dtypes.append(str(rs.trackable_dtype))
        elif isinstance(rs, (list, tuple)):
            for index, r in enumerate(rs):
                if isinstance(r, Trackable_Tensor):
                    # Setting global info for the result
                    r.set_id(local_id + "_" + str(index))
                    output_shapes.append(r.trackable_shape)
                    output_dtypes.append(str(r.trackable_dtype))

                    # Setting local info for the result
                    meta = {}
                    meta["i_shapes"] = input_shapes
                    meta["i_dtypes"] = input_dtypes
                    meta["o_shapes"] = [r.trackable_shape]
                    meta["o_dtypes"] = [str(r.trackable_dtype)]
                    new_tensor = Trackable_Tensor(r)
                    new_tensor.set_id(local_id)
                    tracer.post_run_output_to_node[local_id + "_" + str(index)] = {
                        "name": None,
                        "op_type": "TUPLE_GET_ITEM",
                        "torch_op": "",
                        "id": local_id + "_" + str(index),
                        "args": rs,
                        "kwargs": {"index": index},
                        "meta": meta,
                        "res": r,
                    }
                    tracer.post_run_output_to_input[local_id + "_" + str(index)] = [
                        TrackableTensorArgument(new_tensor, 0, k) for k in range(len(rs))
                    ]
        elif isinstance(rs, bool):
            ## (element == self).any().item()
            new_tensor = Trackable_Tensor(torch.tensor(rs))
            new_tensor.set_id(local_id)
            output_shapes.append(new_tensor.shape)
            output_dtypes.append(str(new_tensor.dtype))
            meta = {}
            meta["i_shapes"] = input_shapes
            meta["i_dtypes"] = input_dtypes
            meta["o_shapes"] = output_shapes
            meta["o_dtypes"] = output_dtypes
            graph_output_to_node[local_id] = {
                "name": None,
                "op_type": func_name,
                "torch_op": {func_module},
                "id": local_id,
                "args": args,
                "kwargs": kwargs,
                "meta": meta,
                "res": new_tensor,
            }
            return
        else:
            print(f"could not set id for result of operation {id}, not a Trackable_Tensor")
        meta = {}
        meta["i_shapes"] = input_shapes
        meta["i_dtypes"] = input_dtypes
        meta["o_shapes"] = output_shapes
        meta["o_dtypes"] = output_dtypes
        graph_output_to_node[local_id] = {
            "name": None,
            "op_type": func_name,
            "torch_op": {func_module},
            "id": local_id,
            "args": args,
            "kwargs": kwargs,
            "meta": meta,
            "res": rs,
        }

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            res = e
            if isinstance(e, Trackable_Tensor):
                if Trackable_Tensor.tracer_data.save_original_tensors:
                    res = e.elem
                else:
                    if e.numel() <= 1 and len(e.trackable_shape) == 0:
                        res = torch.empty(1, device="meta", dtype=e.trackable_dtype)
                    else:
                        res = torch.empty(*e.trackable_shape, device="meta", dtype=e.trackable_dtype)
            else:
                if isinstance(e, torch.Tensor) and not Trackable_Tensor.tracer_data.save_original_tensors:
                    res = torch.empty(*e.shape, device="meta", dtype=e.dtype)
            return res

        def wrap(e):
            return Trackable_Tensor(e) if isinstance(e, torch.Tensor) else e

        # no_dispatch is only needed if you use enable_python_mode.
        # It prevents infinite recursion.
        with no_dispatch():
            if func.name() == "aten::_local_scalar_dense":
                example = """
                try:
                    output = hidden_states.new_trackable_tensor(
                        hidden_states,
                        router_indices,
                        routing_weights,
                        self.gate_up_proj,
                        self.gate_up_proj_bias,
                        self.down_proj,
                        self.down_proj_bias,
                        self.alpha,
                        self.limit,
                        func_name="gpt_oss_experts",
                        func_module=self.forward.__module__,
                        output_shape=hidden_states.shape,
                        output_dtype=hidden_states.dtype
                    )
                    return output
                except:
                    pass
                """
                print(
                    "Found item() operation. Please rewrite it to avoid non-deterministic behavior. Tracer tool will continue, but result will be for different graph input."
                )
                print("See example below:", example)
                if Trackable_Tensor.tracer_data.save_original_tensors:
                    # If the tensor is a Trackable_Tensor, return its elem
                    return args[0].elem.item() if isinstance(args[0], Trackable_Tensor) else args[0].item()
                else:
                    return (
                        torch.zeros(args[0].trackable_shape, dtype=args[0].trackable_dtype).item()
                        if isinstance(args[0], Trackable_Tensor)
                        else args[0].item()
                    )
            else:
                func_args = tree_map(unwrap, args)
                func_kwargs = tree_map(unwrap, kwargs)
                func_res = func(*func_args, **func_kwargs)
                rs = tree_map(wrap, func_res)
        cls.initialize_tracking_info(args, kwargs, rs, func.name(), func.__module__)
        return rs

    def new_trackable_tensor(self, *args, func_name, func_module, output_shape, output_dtype, **kwargs):
        kwargs["output_shape"] = output_shape
        kwargs["output_dtype"] = output_dtype
        rs = Trackable_Tensor(*args, **kwargs)
        kwargs = {k: v for k, v in kwargs.items() if k not in ["output_shape", "output_dtype"]}
        assert output_shape is not None, "output_shape must be provided"
        assert output_dtype is not None, "output_dtype must be provided"
        rs.elem = (
            torch.ones(output_shape, dtype=output_dtype) if Trackable_Tensor.tracer_data.save_original_tensors else None
        )
        rs.module_name = None  # Initialize module_name
        rs.id = "-1"
        rs.const_name = None
        self.initialize_tracking_info(args, kwargs, rs, func_name, func_module)
        rs.trackable_shape = output_shape
        rs.trackable_dtype = output_dtype
        return rs

    def to_frozen(self):
        """
        Convert the Trackable_Tensor to a FrozenTrackableTensor.
        This is useful for freezing the tensor state without further modifications.
        """
        return FrozenTrackableTensor(self)


class OperationGraph:
    def __init__(self, tracer: TracerData):
        self.tracer = tracer
        self.operations: Dict[str, Operation] = {}
        self.graph = nx.DiGraph()

    def parse_args(self, args: List[Any], op_unique_name: str) -> List[Any]:
        """Parse arguments, converting tensors and nested structures."""
        if op_unique_name is None:
            op_unique_name = ""

        def process_arg(arg, index):
            if isinstance(arg, FrozenTrackableTensor) and arg.const_name is None:
                if arg.id in self.tracer.graph_output_to_node or arg.is_graph_input:
                    name = (
                        self.tracer.graph_output_to_node[arg.id]["name"] if not arg.is_graph_input else arg.module_name
                    )
                    return PlaceholderTensor(name=name, value=arg, shape=arg.shape)
            elif isinstance(arg, FrozenTrackableTensor):
                result = ConstantTensor(name=arg.id, value=arg.tensor, id=arg.const_name, shape=arg.shape)
                self.tracer.constants[result.name] = result
                return result
            elif isinstance(arg, torch.Tensor):
                result = ConstantTensor(name=f"{op_unique_name}_{index}", value=arg, shape=arg.shape)
                self.tracer.constants[result.name] = result
                return result
            elif isinstance(arg, (list, tuple)):
                return self.parse_args(arg, op_unique_name)
            return arg

        return [process_arg(arg, index) for index, arg in enumerate(args)]

    def create_operations(self):
        """Create operations from the tracer data."""
        for node_id, node_data in self.tracer.graph_output_to_node.items():
            function_call_name = self._get_function_call_name(node_data)
            kwargs = node_data.get("kwargs", {})
            meta = node_data.get("meta", {})
            res = node_data.get("res", None)
            name = node_data["name"]
            graph_output_indices = None
            if isinstance(res, FrozenTrackableTensor) and res.graph_output_index is not None:
                graph_output_indices = [res.graph_output_index]
            elif isinstance(res, (list, tuple)):
                intermediate_indices = []
                for r in res:
                    if isinstance(r, FrozenTrackableTensor) and r.graph_output_index is not None:
                        intermediate_indices.append(r.graph_output_index)
                if intermediate_indices:
                    graph_output_indices = intermediate_indices
            if node_data["op_type"] == "TUPLE_GET_ITEM":
                if "." not in name:
                    name = "." + name
                bn_name = name.split(".")[-1].split("/")
                if len(bn_name) > 0:
                    bn_name[0] = f"{bn_name[0]}{kwargs['index']}"
                prefix = ".".join(name.split(".")[:-1])
                if len(prefix) > 0:
                    prefix += "."
                node_data["name"] = prefix + "/".join(bn_name)
                name = node_data["name"]
                args = self.parse_args(node_data.get("args", []), node_data["name"])
                parent_index = node_data["kwargs"]["index"]
                parent_id = node_data["args"][parent_index].id.split("_")[0]
                assert parent_id in self.operations, f"Parent ID {parent_id} not found in operations."
                new_args = [arg for arg in args]
                new_args[parent_index] = PlaceholderTensor(
                    name=self.operations[parent_id].unique_name, shape=new_args[parent_index].shape
                )
                operation = TupleOp(
                    id=node_id,
                    unique_name=name,
                    function_call_name=function_call_name,
                    args=new_args,
                    kwargs=kwargs,
                    meta_data=OperationMetadata(meta=meta, res=self.parse_args([res], node_data["name"])[0]),
                    graph_output_indices=graph_output_indices,
                )
            else:
                args = self.parse_args(node_data.get("args", []), node_data["name"])
                operation = Operation(
                    id=node_id,
                    unique_name=name,
                    function_call_name=function_call_name,
                    args=args,
                    kwargs=kwargs,
                    meta_data=OperationMetadata(meta=meta, res=self.parse_args([res], node_data["name"])[0]),
                    graph_output_indices=graph_output_indices,
                )
            self.operations[node_id] = operation
            self._handle_input_operations(node_data)

    def _get_function_call_name(self, node_data: Dict[str, Any]) -> str:
        """Get the function call name for an operation."""
        if node_data["op_type"] == "TUPLE_GET_ITEM":
            return ""
        return (
            f"{list(node_data['torch_op'])[0].replace('torch._op', 'torch.op')}.{node_data['op_type'].split('::')[-1]}"
        )

    def _handle_input_operations(self, node_data: Dict[str, Any]):
        """Handle input operations for tensors."""
        for arg in node_data.get("args", []):
            if isinstance(arg, FrozenTrackableTensor) and arg.is_graph_input:
                self.operations[arg.id] = InputOp(
                    id=arg.id,
                    unique_name=arg.module_name,
                    function_call_name="torch.ones",
                    args=[arg.shape],
                    kwargs={},
                )
                self.operations[arg.id].meta_data = OperationMetadata(
                    meta={
                        "o_shapes": [arg.shape],
                    },
                    res=PlaceholderTensor(name=arg.module_name, shape=arg.shape),
                )
                self.graph.add_edge(arg.id, node_data["id"])
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, FrozenTrackableTensor) and item.is_graph_input:
                        self.operations[item.id] = InputOp(
                            id=item.id,
                            unique_name=item.module_name,
                            function_call_name="torch.ones",
                            args=[item.shape, item],
                            kwargs={},
                        )
                        self.operations[item.id].meta_data = OperationMetadata(
                            meta={
                                "o_shapes": [item.shape],
                            },
                            res=PlaceholderTensor(name=item.module_name, shape=item.shape),
                        )
                        self.graph.add_edge(item.id, node_data["id"])

    def build_graph(self):
        """Build a directed graph using operations and their connections."""
        for op_id, operation in self.operations.items():
            self.graph.add_node(op_id, operation=operation)

        for output, inputs in self.tracer.graph_output_to_input.items():
            for input_node in inputs:
                if output in self.operations and input_node.tensor.id in self.operations:
                    self.graph.add_edge(input_node.tensor.id, output)
                elif output in self.operations and input_node.tensor.const_name is not None:
                    continue
                else:
                    print(f"Warning: Edge from {output} to {input_node} could not be added.")

    def generate(self):
        """
        Generate operations and build the graph.
        """
        self.create_operations()
        self.build_graph()

    @staticmethod
    def from_operation_graph(other_graph: "OperationGraph") -> "OperationGraph":
        """
        Create a new OperationGraph instance from an existing one.

        Args:
            other_graph (OperationGraph): The existing OperationGraph instance.

        Returns:
            OperationGraph: A new OperationGraph instance with copied data.
        """
        new_graph = OperationGraph(other_graph.tracer)
        new_graph.operations = {k: v for k, v in other_graph.operations.items()}
        new_graph.graph = other_graph.graph.copy()
        return new_graph


class WrappedOperationGraph(OperationGraph):
    def __init__(self, tracer: TracerData, verbose=False):
        super().__init__(tracer)
        self.verbose = verbose

    def convert_operations_to_wrapped(self):
        unsupported_wrapped_ops = set()
        for node_id, operation in self.operations.items():
            operation_class = get_operation_class(operation.function_call_name)
            if operation_class:
                self.operations[node_id] = operation.to_operation(operation_class)
            elif len(operation.function_call_name) > 0:
                unsupported_wrapped_ops.add(operation.function_call_name)
        if self.verbose and unsupported_wrapped_ops:
            print(f"Unsupported wrapped operations: {', '.join(unsupported_wrapped_ops)}")

    def remove_detach_on_constants(self):
        for node_id in nx.topological_sort(self.graph):
            operation = self.operations[node_id]
            if operation.function_call_name == "torch.ops.aten.detach":
                if isinstance(operation.args[0], ConstantTensor):
                    parent = operation.args[0]
                    for successor in self.graph.successors(node_id):
                        succ_op = self.operations[successor]
                        new_args = []
                        for succ_arg in succ_op.args:
                            if isinstance(succ_arg, PlaceholderTensor) and succ_arg.name == operation.unique_name:
                                new_args.append(parent)
                            elif isinstance(succ_arg, list):
                                new_list = []
                                for item in succ_arg:
                                    if isinstance(item, PlaceholderTensor) and item.name == operation.unique_name:
                                        new_list.append(parent)
                                    else:
                                        new_list.append(item)
                                new_args.append(new_list)
                            else:
                                new_args.append(succ_arg)
                        succ_op.args = new_args
                        self.operations[successor] = succ_op
                        self.graph.nodes[successor]["operation"] = succ_op
                    for edge in list(self.graph.out_edges(node_id)):
                        self.graph.remove_edge(*edge)
                    self.graph.remove_node(node_id)

    def generate(self):
        """
        Generate operations and build the graph.
        """
        self.create_operations()
        self.convert_operations_to_wrapped()
        self.build_graph()
        self.remove_detach_on_constants()


def register_module_hooks(module, parent_name="", module_calls=None):
    """
    Register hooks to track tensors passing through leaf modules only.

    Args:
        module: The PyTorch module to register hooks on.
        parent_name: The hierarchical name of the parent module.

    Returns:
        List of hook handles for all registered hooks.
    """
    module_name = f"{parent_name}.{module.__class__.__name__}" if parent_name else module.__class__.__name__
    hook_handles = []
    if module_calls is None:
        module_calls = {}

    # Check if the module is a leaf (has no submodules)
    if len(list(module.named_children())) == 0:
        if module_name in module_calls:
            module_calls[module_name] += 1
            module_name = f"{module_name}/{module_calls[module_name]}"
        else:
            module_calls[module_name] = 0

        def forward_hook(module, inputs, outputs):
            def set_module_name(tensor):
                if isinstance(tensor, Trackable_Tensor):
                    tensor.set_module_name(module_name)  # Set the full hierarchical name

            # Set the module name for outputs
            tree_map(set_module_name, outputs)

        # Register the forward hook and store the handle
        hook_handle = module.register_forward_hook(forward_hook)
        hook_handles.append(hook_handle)
    else:
        # Recursively register hooks for submodules
        for name, submodule in module.named_children():
            hook_handles.extend(register_module_hooks(submodule, parent_name=module_name, module_calls=module_calls))

    return hook_handles


def get_serialized_info(value):
    """
    Serialize the information to a format suitable for JSON.
    This function converts tensors to lists and handles other types accordingly.
    """

    if isinstance(value, torch.Tensor):
        return get_serialized_info(value.tolist())  # Convert tensor to list
    elif isinstance(value, (list, tuple, set)):
        return [get_serialized_info(v) for v in value]
    elif isinstance(value, dict):
        return {k: get_serialized_info(v) for k, v in value.items()}
    elif isinstance(value, (torch.memory_format, torch.dtype, torch.device)):
        return str(value)
    return value


def serialize_attrs(attrs):
    """
    Serialize attributes to a format suitable for JSON.
    This function converts attributes to a dictionary format.
    """
    serialized = {}
    for key, value in attrs.items():
        serialized[key] = get_serialized_info(value)
    return serialized


# Assuming `graph_output_to_node` and `outputs_to_inputs` are already populated
def create_json_structure(graph_output_to_node, outputs_to_inputs):
    nodes = []
    arg_nodes = []
    node_row_ptr = []
    current_row = 0
    id_to_row = {}
    outputs = []
    for node, inputs in outputs_to_inputs.items():
        # Example attributes for demonstration purposes
        op = graph_output_to_node[node]["name"]
        attrs = graph_output_to_node[node]["kwargs"]
        serialized_attrs = serialize_attrs(attrs)
        serialized_attrs.update(graph_output_to_node[node]["meta"])
        node_entry = {
            "op": graph_output_to_node[node]["op_type"].replace("aten::", "").replace(".", "_"),
            "name": op,
            "attrs": serialized_attrs,
            "inputs": [
                [id_to_row[input_node.tensor.id], 0, 0] if input_node.tensor.id in id_to_row else []
                for input_node in inputs
            ],
        }
        nodes.append(node_entry)
        if not inputs:  # Assuming nodes without inputs are argument nodes
            arg_nodes.append(len(nodes) - 1)
        id_to_row[node] = current_row
        if isinstance(graph_output_to_node[node]["res"], FrozenTrackableTensor):
            if graph_output_to_node[node]["res"].graph_output_index is not None:
                outputs.append((graph_output_to_node[node]["res"].graph_output_index, current_row))
        elif isinstance(graph_output_to_node[node]["res"], (list, tuple)):
            for res in graph_output_to_node[node]["res"]:
                if isinstance(res, FrozenTrackableTensor) and res.graph_output_index is not None:
                    outputs.append((res.graph_output_index, current_row))
                    break
        current_row += 1
        node_row_ptr.append(current_row)

    # Assuming the last node is the head
    heads = [[output[1], 0, 0] for output in sorted(outputs, key=lambda x: x[0])]

    # Construct the final JSON structure
    json_structure = {
        "nodes": nodes,
        "arg_nodes": arg_nodes,
        "heads": heads,
        "node_row_ptr": node_row_ptr,
        "attrs": {"mxnet_version": ["int", 10700]},
    }

    return json_structure


def get_attrs_for_op(operation: Operation, node) -> Dict[str, Any]:
    """
    Get attributes for the operation to be serialized.

    Args:
        operation (Operation): The operation for which to get attributes.

    Returns:
        Dict[str, Any]: A dictionary of attributes for the operation.
    """
    attrs = {}
    if operation.function_call_name == "torch.ops.aten.convolution":
        attrs["hidden_units"] = operation.args[1].shape[0]
        if len(operation.args[1].shape) == 4:
            attrs["kernel"] = [operation.args[1].shape[2], operation.args[1].shape[3]]
        elif len(operation.args[1].shape) == 3:
            attrs["kernel"] = [operation.args[1].shape[2]]
        attrs["stride"] = operation.args[3]
        attrs["padding"] = operation.args[4]
        attrs["dilation"] = operation.args[5]
        attrs["groups"] = operation.args[8]
    elif operation.function_call_name == "torch.ops.aten.max_pool2d_with_indices":
        attrs["kernel"] = operation.args[1]
        attrs["stride"] = operation.args[2]
        if len(operation.args) > 3:
            attrs["padding"] = operation.args[3]
        if len(operation.args) > 4:
            attrs["dilation"] = operation.args[4]
    elif operation.function_call_name == "torch.ops.aten.cat":
        attrs["dim"] = operation.args[1] if len(operation.args) > 1 else 0
    elif operation.function_call_name == "torch.ops.aten.slice.Tensor":
        attrs["dim"] = operation.args[1]
        if len(operation.args) > 2:
            attrs["start"] = operation.args[2]
        if len(operation.args) > 3:
            attrs["end"] = operation.args[3]
        if len(operation.args) > 4:
            attrs["step"] = operation.args[4]
    elif "aten.transpose." in operation.function_call_name:
        attrs["dim0"] = operation.args[1]
        attrs["dim1"] = operation.args[2]
    elif "._softmax" in operation.function_call_name or ".softmax" in operation.function_call_name:
        attrs["dim"] = operation.args[1] if len(operation.args) > 1 else 0
    elif operation.function_call_name == "torch.ops.aten.split_with_sizes":
        attrs["split_sizes"] = operation.args[1]
        attrs["dim"] = operation.args[2] if len(operation.args) > 2 else 0
    elif operation.function_call_name == "torch.ops.aten.grid_sampler_2d":
        attrs["interpolation_mode"] = operation.args[2]
        if len(operation.args) > 3:
            attrs["padding_mode"] = operation.args[3]
        if len(operation.args) > 4:
            attrs["align_corners"] = operation.args[4] if len(operation.args) > 4 else False
    elif operation.function_call_name == "torch.ops.aten.permute":
        attrs["dims"] = operation.args[1] if len(operation.args) > 1 else None
    for index, arg in enumerate(operation.args):
        if isinstance(arg, ConstantTensor):
            attrs[f"arg_{index}_shape"] = arg.shape
    delete_attrs = []
    for key, value in attrs.items():
        if isinstance(value, Parameter):
            print(f"Skipping Parameter value {value} in attrs {key} of {operation.function_call_name}")
            delete_attrs.append(key)
    attrs = {k: v for k, v in attrs.items() if k not in delete_attrs}
    attrs["id"] = node
    return attrs


def create_graph_json_structure(operation_graph: OperationGraph):
    nodes = []
    arg_nodes = []
    node_row_ptr = []
    current_row = 0
    id_to_row = {}
    outputs = []
    for node in list(nx.topological_sort(operation_graph.graph)):
        inputs = list(operation_graph.graph.predecessors(node))
        # Example attributes for demonstration purposes
        op = operation_graph.graph.nodes[node]["operation"].unique_name
        attrs = operation_graph.graph.nodes[node]["operation"].kwargs
        serialized_attrs = serialize_attrs(attrs)
        meta_data = operation_graph.graph.nodes[node]["operation"].meta_data
        if meta_data is None:
            meta_data = {}
        else:
            meta_data = meta_data.meta
        serialized_attrs.update(serialize_attrs(meta_data))
        serialized_attrs.update(get_attrs_for_op(operation_graph.graph.nodes[node]["operation"], node))
        function_call_name = operation_graph.graph.nodes[node]["operation"].function_call_name
        if function_call_name is None or function_call_name == "":
            function_call_name = operation_graph.graph.nodes[node]["operation"].__class__.__name__
        op_name = function_call_name.replace("torch.ops.", "")
        if op_name.endswith(".Tensor"):
            op_name = op_name[:-7]
        node_entry = {
            "op": op_name,
            "name": op,
            "attrs": serialized_attrs,
            "inputs": [[id_to_row[input_node], 0, 0] if input_node in id_to_row else [] for input_node in inputs],
        }
        try:
            json_dumps = json.dumps(node_entry)
        except Exception as e:
            print(f"Error serializing node {node}: {node_entry}")
        nodes.append(node_entry)
        if not inputs:  # Assuming nodes without inputs are argument nodes
            arg_nodes.append(len(nodes) - 1)
        id_to_row[node] = current_row
        if operation_graph.graph.nodes[node]["operation"].graph_output_indices is not None:
            outputs.append((min(operation_graph.graph.nodes[node]["operation"].graph_output_indices), current_row))
        current_row += 1
        node_row_ptr.append(current_row)

    # Assuming the last node is the head
    heads = [[output[1], 0, 0] for output in sorted(outputs, key=lambda x: x[0])]

    # Construct the final JSON structure
    json_structure = {
        "nodes": nodes,
        "arg_nodes": arg_nodes,
        "heads": heads,
        "node_row_ptr": node_row_ptr,
        "attrs": {"mxnet_version": ["int", 10700]},
    }

    return json_structure


def set_is_graph_output(outputs, index=0):
    """
    Set the is_graph_output attribute for Trackable_Tensor instances in the outputs.

    Args:
        outputs: The outputs of the traced model.
    """
    if isinstance(outputs, Trackable_Tensor):
        outputs.graph_output_index = index
    elif isinstance(outputs, (list, tuple)):
        for index2, output in enumerate(outputs):
            ## no recursion since we only support one level of nesting
            if isinstance(output, Trackable_Tensor):
                output.graph_output_index = index2
            elif isinstance(output, (list, tuple)):
                set_is_graph_output(output, index2)
    elif outputs is not None:
        values = []
        if isinstance(outputs, dict):
            values = list(outputs.values())
        else:
            values = list(outputs.__dict__.values())
        for output in values:
            if isinstance(output, Trackable_Tensor):
                output.graph_output_index = index
            elif isinstance(output, (list, tuple)):
                set_is_graph_output(output, index)


def wrap_state_dict(state_dict):
    """
    Wrap the tensors in the state_dict with Trackable_Tensor.

    Args:
        state_dict: The state_dict of the model.

    Returns:
        The wrapped state_dict.
    """

    wrapped_state_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            wrapped_tensor = Trackable_Tensor(value)
            wrapped_tensor.id = str(Trackable_Tensor.tracer_data.get_next_id())
            wrapped_tensor.const_name = key
            wrapped_tensor.elem = value.to(wrapped_tensor.device)
            wrapped_state_dict[key] = wrapped_tensor
        else:
            wrapped_state_dict[key] = value
    wrapped_state_dict = OrderedDict(wrapped_state_dict)
    return wrapped_state_dict


def wrap_input_tensor(input_tensors, tensor, index):
    input_tensor = Trackable_Tensor(tensor)
    input_tensor.set_id(f"{(index+2)*-1}")
    input_tensor.elem = tensor.to(input_tensor.device)
    input_tensor.set_module_name(f"input_tensor_{index}")
    input_tensors.append(input_tensor)


def get_input_tensors(input_shapes, input_dtypes=None, sample_input_tensors=None):
    input_tensors = []
    if input_dtypes is None:
        input_dtypes = [torch.float32] * len(input_shapes)
    assert (
        len(input_shapes) > 0 or sample_input_tensors is not None
    ), "Input shapes or sample input tensors must be provided"
    if sample_input_tensors is not None:
        for index, tensor in enumerate(sample_input_tensors):
            wrap_input_tensor(input_tensors, tensor, index)
    else:
        for index, input_shape in enumerate(input_shapes):
            rand_tensor = (torch.rand(tuple(input_shape))) * 10
            # create torch dtype from string if input_dtypes is a list of strings
            if isinstance(input_dtypes[index % len(input_dtypes)], str):
                input_dtypes[index % len(input_dtypes)] = getattr(torch, input_dtypes[index % len(input_dtypes)])
            if isinstance(input_dtypes[index % len(input_dtypes)], torch.dtype):
                rand_tensor = rand_tensor.to(dtype=input_dtypes[index % len(input_dtypes)])
            wrap_input_tensor(input_tensors, rand_tensor, index)
    return input_tensors


def generate_graph_and_visualize(tracer, dump_visualization=False, wrap_operations=True):
    try:
        if wrap_operations:
            operation_graph = WrappedOperationGraph(tracer, verbose=True)
        else:
            operation_graph = OperationGraph(tracer)
        operation_graph.generate()
        print(f"Found {len(operation_graph.graph)} operations in the traced model.")
        print(f"Tracer statistics after parsing:\n{tracer.get_node_statistics()}")
        if dump_visualization:
            json_structure = create_graph_json_structure(operation_graph)
            # Write the JSON structure to the specified output file
            with open("operation_graph_viz.json", "w") as f:
                json.dump(json_structure, f, indent=2)
            print(
                f"Dumped visualization to {os.path.abspath('operation_graph_viz.json')}. Load it into netron.app to visualize the model."
            )
    except Exception as e:
        if dump_visualization:
            json_structure = create_json_structure(tracer.graph_output_to_node, tracer.graph_output_to_input)
            # Write the JSON structure to the specified output file
            with open("trace_viz.json", "w") as f:
                json.dump(json_structure, f, indent=2)
            print(
                f"Dumped visualization to {os.path.abspath('trace_viz.json')}. Load it into netron.app to visualize the model."
            )
        raise e
    return operation_graph


def trace_torch_model(
    model,
    input_shapes,
    input_dtypes=None,
    dump_visualization=False,
    wrap_operations=True,
    save_original_tensors=True,
    track_params=True,
    input_tensors=None,
) -> OperationGraph:
    """
    Trace the PyTorch model with the given input tensor.

    Args:
        model: The PyTorch model to trace.
        input_shapes: List of shapes for the input tensors.
        dump_visualization: If True, dump the visualization to a JSON file.

    Returns:
        OperationGraph: The operation graph generated from the traced model.

        if wrap_operations is True, the OperationGraph will be wrapped in WrappedOperationGraph.

    """

    assert len(input_shapes) > 0, "Input shapes must be provided"

    tracer = TracerData(save_original_tensors=save_original_tensors, track_params=track_params)
    if track_params:
        state_dict = model.state_dict()
        Trackable_Tensor.set_tracer_data(tracer)  # Set the tracer data instance for Trackable_Tensor
        wrapped_state_dict = wrap_state_dict(state_dict)
        model.load_state_dict(wrapped_state_dict, assign=True)
    else:
        Trackable_Tensor.set_tracer_data(tracer)  # Set the tracer data instance for Trackable_Tensor
    input_tensors = get_input_tensors(input_shapes, input_dtypes, input_tensors)
    module_calls = {}
    handles = register_module_hooks(model, module_calls=module_calls)
    outputs = model(*input_tensors)
    set_is_graph_output(outputs)
    # remove the hooks after tracing
    for handle in handles:
        handle.remove()

    tracer.finalize_graph()
    operation_graph = generate_graph_and_visualize(tracer, dump_visualization, wrap_operations)
    return operation_graph


def trace_torch_function(func, input_shapes, input_dtypes=None, dump_visualization=False):
    class GenericClass(torch.nn.Module):
        def forward(self, *args):
            return func(*args)

    model = GenericClass()
    operation_graph = trace_torch_model(model, input_shapes, input_dtypes, dump_visualization=dump_visualization)
    return operation_graph
