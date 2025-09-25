import hashlib
import json
import os
import networkx as nx
from dataclasses import dataclass, field
from tracer_backend_utils import (
    Operation,
    WrappedOperation,
    to_valid_variable_name,
    PlaceholderTensor,
    TupleOp,
    ConstantTensor,
    InputOp,
    AtenReluInplace,
    AtenRelu,
    AtenNativeBatchNorm,
    AtenConvolution,
    AtenScaledDotProductFlashAttention,
    WRAPPED_OPERATION_REGISTRY,
)
from tracer_backend import (
    create_graph_json_structure,
    OperationGraph,
    Trackable_Tensor,
    get_input_tensors,
    TracerData,
    set_is_graph_output,
    generate_graph_and_visualize,
    wrap_state_dict,
    register_module_hooks,
)
from typing import ClassVar, List, Dict, Any, Tuple, Optional
from collections import defaultdict


class WrappedOpPatternObj:
    def compareWith(self, wrapped_op: WrappedOperation):
        if self.__class__.__name__ == "WildcardPatternObj":
            return True
        return wrapped_op.__class__.__name__ == self.wrapped_op.__name__


class WrappedOpPatternObjBaseMeta(type):
    def __new__(cls, name, bases, dct):
        if WrappedOpPatternObj not in bases:
            bases = (WrappedOpPatternObj,) + bases
        if "__init__" not in dct:

            def __init__(self, *args):
                for arg in args:
                    assert isinstance(
                        arg, WrappedOpPatternObj
                    ), f"Expected WrappedOpPatternObj, got {type(arg)} for {self}"
                self.parents = args

            dct["__init__"] = __init__
        return super().__new__(cls, name, bases, dct)


@dataclass
class CompositeOperation(Operation):
    """An operation that represents a subgraph of nested operations."""

    sub_operations: List[Operation] = field(default_factory=list)
    fun_body: Optional[str] = None
    func: Optional[str] = None
    code_lines: Optional[List[str]] = None
    vars: Optional[List[str]] = None
    consts: Optional[List[str]] = None
    local_var_names: Optional[List[str]] = None
    append_var_names_to_end: Optional[List[str]] = None
    is_const_fold_composite: bool = False
    graph_outputs: List[str] = field(default_factory=list)
    counter: ClassVar[int] = 0
    generated_code: ClassVar[Dict[str, str]] = None
    duplicate_ops: ClassVar[Dict[str, str]] = None
    prune: ClassVar[bool] = True
    ALL_CONSTANTS: ClassVar[Dict[str, ConstantTensor]] = None

    def __post_init__(self):
        CompositeOperation.counter += 1
        if CompositeOperation.generated_code is None:
            CompositeOperation.generated_code = {}
        if CompositeOperation.duplicate_ops is None:
            CompositeOperation.duplicate_ops = {}
        if CompositeOperation.ALL_CONSTANTS is None:
            CompositeOperation.ALL_CONSTANTS = {}

    def get_unique_representation(self) -> Dict[str, Any]:
        base = super().get_unique_representation()
        base["sub_operations"] = [op.get_unique_representation() for op in self.sub_operations]

        return base

    def reorder_args(self, append_var_name_to_end):
        if self.append_var_names_to_end is None:
            self.append_var_names_to_end = append_var_name_to_end
        else:
            self.append_var_names_to_end.extend(append_var_name_to_end)
        self.append_var_names_to_end = list(set(self.append_var_names_to_end))
        _, _, arg_all_args, arg_consts = self.get_code_lines_and_args()
        arg_all_args_none_consts = [arg for arg in arg_all_args if arg not in self.append_var_names_to_end]
        arg_all_args_consts = [arg for arg in arg_all_args if arg in self.append_var_names_to_end]
        arg_all_args = arg_all_args_none_consts + arg_all_args_consts
        self.vars = arg_all_args
        for op in self.sub_operations:
            if isinstance(op, CompositeOperation):
                op.reorder_args(self.append_var_names_to_end)

    def get_code_lines_and_args(self) -> Tuple[List[str], List[str]]:
        """Get the variable names and arguments for this operation."""

        if (
            self.code_lines is not None
            and self.vars is not None
            and self.consts is not None
            and self.local_var_names is not None
        ):
            return self.local_var_names, self.code_lines, self.vars, self.consts

        args = []
        consts = []
        local_var_names = []
        append_var_names_to_end = []
        append_arg_names_to_end = []
        for op in self.sub_operations:
            var_name = op.output_var_name().strip()
            arg_list = []
            if isinstance(op, CompositeOperation):
                arg_local_var_names, _, arg_vars, arg_consts = op.get_code_lines_and_args()
                all_args = list(set(arg_vars + arg_consts) - set(arg_local_var_names))
                ## If the composite op is const folded, or if all its args are local vars, we can move its var to the end.
                if (
                    op.is_const_fold_composite
                    or len([l_arg for l_arg in arg_vars if l_arg not in arg_local_var_names]) == 0
                ):
                    append_var_names_to_end.append(var_name)
                    var_name = None
                    append_arg_names_to_end.extend([arg.strip() for arg in all_args if arg not in arg_consts])
                else:
                    arg_list.extend([arg.strip() for arg in all_args if arg not in arg_consts])
                consts.extend([arg.strip() for arg in arg_consts if arg not in consts])
            elif isinstance(op, TupleOp):
                arg_list.append(op.generate_code().split("=")[1].split("[")[0].strip())
            elif isinstance(op, InputOp):
                arg_list.append(op.generate_code().split("=")[1].split(".")[0].strip())
            else:
                for arg in op.args:
                    if isinstance(arg, PlaceholderTensor):
                        arg_list.append(arg.generate_code())
                    elif isinstance(arg, (list, tuple)):
                        for a in arg:
                            if isinstance(a, PlaceholderTensor):
                                arg_list.append(a.generate_code())
                            elif isinstance(a, ConstantTensor):
                                CompositeOperation.ALL_CONSTANTS[a.id] = a
                                consts.append(a.generate_code())
                    elif isinstance(arg, ConstantTensor):
                        CompositeOperation.ALL_CONSTANTS[arg.id] = arg
                        consts.append(arg.generate_code())
                    elif arg is None or isinstance(arg, (int, float, str, bool)):
                        pass
                    else:
                        print(f"Warning: Unsupported arg type {type(arg)} in operation {op}")
            if var_name is not None:
                local_var_names.append(var_name)
            if len(arg_list) > 0:
                args.extend(arg_list)

        # need to call op.generate_code() again to get the correct code line after reordering args
        code_lines = []
        for op in self.sub_operations:
            if isinstance(op, CompositeOperation):
                op.reorder_args(append_var_names_to_end)
            code_line = op.generate_code()
            code_lines.append(code_line)
        self.append_var_names_to_end = append_var_names_to_end
        local_var_names.extend(append_var_names_to_end)
        args.extend(append_arg_names_to_end)
        self.local_var_names, self.code_lines, self.vars, self.consts = local_var_names, code_lines, args, consts
        return local_var_names, code_lines, args, consts

    @property
    def fun_name(self) -> str:
        return to_valid_variable_name(self.id)

    def generate_code(self) -> str:
        """Generate PyTorch code for this operation."""
        _, fun_body = self.get_fun_body()
        var_names, code_lines, arg_names, consts = self.get_code_lines_and_args()
        args_not_in_var_names = list(dict.fromkeys(arg for arg in arg_names if arg not in var_names))
        args_not_in_var_names = list(args_not_in_var_names) + consts
        if not self.is_duplicate(fun_body) or not CompositeOperation.prune:
            result = f"{self.output_var_name()} = {self.fun_name}({','.join(args_not_in_var_names)  if len(args_not_in_var_names) > 0 else ''})"
        else:
            result = f"{self.output_var_name()} = {CompositeOperation.generated_code[fun_body]}({','.join(args_not_in_var_names) if len(args_not_in_var_names) > 0 else ''})"
        return result

    def get_fun_body(self) -> str:
        """Generate import statements for this operation."""
        # Default implementation returns an empty list, subclasses can override

        if self.fun_body is not None and self.func is not None:
            return self.func, self.fun_body

        var_names, code_lines, arg_names, consts = self.get_code_lines_and_args()
        args_not_in_var_names = list(dict.fromkeys([arg for arg in arg_names if arg not in var_names]))
        args_not_in_var_names = list(args_not_in_var_names)
        new_line = "\n    "
        orig_func = f"""def {self.fun_name}({','.join(args_not_in_var_names) + ", " if len(args_not_in_var_names) > 0 else ""}{"*args" if len(consts) > 0 else ""}):
    {new_line.join(code_lines)}
    return {self.output_var_name()} END
"""
        new_func = str(orig_func)
        if self.id == "main":
            new_func = new_func.replace("return OUTPUT END", f"return {', '.join(self.graph_outputs)} ")

        for arg_index, arg in enumerate(args_not_in_var_names + var_names):
            new_func = new_func.replace(f"({arg})", f"(var{arg_index})")
            new_func = new_func.replace(f"({arg},", f"(var{arg_index},")
            new_func = new_func.replace(f", {arg},", f", var{arg_index},")
            new_func = new_func.replace(f",{arg},", f",var{arg_index},")
            new_func = new_func.replace(f", {arg})", f", var{arg_index})")
            new_func = new_func.replace(f",{arg})", f",var{arg_index})")
            new_func = new_func.replace(f", {arg}[", f", var{arg_index}[")
            new_func = new_func.replace(f", {arg} ", f", var{arg_index}")
            new_func = new_func.replace(f" {arg})", f", var{arg_index})")
            new_func = new_func.replace(f"= {arg}[", f"= var{arg_index}[")
            new_func = new_func.replace(f"= {arg}.", f"= var{arg_index}.")
            new_func = new_func.replace(f"[{arg}]", f"[var{arg_index}]")
            new_func = new_func.replace(f"[{arg},", f"[var{arg_index},")
            new_func = new_func.replace(f",{arg}]", f",var{arg_index}]")
            new_func = new_func.replace(f", {arg}]", f", var{arg_index}]")
            new_func = new_func.replace(f"\t{arg} = ", f"\tvar{arg_index} = ")
            new_func = new_func.replace(f"    {arg} = ", f"    var{arg_index} = ")
            new_func = new_func.replace(f"return {arg} END", f"return var{arg_index}")
            new_func = new_func.replace(f"return {arg}, ", f"return var{arg_index}, ")
        for arg_index, arg in enumerate(consts):
            new_func = new_func.replace(f"({arg})", f"(args[{arg_index}])")
            new_func = new_func.replace(f"({arg},", f"(args[{arg_index}],")
            new_func = new_func.replace(f", {arg},", f", args[{arg_index}],")
            new_func = new_func.replace(f",{arg},", f",args[{arg_index}],")
            new_func = new_func.replace(f", {arg})", f", args[{arg_index}])")
            new_func = new_func.replace(f" {arg})", f" args[{arg_index}])")
            new_func = new_func.replace(f",{arg})", f",args[{arg_index}])")
            new_func = new_func.replace(f" {arg})", f" args[{arg_index}])")
            new_func = new_func.replace(f" {arg}[", f" args[{arg_index}][")
            new_func = new_func.replace(f"= {arg}[", f"= args[{arg_index}][")
            new_func = new_func.replace(f"[{arg}]", f"[args[{arg_index}]]")
            new_func = new_func.replace(f"[{arg},", f"[args[{arg_index}],")
            new_func = new_func.replace(f",{arg}]", f",args[{arg_index}]]")
            new_func = new_func.replace(f", {arg}]", f", args[{arg_index}]]")
            new_func = new_func.replace(f"\t{arg} = ", f"\targs[{arg_index}] = ")
            new_func = new_func.replace(f"    {arg} = ", f"    args[{arg_index}] = ")
            new_func = new_func.replace(f"return {arg} END", f"return args[{arg_index}]")

        fun_body = new_func.split(f"def {self.fun_name}")[1]
        args_not_in_var_names = list(dict.fromkeys(arg for arg in arg_names if arg not in var_names))
        args_not_in_var_names = list(args_not_in_var_names) + consts
        if not self.is_duplicate(fun_body):
            CompositeOperation.generated_code[fun_body] = self.fun_name
        else:
            CompositeOperation.duplicate_ops[self.fun_name] = CompositeOperation.generated_code[fun_body]
        if CompositeOperation.prune:
            for op in CompositeOperation.duplicate_ops:
                new_func = new_func.replace(f"{op}(", f"{CompositeOperation.duplicate_ops[op]}(")
        self.fun_body = fun_body
        self.func = new_func
        return new_func, fun_body

    def is_duplicate(self, fun_body: Optional[str] = None) -> bool:
        if fun_body is None:
            _, fun_body = self.get_fun_body()
        return (
            fun_body in CompositeOperation.generated_code
            and self.fun_name != CompositeOperation.generated_code[fun_body]
        )

    def generate_import_code(self) -> List[str]:
        import_code = [
            "import torch",
            "from utils import track_input_output",
            "_tensor_io_log = []",
        ]
        new_func, fun_body = self.get_fun_body()
        for op in self.sub_operations:
            import_code.extend(op.generate_import_code())
        if not self.is_duplicate():
            CompositeOperation.generated_code[fun_body] = self.fun_name
            import_code.append("@track_input_output(_tensor_io_log=_tensor_io_log)\n" + new_func)
        else:
            CompositeOperation.duplicate_ops[self.fun_name] = CompositeOperation.generated_code[fun_body]
            if not CompositeOperation.prune:
                import_code.append(f"{self.fun_name} = {CompositeOperation.generated_code[fun_body]}")
        return import_code


class PatternObjFactory:
    WRAPPED_OPERATION_PATTERN_OBJ_REGISTRY = {}

    @staticmethod
    def init_wrapped_operation_pattern_obj_registry():
        if PatternObjFactory.WRAPPED_OPERATION_PATTERN_OBJ_REGISTRY:
            return
        for k, v in WRAPPED_OPERATION_REGISTRY.items():
            PatternObjFactory.WRAPPED_OPERATION_PATTERN_OBJ_REGISTRY[
                v.__name__ + "PatternObj"
            ] = WrappedOpPatternObjBaseMeta(v.__name__ + "PatternObj", (), {"wrapped_op": v})
            setattr(
                PatternObjFactory,
                f"{v.__name__ }PatternObj",
                PatternObjFactory.WRAPPED_OPERATION_PATTERN_OBJ_REGISTRY[v.__name__ + "PatternObj"],
            )
        PatternObjFactory.WRAPPED_OPERATION_PATTERN_OBJ_REGISTRY["TupleOpPatternObj"] = WrappedOpPatternObjBaseMeta(
            "TupleOpPatternObj", (), {"wrapped_op": TupleOp}
        )
        PatternObjFactory.TupleOpPatternObj = PatternObjFactory.WRAPPED_OPERATION_PATTERN_OBJ_REGISTRY[
            "TupleOpPatternObj"
        ]
        PatternObjFactory.WildcardPatternObj = WrappedOpPatternObjBaseMeta("WildcardPatternObj", (), {})
        PatternObjFactory.WRAPPED_OPERATION_PATTERN_OBJ_REGISTRY[
            "CompositeOperationPatternObj"
        ] = WrappedOpPatternObjBaseMeta("CompositeOperationPatternObj", (), {"wrapped_op": CompositeOperation})
        PatternObjFactory.CompositeOperationPatternObj = PatternObjFactory.WRAPPED_OPERATION_PATTERN_OBJ_REGISTRY[
            "CompositeOperationPatternObj"
        ]

    @staticmethod
    def get_pattern_obj_from(wrapped_op: WrappedOperation) -> WrappedOpPatternObj:
        assert (
            wrapped_op.__class__.__name__ + "PatternObj" in PatternObjFactory.WRAPPED_OPERATION_PATTERN_OBJ_REGISTRY
        ), f"Could not find op {wrapped_op.__class__.__name__} in pattern object registry. This should never happen."
        return PatternObjFactory.WRAPPED_OPERATION_PATTERN_OBJ_REGISTRY[wrapped_op.__class__.__name__ + "PatternObj"]


PatternObjFactory.init_wrapped_operation_pattern_obj_registry()


def hash_dict(d: Dict[str, Any]) -> str:
    """Stable hash of a dict (e.g. node or edge attributes)."""
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def merge_nodes_as_composite(
    operation_graph: OperationGraph,
    main_output: Optional[str],
    ancestors: List[str],
    prefix="COMPOSITE",
    keep_existing_name=False,
) -> Optional[CompositeOperation]:
    """
    Merge a set of nodes into a CompositeOperation node in the graph G.
    Removes all edges between nodes_to_merge and only keeps edges to/from outside nodes.
    """
    # 1. Find all incoming edges from outside and outgoing edges to outside
    incoming = set()
    outgoing = set()
    G = operation_graph.graph
    nodes_to_merge = [i for i in ancestors]
    if main_output is not None:
        nodes_to_merge = [main_output] + nodes_to_merge
    nodes_to_merge_set = set(nodes_to_merge)
    try:
        for n in nodes_to_merge_set:
            for pred in G.predecessors(n):
                if pred not in nodes_to_merge_set:
                    incoming.add((pred, n))
            for succ in G.successors(n):
                if succ not in nodes_to_merge_set:
                    outgoing.add((n, succ))
    except:
        # If any node in nodes_to_merge is not in G, we cannot merge. Continue iteration
        return None

    H = G.subgraph(nodes_to_merge)
    # Get the topological order for the subset
    order = list(nx.topological_sort(H))
    if main_output is None and len(order):
        main_output = order[-1]
    if len(set([out_edge[0] for out_edge in outgoing if out_edge[0] != main_output])) != 0:
        return None
    for node in nodes_to_merge:
        if node not in G.nodes:
            return None
        elif (
            isinstance(G.nodes[node]["operation"].graph_output_indices, list)
            and len(G.nodes[node]["operation"].graph_output_indices) != 0
            and node != main_output
        ):
            return None
    if len(ancestors) == 0 and isinstance(G.nodes[main_output]["operation"], CompositeOperation):
        return None
    if keep_existing_name:
        prefix = G.nodes[main_output]["operation"].unique_name + "_" + prefix
    composite_node = f"{prefix}_{CompositeOperation.counter}"
    assert composite_node not in G.nodes, f"FOUND NODE {composite_node} ALREADY IN GRAPH, CANNOT MERGE"
    composite_op = CompositeOperation(
        id=composite_node,
        unique_name=G.nodes[main_output]["operation"].unique_name,
        sub_operations=[G.nodes[child]["operation"] for child in order],
        function_call_name="composite",
        args=[],
        kwargs={},
        meta_data=G.nodes[main_output]["operation"].meta_data,
        graph_output_indices=G.nodes[main_output]["operation"].graph_output_indices,
    )
    for edge in incoming.union(outgoing):
        G.remove_edge(*edge)
    for n in nodes_to_merge_set:
        G.remove_node(n)
    G.add_node(composite_node, operation=composite_op)

    # 4. Add edges from outside predecessors to composite node
    for child in order:
        for pred, orig_child in incoming:
            if pred not in nodes_to_merge or child != orig_child:
                G.add_edge(pred, composite_node)
    # 5. Add edges from composite node to outside successors
    for _, succ in outgoing:
        if succ not in nodes_to_merge:
            G.add_edge(composite_node, succ)

    return composite_op


def combine_conv_bn_relu(operation_graph: OperationGraph):
    G = operation_graph.graph
    changed = True
    while changed:
        changed = False
        for node in list(nx.topological_sort(G))[::-1]:
            op = G.nodes[node]["operation"]
            if isinstance(op, (AtenReluInplace, AtenRelu)):
                predecessors = list(G.predecessors(node))
                if len(predecessors) != 1:
                    continue
                pred_node = predecessors[0]
                pred_op = G.nodes[pred_node]["operation"]
                if isinstance(pred_op, TupleOp):
                    predecessors = list(G.predecessors(pred_node))
                    if len(predecessors) != 1:
                        continue
                    pred_node2 = predecessors[0]
                    pred_op2 = G.nodes[pred_node2]["operation"]
                    if isinstance(pred_op2, AtenNativeBatchNorm):
                        bn_predecessors = list(G.predecessors(pred_node2))
                        if len(bn_predecessors) != 1:
                            continue
                        bn_pred_node = bn_predecessors[0]
                        if bn_pred_node not in G.nodes:
                            continue
                        bn_pred_op = G.nodes[bn_pred_node]["operation"]
                        if isinstance(bn_pred_op, AtenConvolution):
                            # We have a conv -> bn -> relu pattern, merge them
                            composite_op = merge_nodes_as_composite(
                                operation_graph,
                                main_output=node,
                                ancestors=[pred_node, pred_node2, bn_pred_node],
                                keep_existing_name=True,
                            )
                            if composite_op is not None:
                                changed = True
                                break
            elif isinstance(op, TupleOp):
                predecessors = list(G.predecessors(node))
                if len(predecessors) != 1:
                    continue
                pred_node2 = predecessors[0]
                pred_op2 = G.nodes[pred_node2]["operation"]
                if isinstance(pred_op2, AtenNativeBatchNorm):
                    bn_predecessors = list(G.predecessors(pred_node2))
                    if len(bn_predecessors) != 1:
                        continue
                    bn_pred_node = bn_predecessors[0]
                    if bn_pred_node not in G.nodes:
                        continue
                    bn_pred_op = G.nodes[bn_pred_node]["operation"]
                    if isinstance(bn_pred_op, AtenConvolution):
                        # We have a conv -> bn -> relu pattern, merge them
                        composite_op = merge_nodes_as_composite(
                            operation_graph,
                            main_output=node,
                            ancestors=[pred_node2, bn_pred_node],
                            keep_existing_name=True,
                        )
                        if composite_op is not None:
                            changed = True
                            break
    return operation_graph


def make_every_node_composite(operation_graph: OperationGraph):
    G = operation_graph.graph
    for node in list(G.nodes):
        op = G.nodes[node]["operation"]
        if not isinstance(op, (CompositeOperation, InputOp)):
            composite_op = merge_nodes_as_composite(
                operation_graph, main_output=node, ancestors=[], keep_existing_name=True
            )
            if composite_op is not None:
                G.nodes[composite_op.id]["operation"] = composite_op
    return operation_graph


def combine_tuple_get_item_scaled_attention_tuple_get_item(operation_graph: OperationGraph):
    G = operation_graph.graph
    changed = True
    while changed:
        changed = False
        for node in list(nx.topological_sort(G))[::-1]:
            op = G.nodes[node]["operation"]
            if isinstance(op, TupleOp):
                predecessors = list(G.predecessors(node))
                if len(predecessors) != 1:
                    continue
                pred_node = predecessors[0]
                pred_op = G.nodes[pred_node]["operation"]
                if isinstance(pred_op, AtenScaledDotProductFlashAttention):
                    predecessors = list(G.predecessors(pred_node))
                    if len(predecessors) != 3:
                        continue
                    pred_node2 = predecessors[0]
                    pred_op2 = G.nodes[pred_node2]["operation"]
                    pred_node3 = predecessors[1]
                    pred_op3 = G.nodes[pred_node3]["operation"]
                    pred_node4 = predecessors[2]
                    pred_op4 = G.nodes[pred_node4]["operation"]
                    if (
                        isinstance(pred_op2, TupleOp)
                        and isinstance(pred_op3, TupleOp)
                        and isinstance(pred_op4, TupleOp)
                    ):
                        composite_op = merge_nodes_as_composite(
                            operation_graph, main_output=node, ancestors=[pred_node, pred_node2, pred_node3, pred_node4]
                        )
                        if composite_op is not None:
                            changed = True
                            break
    return operation_graph


def get_predecessor_ordering(G, op, predecessors):
    op_args = [arg for arg in op.args if isinstance(arg, PlaceholderTensor)]
    op_args_list = [arg for arg in op.args if isinstance(arg, (list, tuple))]
    op_args_list_flatten = []
    for arg_list in op_args_list:
        place_holders = [a for a in arg_list if isinstance(a, PlaceholderTensor)]
        if len(place_holders) == 0:
            continue
        if len(op_args_list_flatten):
            print(f"Found multiple list/tuple args in operation {op}. This may cause issues in pattern matching.")
        op_args_list_flatten.extend(place_holders)
    if len(op_args) > 0 and len(op_args_list_flatten) > 0:
        print(
            f"Found both single PlaceholderTensor args and list/tuple args in operation {op}. This may cause issues in pattern matching."
        )
        return predecessors
    elif len(op_args_list_flatten) > 0:
        op_args = op_args_list_flatten
    actual_args = []
    for arg in op_args:
        try:
            actual_args.append(arg.name)
        except:
            continue
    args = {}
    for pred in predecessors:
        try:
            if isinstance(G.nodes[pred]["operation"], CompositeOperation):
                args[G.nodes[pred]["operation"].unique_name] = pred
            else:
                res = G.nodes[pred]["operation"].meta_data.res
                if isinstance(res, (list, tuple)):
                    res = [r for r in res if isinstance(r, PlaceholderTensor)]
                    for r in res:
                        args[r.name] = pred
                else:
                    args[res.name] = pred
        except Exception as e:
            continue
    result = []
    for i in actual_args:
        if i in args:
            result.append(args[i])
    if len(predecessors) == len(result):
        return result
    return predecessors


def no_intersection_between_patterns(custom_patterns):
    all_nodes = {}
    for pattern_idx, pattern in enumerate(custom_patterns):
        queue = [pattern]
        nodes = set()
        while queue:
            pat = queue[0]
            queue = queue[1:]
            if pat in nodes:
                continue
            nodes.add(pat)
            for par in pat.parents:
                queue.append(par)
        all_nodes[pattern_idx] = (pattern, nodes)
    for i in all_nodes:
        for j in all_nodes:
            if i >= j:
                continue
            inter = all_nodes[i][1].intersection(all_nodes[j][1])
            if len(inter) > 0:
                print(
                    f"Patterns {i} and {j} have intersection nodes {inter}. This is not allowed. Each pattern must be disjoint."
                )
                return False
    return True


def combine_custom_patterns(operation_graph, custom_patterns):
    G = operation_graph.graph
    assert no_intersection_between_patterns(custom_patterns), "Custom patterns cannot have intersection between them."
    for pattern_idx, pattern in enumerate(custom_patterns):
        assert isinstance(
            pattern, WrappedOpPatternObj
        ), f"Patterns can only be represented by Pattern Objects. Got {pattern}"
        changed = True
        while changed:
            changed = False
            for node in list(nx.topological_sort(G))[::-1]:
                queue = [(pattern, node)]
                ancestors = [node]
                passed = True
                while queue and passed:
                    pat, curr_op = queue[0]
                    queue = queue[1:]
                    op = G.nodes[curr_op]["operation"]
                    if pat.compareWith(op):
                        predecessors = list(G.predecessors(curr_op))
                        predecessors = get_predecessor_ordering(G, op, predecessors)
                        if len(pat.parents) == 0:
                            continue
                        if len(pat.parents) != len(predecessors):
                            passed = False
                            print(
                                f"Pattern {pattern_idx}: Pattern {pat} parent count {len(pat.parents)} does not match predecessor count {len(predecessors)}. Found node {op.unique_name}->{curr_op} that has {len(predecessors)}"
                            )
                            continue
                        for par_pat, par_node in zip(pat.parents, predecessors):
                            if par_pat.__class__.__name__ == "WildcardPatternObj":
                                continue
                            queue.append((par_pat, par_node))
                            ancestors.append(par_node)
                    else:
                        passed = False
                if passed:
                    composite_op = merge_nodes_as_composite(
                        operation_graph, main_output=ancestors[0], ancestors=ancestors[1:]
                    )
                    if composite_op is not None:
                        changed = True
                        print(f"Merged {node} and {set(ancestors[1:])} as composite operation {composite_op.id}")
                        break
                    else:
                        print(f"COULD NOT MERGE {set(ancestors)} AS COMPOSITE OPERATION")

    return operation_graph


def combine_constants(operation_graph):
    G = operation_graph.graph
    nodes_with_only_constant_predecessors = set()
    for node in list(nx.topological_sort(G)):
        op = G.nodes[node]["operation"]
        predecessors = list(G.predecessors(node))
        if isinstance(op, InputOp):
            continue
        if len(predecessors) == 0:
            nodes_with_only_constant_predecessors.add(node)
        elif len([pred for pred in predecessors if pred not in nodes_with_only_constant_predecessors]) == 0:
            nodes_with_only_constant_predecessors.add(node)

    def get_leaf_nodes(list_of_nodes):
        leaf_nodes = set()
        for node in list_of_nodes:
            successors = list(G.successors(node))
            if len([suc for suc in successors if suc in list_of_nodes]) == 0:
                leaf_nodes.add(node)
        return leaf_nodes

    composites = {leaf: set() for leaf in get_leaf_nodes(nodes_with_only_constant_predecessors)}
    for node in composites:
        ancestors = nx.ancestors(G, node)
        composites[node] = ancestors

    # Make sure no two composite sets intersect. If they do, create new composite sets of the intersection and remove those nodes from the original sets.
    # composites will contain all the nodes that can be merged into a composite operation, with the key being the main output node.
    change = True
    while change:
        change = False
        for suc1 in list(composites):
            for suc2 in list(composites):
                if suc1 == suc2:
                    continue
                inter = composites[suc1].intersection(composites[suc2])
                if len(inter) > 0:
                    intersection_leaf_nodes = get_leaf_nodes(inter)
                    composites[suc1] = composites[suc1] - inter
                    composites[suc2] = composites[suc2] - inter
                    for inter_leaf in intersection_leaf_nodes:
                        inter.remove(inter_leaf)
                    for inter_leaf in intersection_leaf_nodes:
                        composites[inter_leaf] = inter
                        print(f"Created new composite set for intersection nodes {inter} with leaf node {inter_leaf}")
                    change = True
                if change:
                    break
            if change:
                break
    for suc, nodes in composites.items():
        composite_op = merge_nodes_as_composite(operation_graph, main_output=suc, ancestors=list(nodes))
        if composite_op is not None:
            composite_op.is_const_fold_composite = True
            print(f"Merged constant nodes {nodes} into {suc} as composite operation {composite_op.id}")
    return operation_graph


def combine_ops(
    reverse_index: Dict[str, List[Tuple[str, List[str]]]],
    operation_graph: OperationGraph,
    old_fingerprints: Dict[str, str],
    composite_ops: Dict[str, CompositeOperation],
) -> bool:
    """
    Combine operations in the graph by merging nodes with the same fingerprint.
    This function modifies the graph in place.
    """
    changed = False
    hits_per_fp = {fp: len(nodes) for fp, nodes in reverse_index.items() if len(nodes) > 1}
    for fp in sorted(hits_per_fp, key=hits_per_fp.get, reverse=True):
        for nodes in reverse_index[fp]:
            if nodes[0] not in old_fingerprints or old_fingerprints[nodes[0]] != fp:
                changed = True
                old_fingerprints[nodes[0]] = fp
                composite_op = merge_nodes_as_composite(
                    operation_graph,
                    nodes[0],
                    nodes[1],
                    keep_existing_name=True,
                )
                if composite_op is not None:
                    composite_ops[composite_op.id] = composite_op
    return changed


def compute_subgraph_fingerprints(
    operation_graph: OperationGraph, composite_ops: Dict[str, CompositeOperation]
) -> Tuple[Dict[str, str], Dict[int, Dict[str, str]]]:
    G = operation_graph.graph
    old_fingerprints: Dict[str, str] = {}
    iteration_result: Dict[int, Dict[str, str]] = {}

    for i in range(len(G.nodes)):
        changed = False
        fingerprints: Dict[str, Tuple[str, List[str]]] = {}
        for node in list(nx.topological_sort(G))[::-1]:
            parent_representation = G.nodes[node]["operation"].get_unique_representation()
            node_attr_hash = hash_dict(parent_representation)
            ancestors = list(G.predecessors(node))
            ancestors_representations = [
                G.nodes[ancestor]["operation"].get_unique_representation() for ancestor in ancestors
            ]
            ancestor_hashes = [
                old_fingerprints.get(ancestor, hash_dict(representation))
                for ancestor, representation in zip(ancestors, ancestors_representations)
            ]
            combined = (node_attr_hash, ancestor_hashes)
            combined_str = json.dumps(combined)
            fingerprint = hashlib.sha256(combined_str.encode()).hexdigest()
            fingerprints[node] = (fingerprint, ancestors)
        reverse_index: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
        for node, fp in fingerprints.items():
            reverse_index[fp[0]].append((node, fp[1]))
        changed = combine_ops(reverse_index, operation_graph, old_fingerprints, composite_ops)

        iteration_result[i] = old_fingerprints
        if not changed:
            # No new fingerprints found, we can stop
            break
    return old_fingerprints, iteration_result


def dump_graph_patterns(operation_graph: OperationGraph, file_path: str):
    G = operation_graph.graph
    text = []
    for node in list(nx.topological_sort(G))[::-1]:
        op = G.nodes[node]["operation"]
        parents = list(G.predecessors(node))
        parents = get_predecessor_ordering(G, op, parents)
        if isinstance(op, (WrappedOperation, TupleOp, CompositeOperation)):
            op_po = PatternObjFactory.get_pattern_obj_from(op)
            parent_unique_names = [
                (
                    to_valid_variable_name(G.nodes[parent]["operation"].unique_name)
                    if isinstance(G.nodes[parent]["operation"], (WrappedOperation, TupleOp, CompositeOperation))
                    else "POFactory.WildcardPatternObj()"
                )
                for parent in parents
            ]
            text.append(
                f"{to_valid_variable_name(op.unique_name)} = POFactory.{op_po.__name__}({', '.join([p for p in parent_unique_names])})"
            )
    text.append("from find_repeated_subgraphs import PatternObjFactory as POFactory")
    text = text[::-1]
    with open(file_path, "w") as f:
        f.write("\n".join(text))
    print(f"Generated graph pattern code dumped to {os.path.abspath(file_path)}")


def find_repeated_subgraphs(
    operation_graph: OperationGraph, custom_patterns: Optional[List[WrappedOpPatternObj]] = None
):
    orig_config = ConstantTensor.ConstantTensorFromModel
    ConstantTensor.ConstantTensorFromModel = True
    print("Setting ConstantTensor.ConstantTensorFromModel to True")
    composite_ops: Dict[str, CompositeOperation] = {}
    new_operation_graph = OperationGraph.from_operation_graph(operation_graph)
    combine_constants(new_operation_graph)
    if custom_patterns is not None:
        combine_custom_patterns(new_operation_graph, custom_patterns)
        json_structure = create_graph_json_structure(new_operation_graph)
        # Write the JSON structure to the specified output file
        with open("intermediate_combined_operation_graph_viz.json", "w") as f:
            json.dump(json_structure, f, indent=2)
            print(
                f"Dumped visualization to intermediate_combined_operation_graph_viz.json. Load it into netron.app to visualize the model."
            )
    dump_graph_patterns(new_operation_graph, "graph_patterns.py")
    combine_conv_bn_relu(new_operation_graph)
    combine_tuple_get_item_scaled_attention_tuple_get_item(new_operation_graph)
    compute_subgraph_fingerprints(new_operation_graph, composite_ops)
    make_every_node_composite(new_operation_graph)
    json_structure = create_graph_json_structure(new_operation_graph)
    # Write the JSON structure to the specified output file
    with open("combined_operation_graph_viz.json", "w") as f:
        json.dump(json_structure, f, indent=2)
        print(
            f"Dumped visualization to combined_operation_graph_viz.json. Load it into netron.app to visualize the model."
        )
    ConstantTensor.ConstantTensorFromModel = orig_config
    return new_operation_graph, composite_ops


def make_tree():
    """Helper to create recursive dict structure."""
    return {"nodes": [], "children": {}}


def group_nodes_hierarchical(node_map):
    """
    Build hierarchical grouping of nodes.

    Args:
        node_map (dict): {node_id: path}

    Returns:
        dict: nested structure of groups
    """
    root = {}

    for node_id, path in node_map.items():
        # Extract counter
        if "/" in path:
            base, counter = path.rsplit("/", 1)
        else:
            base, counter = path, "0"

        # Parent path (drop last ".suffix" if present)
        if "." in base:
            parent_base = base.rsplit(".", 1)[0]
        else:
            parent_base = None

        # Current group key
        group_key = f"{parent_base or base} -> {counter}"

        # Traverse down the hierarchy
        current = root
        segments = (parent_base or base).split(".")
        full_path = []
        for seg in segments:
            full_path.append(seg)
            key = ".".join(full_path) + f" -> {counter}"
            if key not in current:
                current[key] = make_tree()
            current = current[key]["children"]

        # Place node here
        if group_key not in current:
            current[group_key] = make_tree()
        current[group_key]["nodes"].append(node_id)

    return root


def get_most_nested_children(tree):
    """
    Recursively find the most nested children (deepest leaves) from the hierarchical tree.
    """
    result = {}

    def dfs(node, path):
        # If node has no children, it's a leaf -> add to result
        if not node.get("children"):
            result[path] = node["nodes"]
        else:
            for child_name, child_node in node["children"].items():
                dfs(child_node, child_name)

    for group_name, group_node in tree.items():
        dfs(group_node, group_name)

    return result


def merge_nodes(operation_graph, ancestors, group_name):
    G = operation_graph.graph
    composite_op = merge_nodes_as_composite(
        operation_graph,
        main_output=None,
        ancestors=ancestors,
        prefix="_".join(group_name.split(" -> ")) + "_COMPOSITE",
    )
    if composite_op is not None:
        G.nodes[composite_op.id]["operation"] = composite_op
        return [composite_op.id]
    return ancestors


def merge_nested_tree(tree, operation_graph):
    all_nodes = []
    for group_name, group_node in tree.items():
        if len(group_node["children"]) > 0:
            assert group_node["nodes"] == [], "Group node should not have both children and nodes"
            nodes = []
            for child_name, child_node in group_node["children"].items():
                nodes.extend(merge_nested_tree({child_name: child_node}, operation_graph))
            group_node["nodes"] = list(set(nodes))
        if len(group_node["nodes"]) > 1:
            all_nodes.extend(merge_nodes(operation_graph, group_node["nodes"], group_name))
        elif len(group_node["nodes"]) == 1:
            all_nodes.extend(group_node["nodes"])
    return list(set(all_nodes))


def unloop_composites_with_less_than_n_nodes(new_operation_graph, n=2):
    G = new_operation_graph.graph
    changed = True
    while changed:
        changed = False
        for node in list(nx.topological_sort(G))[::-1]:
            op = G.nodes[node]["operation"]
            if isinstance(op, CompositeOperation):
                found_parent_lead_child = False
                while not found_parent_lead_child:
                    found_parent_lead_child = True
                    for sub_op in op.sub_operations:
                        if isinstance(sub_op, CompositeOperation) and any(
                            isinstance(child_op, CompositeOperation) for child_op in sub_op.sub_operations
                        ):
                            found_parent_lead_child = False
                            op = sub_op
                            break
                for index, sub_op in enumerate(op.sub_operations):
                    if (
                        isinstance(sub_op, CompositeOperation)
                        and len(sub_op.sub_operations) + len(op.sub_operations) <= n
                    ):
                        op.sub_operations = (
                            op.sub_operations[:index] + sub_op.sub_operations + op.sub_operations[index + 1 :]
                        )
                        changed = True
                        break
            if changed:
                break
    return new_operation_graph


def trace_model_structure(
    model,
    input_shapes,
    input_dtypes=None,
    dump_visualization=False,
    wrap_operations=True,
    save_original_tensors=True,
    track_params=True,
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

    tracer = TracerData(save_original_tensors=save_original_tensors)
    if track_params:
        state_dict = model.state_dict()
        Trackable_Tensor.set_tracer_data(tracer)  # Set the tracer data instance for Trackable_Tensor
        wrapped_state_dict = wrap_state_dict(state_dict)
        model.load_state_dict(wrapped_state_dict, assign=True)
    else:
        Trackable_Tensor.set_tracer_data(tracer)  # Set the tracer data instance for Trackable_Tensor
    input_tensors = get_input_tensors(input_shapes, input_dtypes)
    module_calls = {}
    handles = register_module_hooks(model, module_calls=module_calls)
    outputs = model(*input_tensors)
    set_is_graph_output(outputs)
    # remove the hooks after tracing
    for handle in handles:
        handle.remove()

    tracer.finalize_graph()
    operation_graph = generate_graph_and_visualize(tracer, dump_visualization, wrap_operations)

    id_to_module = {}
    for op in operation_graph.graph.nodes:
        operation = operation_graph.graph.nodes[op]["operation"]
        id_to_module[op] = operation.unique_name
    tree = group_nodes_hierarchical(id_to_module)
    with open("tree.json", "w") as f:
        json.dump(tree, f, indent=2)
    combine_constants(operation_graph)
    combine_conv_bn_relu(operation_graph)
    merge_nested_tree(tree, operation_graph)
    composite_ops: Dict[str, CompositeOperation] = {}
    new_operation_graph = OperationGraph.from_operation_graph(operation_graph)
    try:
        compute_subgraph_fingerprints(operation_graph, composite_ops)
        operation_graph = new_operation_graph
    except:
        pass
    make_every_node_composite(new_operation_graph)
    unloop_composites_with_less_than_n_nodes(new_operation_graph, n=15)
    json_structure = create_graph_json_structure(operation_graph)
    # Write the JSON structure to the specified output file
    with open("combined_operation_graph_viz.json", "w") as f:
        json.dump(json_structure, f, indent=2)
        print(
            f"Dumped visualization to combined_operation_graph_viz.json. Load it into netron.app to visualize the model."
        )
    return operation_graph
