import ast
import json
import textwrap
import argparse
from collections import defaultdict
from pytorch_graph_utils import format_file_with_black

# ---------- AST utilities ----------


def strip_decorators(source):
    """Remove all decorators from function definitions using AST."""
    tree = ast.parse(source)

    class DecoratorStripper(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.decorator_list = []
            self.generic_visit(node)
            return node

        def visit_AsyncFunctionDef(self, node):
            node.decorator_list = []
            self.generic_visit(node)
            return node

    tree = DecoratorStripper().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class FunctionRenamer(ast.NodeTransformer):
    """Rename function calls inside a function body according to rename_map."""

    def __init__(self, rename_map):
        self.rename_map = rename_map

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self.rename_map:
            node.func.id = self.rename_map[node.func.id]
        self.generic_visit(node)
        return node


def rename_function_calls(source, rename_map):
    tree = ast.parse(source)
    tree = FunctionRenamer(rename_map).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# ---------- Function extraction ----------


def extract_functions(filepath, func_names):
    """Extract source code of specific functions and remove decorators."""
    with open(filepath, "r") as f:
        source = f.read()

    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    extracted = {}
    for fn in func_names:
        if fn not in functions:
            raise ValueError(f"Function {fn} not found in {filepath}")
        node = functions[fn]
        src = strip_decorators(ast.unparse(node))
        extracted[fn] = src
    return extracted


# ---------- Test file generation ----------


def generate_test_file(json_file, ref_file, ttnn_file, output_file="test_generated.py"):
    # Load JSON (list of test cases)
    with open(json_file, "r") as f:
        test_cases = json.load(f)

    # Group inputs by function name
    func_to_inputs = defaultdict(list)
    for case in test_cases:
        func_to_inputs[case["function"]].append(case["inputs"])

    all_funcs = list(func_to_inputs.keys())

    # Extract function sources
    ref_sources = extract_functions(ref_file, all_funcs)
    ttnn_sources = extract_functions(ttnn_file, all_funcs)

    # Build rename maps
    ref_rename_map = {name: f"ref_{name}" for name in ref_sources.keys()}
    ttnn_rename_map = {name: f"ttnn_{name}" for name in ttnn_sources.keys()}

    with open(output_file, "w") as f:
        f.write("import pytest\n")
        f.write("import torch\n")
        f.write("import ttnn\n")
        f.write("from utils import get_tensors_from_input_spec\n")
        f.write("from tests.ttnn.utils_for_testing import assert_with_pcc\n\n\n")

        # ---------- Add helper function ----------
        ######### End

        # Write ref functions
        for name, src in ref_sources.items():
            src = rename_function_calls(src, ref_rename_map)
            f.write("\n" + src.replace(f"def {name}(", f"def ref_{name}(") + "\n")

        # Write opt functions
        for name, src in ttnn_sources.items():
            src = rename_function_calls(src, ttnn_rename_map)
            f.write("\n" + src.replace(f"def {name}(", f"def ttnn_{name}(") + "\n")

        # Generate pytest test functions
        for func_name, input_lists in func_to_inputs.items():
            if func_name not in ref_sources:
                continue
            if "torch." not in ref_sources[func_name]:
                continue
            # parametrize uses single argument: input_specs
            param_values = []
            for inputs in input_lists:
                single_case = []
                for inp in inputs:
                    if isinstance(inp, (list, tuple)):
                        single_case.append(
                            [
                                (nested_inp["shape"], nested_inp["dtype"], nested_inp["min_max"])
                                for nested_inp in inp
                                if isinstance(nested_inp, dict)
                            ]
                        )
                    else:
                        single_case.append((inp["shape"], inp["dtype"], inp["min_max"]))
                param_values.append(single_case)

            param_str = ",\n    ".join([str(v) for v in param_values])

            # Build test function body
            test_body = textwrap.dedent(
                f"""
                torch.manual_seed(0)
                tensors = get_tensors_from_input_spec(input_specs)
                out_ref = ref_{func_name}(*tensors)
                tensors_tt = [ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device) for t in tensors]
                out_opt = ttnn_{func_name}(*tensors_tt)
                out_opt = ttnn.to_torch(out_opt)
                assert_with_pcc(out_ref, out_opt, 0.9999)
                """
            )
            test_body = textwrap.indent(test_body, "    ")

            # Write the parametrize test
            f.write(
                textwrap.dedent(
                    f"""
@pytest.mark.parametrize("input_specs", [
    {param_str}
])
def test_{func_name}(device, input_specs):
                """
                )
            )
            f.write(test_body + "\n\n")
    format_file_with_black(args.out)


# ---------- CLI ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pytest tests from JSON and two Python modules")
    parser.add_argument("file1", help="Reference module file")
    parser.add_argument("file2", help="Optimized module file")
    parser.add_argument("json_file", help="JSON file with inputs")
    parser.add_argument("--out", "-o", default="test_generated.py", help="Output pytest file")
    args = parser.parse_args()

    generate_test_file(args.json_file, args.file1, args.file2, args.out)
