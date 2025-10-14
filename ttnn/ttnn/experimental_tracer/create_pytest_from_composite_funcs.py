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


def get_test_params_from_tensor_summary(tensor_summary):
    if tensor_summary["summary"]["type"] == "dynamic_quantiles_stats":
        return (
            tensor_summary["shape"],
            tensor_summary["dtype"],
            tensor_summary["summary"]["type"],
            [
                tensor_summary["summary"]["quantiles"],
                (
                    tensor_summary["summary"]["mean"],
                    tensor_summary["summary"]["std"],
                    tensor_summary["summary"]["norm"],
                ),
            ],
        )
    elif tensor_summary["summary"]["type"] == "state_dict_constant":
        return (
            tensor_summary["shape"],
            tensor_summary["dtype"],
            tensor_summary["summary"]["type"],
            tensor_summary["summary"]["name"],
        )
    raise ValueError(f"Unknown tensor summary type: {tensor_summary['summary']['type']}")


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
        f.write("import numpy as np\n")
        f.write("from utils import get_tensors_from_input_spec\n")
        f.write("from tests.ttnn.utils_for_testing import assert_with_pcc\n\n\n")
        f.write("def download_original_model_state_dict():\n")
        f.write("    # Implement this function to download and return the original model parameters if needed\n")
        f.write("    return None  # return original_model.state_dict()\n")
        f.write("STATE_DICT = download_original_model_state_dict()\n")

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
            if "torch." not in ref_sources[func_name] and "main" not in func_name:
                continue
            # parametrize uses single argument: input_specs
            param_values = []
            for inputs in input_lists:
                single_case = []
                for inp in inputs:
                    if isinstance(inp, (list, tuple)):
                        single_case.append(
                            [
                                get_test_params_from_tensor_summary(nested_inp)
                                for nested_inp in inp
                                if isinstance(nested_inp, dict)
                            ]
                        )
                    else:
                        single_case.append(get_test_params_from_tensor_summary(inp))
                param_values.append(single_case)

            param_str = ",\n    ".join([str(v) for v in param_values])

            # Build test function body
            test_body = textwrap.dedent(
                f"""
                torch.manual_seed(0)
                tensors = get_tensors_from_input_spec(input_specs, STATE_DICT)
                out_ref = ref_{func_name}(*tensors)
                tensors_tt = [ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device) for t in tensors]
                out_opt = ttnn_{func_name}(*tensors_tt)
                out_opt = ttnn.to_torch(out_opt)
                diff = torch.abs(out_ref - out_opt)
                print(diff.mean(), diff.std(), diff.max(), diff.min())
                print(out_ref.mean(), out_ref.std(), out_ref.max(), out_ref.min())
                print(out_opt.mean(), out_opt.std(), out_opt.max(), out_opt.min())
                assert_with_pcc(out_ref, out_opt, 0.999)
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
    print(f"Generated test file: {output_file}")


# ---------- CLI ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pytest tests from JSON and two Python modules")
    parser.add_argument("file1", help="Reference module file")
    parser.add_argument("file2", help="Optimized module file")
    parser.add_argument("json_file", help="JSON file with inputs")
    parser.add_argument("--out", "-o", default="test_generated.py", help="Output pytest file")
    args = parser.parse_args()

    generate_test_file(args.json_file, args.file1, args.file2, args.out)
