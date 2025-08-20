import ast
import sys
import astunparse
from copy import deepcopy
from pytorch_graph_utils import format_file_with_black


def is_composite_func(name):
    return name.startswith("COMPOSITE_")


def get_func_defs(tree):
    """Return a dict of function name -> ast.FunctionDef for all top-level functions."""
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def inline_composite_calls(node, func_defs, var_counter=None):
    """
    Recursively inline composite function calls in the given AST node.
    Returns a list of AST statements.
    """
    if var_counter is None:
        var_counter = [0]

    def unique_var(base):
        var_counter[0] += 1
        return f"{base}_inl{var_counter[0]}"

    class Inliner(ast.NodeTransformer):
        def visit_Assign(self, assign_node):
            # Only handle simple assignments of the form: var = COMPOSITE_*(...)
            if (
                isinstance(assign_node.value, ast.Call)
                and isinstance(assign_node.value.func, ast.Name)
                and is_composite_func(assign_node.value.func.id)
            ):
                func_name = assign_node.value.func.id
                func_def = func_defs[func_name]
                # Map arguments to parameter names
                arg_map = {}
                num_named = len(func_def.args.args)
                positional_args = assign_node.value.args[num_named:]
                for idx, arg in enumerate(assign_node.value.args):
                    if idx < num_named:
                        param = func_def.args.args[idx].arg
                        arg_map[param] = arg
                # Handle *args
                if func_def.args.vararg:
                    arg_map[func_def.args.vararg.arg] = positional_args

                # Inline the function body
                stmts = []
                local_var_map = {}

                class ParamReplacer(ast.NodeTransformer):
                    def visit_Name(self, node):
                        if node.id in arg_map and not isinstance(arg_map[node.id], list):
                            value = arg_map[node.id]
                            if isinstance(value, ast.AST):
                                return deepcopy(value)
                            elif isinstance(value, str):
                                return ast.copy_location(ast.Name(id=value, ctx=node.ctx), node)
                            else:
                                return ast.copy_location(ast.Constant(value=value), node)
                        if node.id in local_var_map:
                            return ast.copy_location(ast.Name(id=local_var_map[node.id], ctx=node.ctx), node)
                        return node

                    def visit_Subscript(self, node):
                        # Replace args[0], args[1], ... with actual star_args
                        if (
                            isinstance(node.value, ast.Name)
                            and node.value.id in arg_map
                            and isinstance(arg_map[node.value.id], list)
                        ):
                            idx = None
                            if hasattr(node.slice, "value"):  # Python <3.9
                                idx = node.slice.value
                            else:  # Python 3.9+
                                idx = node.slice
                            if isinstance(idx, ast.Constant):
                                idx = idx.value
                            if isinstance(idx, int):
                                value = arg_map[node.value.id][idx]
                                if isinstance(value, ast.AST):
                                    return deepcopy(value)
                                elif isinstance(value, str):
                                    return ast.copy_location(ast.Name(id=value, ctx=ast.Load()), node)
                                else:
                                    return ast.copy_location(ast.Constant(value=value), node)
                        return self.generic_visit(node)

                for stmt in func_def.body:
                    stmt = deepcopy(stmt)
                    stmt = ParamReplacer().visit(stmt)
                    if isinstance(stmt, ast.Assign):
                        # Rename variables to avoid collisions
                        orig_var = stmt.targets[0].id
                        new_var = unique_var(orig_var)
                        local_var_map[orig_var] = new_var
                        stmt.targets[0].id = new_var
                        stmt = self.visit_Assign(stmt)  # Recursively inline
                        stmts.append(stmt)
                    elif isinstance(stmt, ast.Return):
                        # Assign return value to the LHS of the original assignment
                        ret_assign = ast.Assign(targets=assign_node.targets, value=ParamReplacer().visit(stmt.value))
                        stmts.append(ret_assign)
                return stmts
            else:
                return self.generic_visit(assign_node)

        def visit_Expr(self, node):
            # Inline composite calls in expressions (if any)
            return self.generic_visit(node)

    # Flatten the resulting list of statements
    def flatten(stmts):
        flat = []
        for stmt in stmts:
            if isinstance(stmt, list):
                flat.extend(flatten(stmt))
            else:
                flat.append(stmt)
        return flat

    docstring_stmt = None
    stmts = node.body
    if (
        stmts
        and isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, ast.Constant)
        and isinstance(stmts[0].value.value, str)
    ):
        docstring_stmt = stmts[0]
        stmts = stmts[1:]

    # Recursively inline until no composite calls remain
    changed = True
    while changed:
        changed = False
        new_stmts = []
        for stmt in stmts:
            result = Inliner().visit(stmt)
            if isinstance(result, list):
                new_stmts.extend(result)
                changed = True
            else:
                new_stmts.append(result)
        stmts = flatten(new_stmts)

    # --- Restore docstring at the beginning ---
    if docstring_stmt:
        string = docstring_stmt.value.value
        node.body = [ast.Expr(value=ast.Constant(value=string))] + stmts
    else:
        node.body = stmts
    return node


def main():
    if len(sys.argv) != 3:
        print("Usage: python unloop_composites.py input_file.py output_file.py")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    with open(input_file, "r") as f:
        src = f.read()
    tree = ast.parse(src)
    func_defs = get_func_defs(tree)
    if "main" not in func_defs:
        print("No main() function found.")
        sys.exit(1)
    main_def = func_defs["main"]
    inline_composite_calls(main_def, func_defs)
    # Remove COMPOSITE_* function definitions from the AST
    tree.body = [node for node in tree.body if not (isinstance(node, ast.FunctionDef) and is_composite_func(node.name))]
    with open(output_file, "w") as f:
        code = astunparse.unparse(tree)
        import re

        # Replace single-quoted docstrings with triple-quoted if they contain \n
        def repl(match):
            content = match.group(1)
            # Unescape newlines for pretty formatting
            pretty = content.encode("utf-8").decode("unicode_escape")
            return f'"""{pretty}"""'

        code = re.sub(r'"""(.*?)"""', repl, code, flags=re.DOTALL)
        code = re.sub(r"\'([^\']*\\n[^\']*)\'", repl, code)
        f.write(code)
    format_file_with_black(output_file)


if __name__ == "__main__":
    main()
