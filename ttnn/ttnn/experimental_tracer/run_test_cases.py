import importlib.util
import sys
import json
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
import numpy as np
import types
from pathlib import Path
import argparse
import re


def load_module_from_file(filepath, module_name):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_dtype(dtype_str):
    # e.g., "torch.float32"
    if dtype_str.startswith("torch."):
        return getattr(torch, dtype_str.split(".")[1])
    raise ValueError(f"Unknown dtype: {dtype_str}")


def generate_input(input_spec):
    if isinstance(input_spec, list):
        return [generate_input(spec) for spec in input_spec]
    if isinstance(input_spec, str) and input_spec == "<class 'int'>":
        # TODO: find out why this is happening
        return 1
    shape = input_spec["shape"]
    dtype = parse_dtype(input_spec["dtype"])

    # Use torch.randn for float, torch.randint for int
    if dtype.is_floating_point:
        if len(shape):
            res = torch.randn(*shape, dtype=dtype)
        else:
            res = torch.rand(shape, dtype=dtype)
        return res

    elif dtype in (torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8):
        return torch.randint(0, 10, shape, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def main(file1, file2, json_file, key_regex=None):
    # Load both modules
    mod1 = load_module_from_file(file1, "mod1")
    mod2 = load_module_from_file(file2, "mod2")

    # Load test cases
    with open(json_file, "r") as f:
        test_cases = json.load(f)

    results = []
    device_id = 0
    device = ttnn.open_device(device_id=device_id)
    for case in test_cases:
        func_name = case["function"]
        if key_regex and not re.search(key_regex, func_name):
            continue
        inputs_spec = case["inputs"]
        output_spec = case["outputs"]

        # Check both modules have the function
        if not hasattr(mod1, func_name) or not hasattr(mod2, func_name):
            print(f"Function {func_name} not found in both files, skipping.")
            continue

        func1 = getattr(mod1, func_name)
        func2 = getattr(mod2, func_name)

        # Generate inputs
        inputs = [generate_input(spec) for spec in inputs_spec]

        def to_bfloat16(tensors):
            if isinstance(tensors, list):
                return [to_bfloat16(t) for t in tensors]
            return (
                tensors.to(torch.bfloat16)
                if isinstance(tensors, torch.Tensor) and tensors.dtype.is_floating_point
                else tensors
            )

        inputs = to_bfloat16(inputs)

        # convert inputs to ttnn tensors
        def to_ttnn(tensors):
            if isinstance(tensors, list):
                return [to_ttnn(t) for t in tensors]
            return ttnn.from_torch(tensors, device=device) if isinstance(tensors, torch.Tensor) else tensors

        def to_torch(tensors):
            if isinstance(tensors, (tuple, list)):
                return [to_torch(t) for t in tensors]
            return ttnn.to_torch(tensors) if isinstance(tensors, ttnn.Tensor) else tensors

        inputs2 = to_ttnn(inputs)

        # Run both functions
        try:
            out1 = func1(*inputs)
            out2 = func2(*inputs2)
            out2 = to_torch(out2)
        except Exception as e:
            print(f"Error running {func_name}: {e}")
            results.append((func_name, False, f"Exception: {e}"))
            continue

        def compare_pcc(tensors1, tensors2):
            if isinstance(tensors1, (list, tuple)) and not isinstance(tensors2, (tuple, list)):
                return False, f"Unmatched tensor types: {type(tensors1)} vs {type(tensors2)}"
            if isinstance(tensors2, (tuple, list)) and isinstance(tensors1, (tuple, list)):
                if len(tensors1) != len(tensors2):
                    shapes1 = [i.shape if isinstance(i, torch.Tensor) else i for i in tensors1]
                    shapes2 = [i.shape if isinstance(i, torch.Tensor) else i for i in tensors2]
                    return (
                        False,
                        f"Length mismatch: {len(tensors1)} vs {len(tensors2)} with shapes {shapes1} and {shapes2}",
                    )
                msgs = []
                res = True
                counter = 0
                for t1, t2 in zip(tensors1, tensors2):
                    equal, msg = check_with_pcc_without_tensor_printout(t1, t2)
                    res = res and equal
                    if isinstance(msg, (int, float)):
                        msg = f"{counter})PCC:{msg}"
                    msgs.append(str(msg))
                    counter += 1
                return res, "\t".join(msgs)
            elif isinstance(tensors1, torch.Tensor) and isinstance(tensors2, torch.Tensor):
                return check_with_pcc_without_tensor_printout(tensors1, tensors2)
            return (
                False,
                f"Unrecognized types for tensor1 {type(tensors1)} and tensor2 {type(tensors2)} for PCC calculation",
            )

        # Compare outputs
        try:
            if isinstance(out1, torch.Tensor) and isinstance(out2, torch.Tensor):
                equal, pcc_msg = compare_pcc([out1], [out2])
            else:
                equal, pcc_msg = compare_pcc(out1, out2)
        except RuntimeError as e:
            print(f"Error comparing outputs for {func_name}: {e}")
            results.append((func_name, False, f"Comparison Exception: {e}"))
            continue

        if equal:
            results.append((func_name, True, pcc_msg))
        else:
            results.append((func_name, False, pcc_msg))
    ttnn.close_device(device)

    # Summary
    print("\nSummary:")
    for func_name, passed, msg in results:
        print(f"{func_name}: {'PASS' if passed else 'FAIL'} {msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file1")
    parser.add_argument("file2")
    parser.add_argument("json_file")
    parser.add_argument("--key", "-k", type=str, default=None, help="Regex to filter function names")
    args = parser.parse_args()
    main(args.file1, args.file2, args.json_file, args.key)
