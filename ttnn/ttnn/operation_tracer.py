# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Operation parameter tracing functionality.

This module provides functionality to serialize operation parameters and return values
to JSON files for debugging and analysis purposes.
"""

import json
import pathlib
import sys
from datetime import datetime
from functools import wraps

from loguru import logger

import ttnn

# Global counter for operation numbering in trace files
_OPERATION_COUNTER = 0

# Flag to enable parameter tracing (set by pytest when --trace-params is used)
# This is accessed from decorators.py, so we expose it here
_ENABLE_TRACE = False

# Cached check for --trace-params flag in sys.argv (computed once at module load)
_TRACE_PARAMS_IN_ARGV = None

# Flag to prevent recursion during serialization
_IS_SERIALIZING = False


def _is_tracing_enabled():
    """Check if tracing is enabled, caching the sys.argv check."""
    global _TRACE_PARAMS_IN_ARGV

    # Check sys.argv only once and cache the result
    if _TRACE_PARAMS_IN_ARGV is None:
        _TRACE_PARAMS_IN_ARGV = "--trace-params" in sys.argv

    return _ENABLE_TRACE or _TRACE_PARAMS_IN_ARGV


def serialize_operation_parameters(
    operation_name: str, function_args: tuple, function_kwargs: dict, log_dir: pathlib.Path, return_value=None
):
    """Serialize operation parameters and return value to a single JSON file.

    Args:
        operation_name: Fully qualified operation name (e.g., "ttnn.add")
        function_args: Positional arguments passed to the operation
        function_kwargs: Keyword arguments passed to the operation
        log_dir: Directory where to save the serialized parameters
        return_value: Optional return value from the operation to serialize
    """
    global _OPERATION_COUNTER, _IS_SERIALIZING

    # Prevent recursion: if we're already serializing, don't trace operations called during serialization
    if _IS_SERIALIZING:
        return

    _IS_SERIALIZING = True

    log_dir.mkdir(parents=True, exist_ok=True)

    # Increment operation counter and create unique filename
    _OPERATION_COUNTER += 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_op_name = operation_name.replace(".", "_").replace("::", "_")
    filename = f"{_OPERATION_COUNTER}_{safe_op_name}_{timestamp}.json"
    file_path = log_dir / filename

    # Serialize arguments
    serialized_args = []
    tensor_counter = 0

    def serialize_value(value, name_prefix=""):
        """Recursively serialize a value, handling tensors specially."""
        nonlocal tensor_counter

        if isinstance(value, ttnn.Tensor):
            # Move tensor to CPU and convert to numpy for human-readable format
            cpu_tensor = value
            if hasattr(ttnn, "from_device"):
                cpu_tensor = ttnn.from_device(value)
            elif hasattr(value, "cpu"):
                cpu_tensor = value.cpu()

            # Convert to torch then to numpy for readable values
            import torch

            torch_tensor = ttnn.to_torch(cpu_tensor)
            # Convert to float32 first to handle bfloat16 and other unsupported numpy dtypes
            if torch_tensor.dtype == torch.bfloat16:
                torch_tensor = torch_tensor.float()
            numpy_array = torch_tensor.numpy()

            # Convert numpy array to nested lists for JSON serialization
            values = numpy_array.tolist()

            # Store tensor data directly in metadata (not as separate file)
            tensor_data = {
                "type": "ttnn.Tensor",
                "shape": list(numpy_array.shape),
                "dtype": str(numpy_array.dtype),
                "values": values,
                "original_shape": list(value.shape) if hasattr(value, "shape") else None,
                "original_dtype": str(value.dtype) if hasattr(value, "dtype") else None,
            }

            # Get layout if available (check if it's a property or method)
            if hasattr(value, "layout"):
                layout_attr = getattr(value, "layout", None)
                if callable(layout_attr):
                    layout_value = layout_attr()
                else:
                    layout_value = layout_attr
                if layout_value is not None:
                    tensor_data["layout"] = str(layout_value)

            # Get storage type if available (it's a method)
            if hasattr(value, "storage_type"):
                storage_type_value = value.storage_type()
                if storage_type_value is not None:
                    tensor_data["storage_type"] = str(storage_type_value)

            tensor_counter += 1
            return tensor_data

        # Handle torch.Tensor objects (e.g., passed to from_torch)
        import torch

        if isinstance(value, torch.Tensor):
            # Convert torch tensor to numpy for serialization
            # Convert to float32 first to handle bfloat16 and other unsupported numpy dtypes
            if value.dtype == torch.bfloat16:
                numpy_array = value.detach().cpu().float().numpy()
            else:
                numpy_array = value.detach().cpu().numpy()

            tensor_data = {
                "type": "torch.Tensor",
                "shape": list(numpy_array.shape),
                "dtype": str(numpy_array.dtype),
                "values": numpy_array.tolist(),
            }
            tensor_counter += 1
            return tensor_data

        elif isinstance(value, (int, float, bool, str, type(None))):
            # Simple types that can be JSON serialized
            return value

        elif isinstance(value, (list, tuple)):
            # Recursively serialize list/tuple elements
            return [serialize_value(item, f"{name_prefix}[{i}]") for i, item in enumerate(value)]

        elif isinstance(value, dict):
            # Recursively serialize dict values
            return {k: serialize_value(v, f"{name_prefix}.{k}") for k, v in value.items()}

        else:
            # For other types, convert to string or get basic info
            if hasattr(value, "__dict__"):
                return {"type": type(value).__name__, "repr": str(value)}
            else:
                return {"type": type(value).__name__, "value": str(value)}

    # Serialize positional arguments
    for i, arg in enumerate(function_args):
        serialized_args.append({"position": i, "value": serialize_value(arg, f"arg_{i}")})

    # Serialize keyword arguments
    serialized_kwargs = {k: serialize_value(v, f"kwarg_{k}") for k, v in function_kwargs.items()}

    # Serialize return value if provided
    serialized_return_value = None
    if return_value is not None:
        serialized_return_value = serialize_value(return_value, "return_value")

    # Create complete operation data JSON
    operation_data = {
        "operation_number": _OPERATION_COUNTER,
        "operation_name": operation_name,
        "timestamp": timestamp,
        "args": serialized_args,
        "kwargs": serialized_kwargs,
        "num_tensors": tensor_counter,
    }

    # Add return value if available
    if serialized_return_value is not None:
        operation_data["return_value"] = serialized_return_value

    # Save to single JSON file
    with open(file_path, "w") as f:
        json.dump(operation_data, f, indent=2, default=str)

    logger.debug(f"Serialized operation {operation_name} parameters to {file_path}")

    # Reset serialization flag
    _IS_SERIALIZING = False


def wrap_function_for_tracing(original_function, operation_name: str):
    """Wrap a function to trace its parameters and return value.


    Args:
        original_function: The function to wrap
        operation_name: Fully qualified operation name (e.g., "ttnn.add")

    Returns:
        A wrapped function that checks _ENABLE_TRACE at call time
    """
    # Check if function is already wrapped or is an Operation instance to avoid recursion
    if hasattr(original_function, "__wrapped__") or hasattr(original_function, "decorated_function"):
        return original_function

    # Store original function in closure to avoid recursion
    _original_func = original_function

    # Always wrap, but check the boolean flag at runtime
    @wraps(original_function)
    def wrapped_function(*function_args, **function_kwargs):
        # Call the original function first (use closure variable)
        return_value = _original_func(*function_args, **function_kwargs)

        # Check boolean flag at runtime (single boolean check - very cheap)
        # Initialize _TRACE_PARAMS_IN_ARGV if not already done
        global _TRACE_PARAMS_IN_ARGV
        if _TRACE_PARAMS_IN_ARGV is None:
            _TRACE_PARAMS_IN_ARGV = "--trace-params" in sys.argv

        if _ENABLE_TRACE or _TRACE_PARAMS_IN_ARGV:
            # Determine log directory - use config if available, otherwise default
            log_dir = None
            if hasattr(ttnn.CONFIG, "operation_parameter_log_dir") and ttnn.CONFIG.operation_parameter_log_dir:
                log_dir = pathlib.Path(ttnn.CONFIG.operation_parameter_log_dir)
            elif hasattr(ttnn.CONFIG, "root_report_path") and ttnn.CONFIG.root_report_path:
                log_dir = pathlib.Path(ttnn.CONFIG.root_report_path) / "operation_parameters"
            else:
                log_dir = pathlib.Path("generated/ttnn/operation_parameters")

            # Serialize parameters and return value after the operation completes
            serialize_operation_parameters(operation_name, function_args, function_kwargs, log_dir, return_value)

        return return_value

    return wrapped_function
