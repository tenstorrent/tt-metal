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
import os
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional, Tuple
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

# Flag to control whether tensor values are serialized (default False)
_SERIALIZE_TENSOR_VALUES = False

# Command-line flag constant
_TRACE_PARAMS_FLAG = "--trace-params"


# Helper function to get serializers from tensor_utils (imported lazily to avoid circular imports)
def _get_tensor_utils_serializers():
    """Lazy import of tensor_utils serializers to avoid circular imports."""
    try:
        # Import inside function to avoid circular dependency at module load time
        import importlib.util

        # Check if models.common.tensor_utils exists
        spec = importlib.util.find_spec("models.common.tensor_utils")
        if spec is None:
            return None

        from models.common import tensor_utils

        return {
            "memory_config_to_dict": tensor_utils.memory_config_to_dict,
            "compute_kernel_config_to_dict": tensor_utils.compute_kernel_config_to_dict,
            "program_config_to_dict": tensor_utils.program_config_to_dict,
        }
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not import tensor_utils serializers: {e}")
        return None


def enable_tracing(enable: bool = True) -> None:
    """Enable or disable operation parameter tracing.

    When enabled, all ttnn operations will have their parameters and return values
    serialized to JSON files for debugging and analysis.

    Args:
        enable: If True, tracing is enabled. If False, tracing is disabled.

    Example:
        import ttnn.operation_tracer

        # Enable tracing
        ttnn.operation_tracer.enable_tracing(True)

        # Disable tracing
        ttnn.operation_tracer.enable_tracing(False)
    """
    global _ENABLE_TRACE
    _ENABLE_TRACE = enable


def is_tracing_enabled() -> bool:
    """Check if operation parameter tracing is currently enabled.

    Returns:
        True if tracing is enabled, False otherwise.
    """
    return _is_tracing_enabled()


def enable_tensor_value_serialization(enable: bool = True) -> None:
    """Enable or disable tensor value serialization in trace files.

    By default, only tensor metadata (shape, dtype, layout) is serialized.
    Call this function with enable=True to also serialize tensor values.

    Args:
        enable: If True, tensor values will be serialized. If False, only metadata.

    Example:
        import ttnn.operation_tracer

        # Enable tensor value serialization
        ttnn.operation_tracer.enable_tensor_value_serialization(True)

        # Disable tensor value serialization
        ttnn.operation_tracer.enable_tensor_value_serialization(False)
    """
    global _SERIALIZE_TENSOR_VALUES
    _SERIALIZE_TENSOR_VALUES = enable


def _is_tracing_enabled() -> bool:
    """Check if tracing is enabled, caching the sys.argv check."""
    global _TRACE_PARAMS_IN_ARGV

    # Check sys.argv only once and cache the result
    if _TRACE_PARAMS_IN_ARGV is None:
        _TRACE_PARAMS_IN_ARGV = _TRACE_PARAMS_FLAG in sys.argv

    return _ENABLE_TRACE or _TRACE_PARAMS_IN_ARGV


def _serialize_ttnn_tensor(value: Any, serialize_values: bool) -> Dict[str, Any]:
    """Serialize a ttnn.Tensor to a dictionary.

    Args:
        value: The ttnn.Tensor to serialize
        serialize_values: If True, include tensor values. If False, only metadata.

    Returns:
        Dictionary containing tensor metadata and optionally values.
    """
    import torch

    tensor_data: Dict[str, Any] = {
        "type": "ttnn.Tensor",
        "original_shape": list(value.shape) if hasattr(value, "shape") else None,
        "original_dtype": str(value.dtype) if hasattr(value, "dtype") else None,
    }

    # Check if tensor is distributed and get logical shape
    if hasattr(value, "logical_shape"):
        try:
            logical_shape = value.logical_shape()
            if logical_shape != value.shape:
                tensor_data["logical_shape"] = list(logical_shape)
        except Exception as exc:
            logger.debug(f"Failed to get logical_shape during tracing: {exc}")
    # Get tensor topology and placement information for distributed tensors
    if hasattr(value, "tensor_topology"):
        try:
            topology = value.tensor_topology()
            # Get placements
            placements = topology.placements()
            if placements:
                placement_strs = []
                for placement in placements:
                    placement_type = type(placement).__name__
                    if hasattr(placement, "dim"):
                        placement_strs.append(f"{placement_type}({placement.dim})")
                    else:
                        placement_strs.append(placement_type)
                if placement_strs:
                    if "mesh_device" not in tensor_data:
                        tensor_data["mesh_device"] = {}
                    tensor_data["mesh_device"]["placements"] = placement_strs
            # Get distribution shape
            if hasattr(topology, "distribution_shape"):
                try:
                    dist_shape = topology.distribution_shape()
                    if "mesh_device" not in tensor_data:
                        tensor_data["mesh_device"] = {}
                    tensor_data["mesh_device"]["distribution_shape"] = list(dist_shape)
                except Exception as exc:
                    logger.debug(f"Failed to serialize distribution_shape during tracing: {exc}")

        except Exception as exc:
            logger.debug(f"Failed to serialize tensor_topology during tracing: {exc}")
    # Check if tensor is on a mesh device and capture mesh information
    if hasattr(value, "device") and value.device is not None:
        try:
            device = value.device()
            # Initialize mesh_device dict if it's a mesh device
            is_mesh_device = type(device).__name__ == "MeshDevice" or hasattr(device, "get_device_ids")
            if is_mesh_device:
                if "mesh_device" not in tensor_data:
                    tensor_data["mesh_device"] = {}
                # Check if device is a MeshDevice and get shape (it's a property, not a method)
                if hasattr(device, "shape"):
                    try:
                        mesh_shape = device.shape
                        # MeshShape is indexable, convert to list
                        tensor_data["mesh_device"]["shape"] = list(mesh_shape)
                    except Exception as exc:
                        # Fallback to string representation
                        logger.debug(f"Failed to convert mesh_shape to list, trying string: {exc}")
                        try:
                            tensor_data["mesh_device"]["shape"] = str(mesh_shape)
                        except Exception as exc2:
                            logger.debug(f"Failed to serialize mesh_shape during tracing: {exc2}")

                # Try to get device IDs if available
                if hasattr(device, "get_device_ids"):
                    try:
                        device_ids = device.get_device_ids()
                        tensor_data["mesh_device"]["device_ids"] = list(device_ids) if device_ids else None
                    except Exception as exc:
                        logger.debug(f"Failed to get device_ids during tracing: {exc}")
        except Exception as exc:
            logger.debug(f"Failed to serialize device/mesh information during tracing: {exc}")

    if serialize_values:
        # Move tensor to CPU and convert to numpy for human-readable format
        cpu_tensor = value
        if hasattr(ttnn, "from_device"):
            cpu_tensor = ttnn.from_device(value)
        elif hasattr(value, "cpu"):
            cpu_tensor = value.cpu()

        # Convert to torch then to numpy for readable values
        torch_tensor = ttnn.to_torch(cpu_tensor)
        # Convert to float32 first to handle bfloat16 and other unsupported numpy dtypes
        if torch_tensor.dtype == torch.bfloat16:
            torch_tensor = torch_tensor.float()
        numpy_array = torch_tensor.numpy()

        tensor_data["shape"] = list(numpy_array.shape)
        tensor_data["dtype"] = str(numpy_array.dtype)
        tensor_data["values"] = numpy_array.tolist()
    else:
        # Only include shape and dtype metadata without values
        if hasattr(value, "shape"):
            tensor_data["shape"] = list(value.shape)
        if hasattr(value, "dtype"):
            tensor_data["dtype"] = str(value.dtype)

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

    # Get memory config if available (it's a method)
    if hasattr(value, "memory_config"):
        try:
            memory_config_value = value.memory_config()
            if memory_config_value is not None:
                # Try to use tensor_utils serializer first
                serializers = _get_tensor_utils_serializers()
                if serializers and "memory_config_to_dict" in serializers:
                    try:
                        tensor_data["memory_config"] = serializers["memory_config_to_dict"](memory_config_value)
                    except Exception as exc:
                        # Fallback to repr if serializer fails
                        logger.debug(f"memory_config_to_dict serializer failed, using repr: {exc}")
                        tensor_data["memory_config"] = repr(memory_config_value)
                else:
                    # No serializer available, use repr
                    tensor_data["memory_config"] = repr(memory_config_value)
        except Exception as exc:
            logger.debug(f"Failed to serialize memory_config during tracing: {exc}")
    return tensor_data


def _serialize_torch_tensor(value: Any, serialize_values: bool) -> Dict[str, Any]:
    """Serialize a torch.Tensor to a dictionary.

    Args:
        value: The torch.Tensor to serialize
        serialize_values: If True, include tensor values. If False, only metadata.

    Returns:
        Dictionary containing tensor metadata and optionally values.
    """
    import torch

    tensor_data: Dict[str, Any] = {
        "type": "torch.Tensor",
    }

    if serialize_values:
        # Convert torch tensor to numpy for serialization
        # Convert to float32 first to handle bfloat16 and other unsupported numpy dtypes
        if value.dtype == torch.bfloat16:
            numpy_array = value.detach().cpu().float().numpy()
        else:
            numpy_array = value.detach().cpu().numpy()

        tensor_data["shape"] = list(numpy_array.shape)
        tensor_data["dtype"] = str(numpy_array.dtype)
        tensor_data["values"] = numpy_array.tolist()
    else:
        # Only include shape and dtype metadata without values
        tensor_data["shape"] = list(value.shape)
        tensor_data["dtype"] = str(value.dtype)

    return tensor_data


def serialize_operation_parameters(
    operation_name: str,
    function_args: Tuple[Any, ...],
    function_kwargs: Dict[str, Any],
    log_dir: pathlib.Path,
    return_value: Optional[Any] = None,
    serialize_tensor_values: bool = True,
) -> None:
    """Serialize operation parameters and return value to a single JSON file.

    Args:
        operation_name: Fully qualified operation name (e.g., "ttnn.add")
        function_args: Positional arguments passed to the operation
        function_kwargs: Keyword arguments passed to the operation
        log_dir: Directory where to save the serialized parameters
        return_value: Optional return value from the operation to serialize
        serialize_tensor_values: If True, include tensor values in serialization. If False, only include metadata.
    """
    global _OPERATION_COUNTER, _IS_SERIALIZING

    # Prevent recursion: if we're already serializing, don't trace operations called during serialization
    if _IS_SERIALIZING:
        return

    _IS_SERIALIZING = True
    try:
        log_dir.mkdir(parents=True, exist_ok=True)

        # Increment operation counter and create unique filename
        _OPERATION_COUNTER += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_op_name = operation_name.replace(".", "_").replace("::", "_")
        filename = f"{_OPERATION_COUNTER}_{safe_op_name}_{timestamp}.json"
        file_path = log_dir / filename

        # Serialize arguments
        serialized_args = []

        def serialize_value(value: Any, name_prefix: str = "") -> Any:
            """Recursively serialize a value, handling tensors specially."""
            # Lazy import torch to avoid global import validation errors
            import torch

            if isinstance(value, ttnn.Tensor):
                return _serialize_ttnn_tensor(value, serialize_tensor_values)

            if isinstance(value, torch.Tensor):
                return _serialize_torch_tensor(value, serialize_tensor_values)

            if isinstance(value, (int, float, bool, str, type(None))):
                # Simple types that can be JSON serialized
                return value

            elif isinstance(value, (list, tuple)):
                # Recursively serialize list/tuple elements
                return [serialize_value(item, f"{name_prefix}[{i}]") for i, item in enumerate(value)]

            elif isinstance(value, dict):
                # Recursively serialize dict values
                return {k: serialize_value(v, f"{name_prefix}.{k}") for k, v in value.items()}

            else:
                # Special handling for MeshDevice
                if type(value).__name__ == "MeshDevice":
                    mesh_data = {
                        "type": "MeshDevice",
                        "repr": str(value),
                    }
                    # Try to get mesh shape (it's a property, not a method)
                    if hasattr(value, "shape"):
                        try:
                            mesh_shape = value.shape
                            # MeshShape is indexable, convert to list
                            mesh_data["shape"] = list(mesh_shape)
                        except Exception as exc:
                            # Fallback to string representation
                            logger.debug(f"Failed to convert MeshDevice shape to list, trying string: {exc}")
                            try:
                                mesh_data["shape"] = str(mesh_shape)
                            except Exception as exc2:
                                logger.debug(f"Failed to serialize MeshDevice shape during tracing: {exc2}")

                    # Try to get device IDs
                    if hasattr(value, "get_device_ids"):
                        try:
                            device_ids = value.get_device_ids()
                            mesh_data["device_ids"] = list(device_ids) if device_ids else None
                        except Exception as exc:
                            logger.debug(f"Failed to get MeshDevice device_ids during tracing: {exc}")
                    return mesh_data
                # For other types, try specialized serializers
                type_name = type(value).__name__

                # Try to use tensor_utils serializers first
                serializers = _get_tensor_utils_serializers()

                # Try tensor_utils serializers for specific types
                if serializers:
                    try:
                        if type_name == "MemoryConfig":
                            return serializers["memory_config_to_dict"](value)
                        elif type_name in [
                            "WormholeComputeKernelConfig",
                            "BlackholeComputeKernelConfig",
                            "GrayskullComputeKernelConfig",
                            "DeviceComputeKernelConfig",
                        ]:
                            return serializers["compute_kernel_config_to_dict"](value)
                        elif "ProgramConfig" in type_name and "Matmul" in type_name:
                            # tensor_utils only handles Matmul program configs
                            return serializers["program_config_to_dict"](value)
                    except Exception as e:
                        logger.debug(f"tensor_utils serializer failed for {type_name}: {e}")
                        # Fall through to use __repr__
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
            "operation_id": _OPERATION_COUNTER,
            "operation_name": operation_name,
            "args": serialized_args,
            "kwargs": serialized_kwargs,
        }

        # Add return value if available
        if serialized_return_value is not None:
            operation_data["return_value"] = serialized_return_value

        # Save to single JSON file
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(operation_data, f, indent=2, default=str)
            logger.debug(f"Serialized operation {operation_name} parameters to {file_path}")
        except (OSError, IOError) as e:
            logger.error(f"Failed to write trace file {file_path}: {e}")
            raise
    finally:
        # Always reset serialization flag, even if an exception occurs
        _IS_SERIALIZING = False


def wrap_function_for_tracing(original_function: Any, operation_name: str) -> Any:
    """Wrap a function to trace its parameters and return value.

    Args:
        original_function: The function to wrap
        operation_name: Fully qualified operation name (e.g., "ttnn.add")

    Returns:
        A wrapped function that checks _ENABLE_TRACE at call time
    """
    # Check if function is an Operation instance to avoid recursion
    # (Don't check __wrapped__ as that's set by functools.wraps and is too common)
    if hasattr(original_function, "decorated_function"):
        return original_function

    # Store original function in closure to avoid recursion
    _original_func = original_function

    # Always wrap, but check the boolean flag at runtime
    @wraps(original_function)
    def wrapped_function(*function_args, **function_kwargs):
        # Call the original function first (use closure variable)
        return_value = _original_func(*function_args, **function_kwargs)

        # Check if tracing is enabled (uses helper with cached sys.argv check)
        if _is_tracing_enabled():
            # Determine log directory - use config if available, otherwise default
            log_dir = None
            # First check environment variable (highest priority for custom trace directory)
            if os.environ.get("TTNN_OPERATION_TRACE_DIR"):
                log_dir = pathlib.Path(os.environ["TTNN_OPERATION_TRACE_DIR"])
            elif hasattr(ttnn.CONFIG, "operation_parameter_log_dir") and ttnn.CONFIG.operation_parameter_log_dir:
                log_dir = pathlib.Path(ttnn.CONFIG.operation_parameter_log_dir)
            elif hasattr(ttnn.CONFIG, "root_report_path") and ttnn.CONFIG.root_report_path:
                log_dir = pathlib.Path(ttnn.CONFIG.root_report_path) / "operation_parameters"
            else:
                log_dir = pathlib.Path("generated/ttnn/operation_parameters")

            # Determine if tensor values should be serialized (default False)
            serialize_values = _SERIALIZE_TENSOR_VALUES

            # Serialize parameters and return value after the operation completes
            serialize_operation_parameters(
                operation_name,
                function_args,
                function_kwargs,
                log_dir,
                return_value,
                serialize_tensor_values=serialize_values,
            )

        return return_value

    return wrapped_function
