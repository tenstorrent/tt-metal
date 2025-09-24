# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import dataclasses
import pathlib
import shutil
import sys
import time
import traceback
import types
import os
import yaml
from threading import Lock
import inspect

from contextlib import contextmanager
from functools import wraps
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
from typing import Callable

from loguru import logger

import ttnn
import ttnn.database

# Global lock for thread-safe YAML file operations
_yaml_lock = Lock()


def format_arguments_for_logging(function_args, function_kwargs, operation_name):
    """Simple argument capture for logging - no coloring, minimal formatting."""
    arg_lines = []

    # Get function signature to map positional args to parameter names
    param_names = []
    try:
        if hasattr(ttnn, operation_name.split(".")[-1]):
            func = getattr(ttnn, operation_name.split(".")[-1])
            if hasattr(func, "function"):  # It's an Operation/FastOperation
                func = func.function
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
    except:
        pass

    # Format positional arguments
    for i, arg in enumerate(function_args):
        param_name = param_names[i] if i < len(param_names) else f"arg_{i}"
        arg_info = format_single_argument(param_name, arg)
        if arg_info:
            arg_lines.append(arg_info)

    # Format keyword arguments
    for key, value in function_kwargs.items():
        arg_info = format_single_argument(key, value)
        if arg_info:
            arg_lines.append(arg_info)

    return "\n".join(arg_lines)


def format_single_argument(param_name, value):
    """Simple argument formatting - no coloring."""
    try:
        if hasattr(value, "__class__") and "ttnn" in str(value.__class__):
            # Handle TTNN tensors
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                shape_str = str(value.shape) if hasattr(value, "shape") else "unknown_shape"
                dtype_str = str(value.dtype) if hasattr(value, "dtype") else "unknown_dtype"
                layout_str = str(value.layout) if hasattr(value, "layout") else "unknown_layout"

                # Try to get memory config info
                memory_info = ""
                if hasattr(value, "memory_config"):
                    try:
                        memory_info = f" memory_config = {value.memory_config()}"
                    except:
                        memory_info = " memory_config = unknown"

                return f"{param_name} : shape = {shape_str} data_type = {dtype_str} layout = {layout_str}{memory_info}"

        elif isinstance(value, (int, float, complex)):
            # Handle scalar values
            if isinstance(value, float):
                return f"{param_name} = {value:.6f}"
            else:
                return f"{param_name} = {value}"

        elif isinstance(value, (str, bool)):
            return f"{param_name} = {value}"

        elif isinstance(value, (list, tuple)):
            if len(value) <= 3:  # Only show small collections
                return f"{param_name} = {value}"
            else:
                return f"{param_name} = [{type(value[0]).__name__} array of length {len(value)}]"

        elif value is None:
            return f"{param_name} = None"

        else:
            # Generic fallback
            return f"{param_name} = {type(value).__name__}"

    except Exception as e:
        return f"{param_name} = <error formatting: {e}>"

    return None


def log_operation_info(operation_name, function_args, function_kwargs):
    """Shared function to handle operation logging for both Operation and FastOperation."""
    pre_operation_state = None
    try:
        # Capture state before operation
        python_op_id = ttnn._ttnn.fetch_and_increment_python_operation_id()
        # Clear any previous thread-local device operation ID
        ttnn._ttnn.clear_first_assigned_device_operation_id()

        pre_operation_state = {
            "python_op_id": python_op_id,
            "callstack": get_filtered_python_call_stack(),
            "args_info": format_arguments_for_logging(function_args, function_kwargs, operation_name),
        }
    except Exception as e:
        print(f"Warning: Failed to capture pre-operation state for {operation_name}: {e}")

    return pre_operation_state


def finalize_operation_logging(operation_name, pre_operation_state):
    """Shared function to finalize operation logging after execution."""
    if pre_operation_state:
        try:
            # Get the first assigned device operation ID for this thread
            first_device_op_id = ttnn._ttnn.get_first_assigned_device_operation_id()

            # Check if a device operation actually occurred
            if first_device_op_id > 0:
                # Device operation occurred - use the first assigned ID
                actual_device_op_id = first_device_op_id
            else:
                # No device operation - this is a host-only operation
                actual_device_op_id = None

            write_new_operation_info(
                operation_name=operation_name,
                python_op_id=pre_operation_state["python_op_id"],
                device_op_id=actual_device_op_id,
                callstack=pre_operation_state["callstack"],
                args_info=pre_operation_state["args_info"],
            )
        except Exception as e:
            print(f"Warning: Failed to log operation {operation_name}: {e}")


def write_new_operation_info(operation_name: str, python_op_id: int, device_op_id, callstack: str, args_info: str = ""):
    """Write operation info to new ops-new.yaml file with both python_operation_id and device_operation_id."""

    # Create directory if it doesn't exist
    dir_path = "./generated/inspector/ops"
    os.makedirs(dir_path, exist_ok=True)

    # Create filename with full path
    filename = os.path.join(dir_path, "ops.yaml")

    # Prepare operation data
    operation_data = {
        "python_operation_id": python_op_id,
        "device_operation_id": device_op_id if device_op_id is not None else "none",
        "operation_name": operation_name,
        "callstack": callstack,
        "arguments": args_info,
    }

    # Thread-safe file writing
    with _yaml_lock:
        try:
            # Load existing data if file exists
            existing_data = []
            if os.path.exists(filename):
                try:
                    with open(filename, "r") as f:
                        existing_data = yaml.safe_load(f) or []
                except Exception:
                    existing_data = []

            # Append new operation
            existing_data.append(operation_data)

            # Write back to file
            with open(filename, "w") as f:
                yaml.safe_dump(existing_data, f, default_flow_style=False)

        except Exception as e:
            # Don't fail the operation if logging fails, just print a warning
            print(f"Warning: Failed to write to ops.yaml: {e}")


def get_filtered_python_call_stack():
    """Get Python call stack, filtering out internal framework code."""
    stack_frames = traceback.format_stack()
    filtered_frames = []

    for frame in stack_frames:
        # Skip internal framework files
        if not any(
            internal in frame
            for internal in ["/site-packages/", "/lib/python", "decorators.py", "/pytest/", "/pluggy/", "__call__"]
        ):
            filtered_frames.append(frame.strip())

    return "\n".join(filtered_frames)


def compare_tensors_using_pcc(
    python_fully_qualified_name, golden_outputs, outputs, desired_pcc, level, fail_on_bad_comparison
):
    import torch

    from models.common.utility_functions import comp_pcc

    if isinstance(outputs, ttnn.Tensor):
        if not isinstance(golden_outputs, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(golden_outputs)}")
        outputs = [outputs]
        golden_outputs = [golden_outputs]
    elif isinstance(outputs, torch.Tensor):
        if not isinstance(golden_outputs, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(golden_outputs)}")
        outputs = [outputs]
        golden_outputs = [golden_outputs]
    else:
        if not isinstance(outputs, (list, tuple)):
            raise TypeError(f"Expected list or tuple, got {type(outputs)}")
        if not isinstance(golden_outputs, (list, tuple)):
            raise TypeError(f"Expected list or tuple, got {type(golden_outputs)}")

    comparison_records = []
    for index, (golden_output, output) in enumerate(zip(golden_outputs, outputs)):
        if not isinstance(output, torch.Tensor):
            torch_output = ttnn.to_torch(output)
        else:
            torch_output = output
        matches, actual_pcc = comp_pcc(golden_output, torch_output, desired_pcc)
        comparison_record = ttnn.database.TensorComparisonRecord(
            tensor_id=output.tensor_id,
            golden_tensor_id=golden_output.tensor_id,
            matches=matches,
            desired_pcc=desired_pcc,
            actual_pcc=actual_pcc,
        )
        comparison_records.append(comparison_record)

        if not matches:
            error_message = f"{python_fully_qualified_name}: Comparing output tensor {index} against CPU {level} failed: pcc is {actual_pcc} but should be >={desired_pcc}"
            if fail_on_bad_comparison:
                raise RuntimeError(error_message)
            else:
                logger.error(error_message)

    return comparison_records


PRE_OPERATION_HOOKS = []
POST_OPERATION_HOOKS = []

push_current_command_queue_id_for_thread = ttnn._ttnn.core.push_current_command_queue_id_for_thread
pop_current_command_queue_id_for_thread = ttnn._ttnn.core.pop_current_command_queue_id_for_thread
get_current_command_queue_id_for_thread = ttnn._ttnn.core.get_current_command_queue_id_for_thread


@contextmanager
def register_pre_operation_hook(hook):
    """

    register_pre_operation_hook is a context manager that registers a pre-operation hook. The hook can be used to run custom code before the operation is executed.

    Args:
        operation: The operation that is being called.
        args: The arguments that are passed to the operation.
        kwargs: The keyword arguments that are passed to the operation.

    Returns:
        `None`: the hook is executed.

    """

    global PRE_OPERATION_HOOKS
    PRE_OPERATION_HOOKS.append(hook)
    yield
    PRE_OPERATION_HOOKS.pop()


@contextmanager
def command_queue(cq_id: int):
    """Context manager to set a default command queue for all TTNN operations within this context.

    Operations within this context will use the specified cq_id unless they explicitly
    provide their own cq_id parameter, which takes precedence.

    Args:
        cq_id: The command queue ID to use for operations in this context

    Example:
        with ttnn.command_queue(1):
            result = ttnn.some_operation(tensor)  # Will use cq_id 1
            result2 = ttnn.other_operation(tensor, queue_id=0)  # Will use cq_id 0 (overrides context)
    """
    if cq_id is None:
        raise ValueError("cq_id cannot be None in command_queue context")

    push_current_command_queue_id_for_thread(cq_id)
    try:
        yield
    finally:
        # Check if command queue is in expected state when exiting context
        current_cq_id = get_current_command_queue_id_for_thread()
        if current_cq_id != cq_id:
            logger.warning(
                f"command_queue({cq_id}) context exiting with unexpected command queue ID: {current_cq_id}. "
                f"This might indicate an operation didn't properly restore the command queue state. "
                f"Restoring to original value {cq_id}."
            )
        pop_current_command_queue_id_for_thread()


@contextmanager
def register_post_operation_hook(hook):
    """

    register_post_operation_hook is a context manager that registers a post-operation hook. The hook can be used to run custom code after the operation is executed.

    Args:
        operation: The operation that is being called.
        args: The arguments that are passed to the operation.
        kwargs: The keyword arguments that are passed to the operation.
        output: The output of the operation.

    Returns:
        `None`: the hook is executed.

    """

    global POST_OPERATION_HOOKS
    POST_OPERATION_HOOKS.append(hook)
    yield
    POST_OPERATION_HOOKS.pop()


def get_devices(object_value):
    devices = set()
    if isinstance(object_value, ttnn.Tensor):
        if ttnn.is_tensor_storage_on_device(object_value) and object_value.is_allocated():
            devices.update(object_value.devices())
    elif isinstance(object_value, ttnn.Device):
        devices.add(object_value)
    elif isinstance(object_value, (list, tuple)):
        for element in object_value:
            devices |= get_devices(element)
    elif isinstance(object_value, dict):
        for value in object_value.values():
            devices |= get_devices(value)
    return devices


def get_tensors(object_value, tensor_type):
    tensors = []
    if isinstance(object_value, tensor_type):
        tensors.append(object_value)
    elif isinstance(object_value, (list, tuple)):
        for element in object_value:
            tensors += get_tensors(element, tensor_type)
    elif isinstance(object_value, dict):
        for value in object_value.values():
            tensors += get_tensors(value, tensor_type)
    return tensors


def get_ttnn_tensors(object_value):
    return get_tensors(object_value, ttnn.Tensor)


def get_all_tensors(object_value):
    import torch

    return get_tensors(object_value, (ttnn.Tensor, torch.Tensor))


def set_tensor_id(tensor, force=False):
    import torch

    if isinstance(tensor, (ttnn.Tensor, torch.Tensor)):
        if not force and hasattr(tensor, "tensor_id") and tensor.tensor_id is not None:
            return
        tensor.tensor_id = ttnn._ttnn.fetch_and_increment_tensor_id()
    elif isinstance(tensor, (list, tuple)):
        for element in tensor:
            set_tensor_id(element, force)
    else:
        raise RuntimeError(f"Unsupported input to set_tensor_id: {type(tensor)}")


OPERATION_CALL_STACK = []


@dataclasses.dataclass
class OutputWithDuration:
    output: any
    duration: float


def default_preprocess_golden_function_inputs(function_args, function_kwargs):
    def recursive_preprocess_golden_function_inputs(object_value):
        if isinstance(object_value, ttnn.Tensor):
            return ttnn.to_torch(object_value)
        elif isinstance(object_value, (list, tuple)):
            new_object_value = [recursive_preprocess_golden_function_inputs(element) for element in object_value]
            return type(object_value)(new_object_value)
        else:
            return object_value

    new_args = []
    for arg in function_args:
        new_arg = recursive_preprocess_golden_function_inputs(arg)
        new_args.append(new_arg)
    new_kwargs = {}
    for key, value in function_kwargs.items():
        new_value = recursive_preprocess_golden_function_inputs(value)
        new_kwargs[key] = new_value
    return tuple(new_args), new_kwargs


def default_postprocess_golden_function_outputs(output, function_args, function_kwargs):
    input_tensors = get_ttnn_tensors((function_args, function_kwargs))

    input_dtype = None
    input_layout = None
    input_device = None
    if input_tensors:
        input_tensor, *_ = input_tensors
        input_dtype = input_tensor.dtype
        input_layout = input_tensor.layout
        if ttnn.is_tensor_storage_on_device(input_tensor):
            input_device = input_tensor.device()

    def recursive_postprocess_golden_function_outputs(output):
        import torch

        if isinstance(output, torch.Tensor):
            return ttnn.from_torch(output, dtype=input_dtype, layout=input_layout, device=input_device)
        elif isinstance(output, (list, tuple)):
            new_output = [recursive_postprocess_golden_function_outputs(element) for element in output]
            return type(output)(new_output)
        else:
            raise RuntimeError(f"Unsupported output type: {type(output)}")

    output = recursive_postprocess_golden_function_outputs(output)
    return output


TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR = {}


def preprocess_global_golden_function_inputs(function_args, function_kwargs):
    input_index = 0

    def recursive_preprocess_golden_function_inputs(object_value):
        nonlocal input_index
        if isinstance(object_value, ttnn.Tensor):
            if object_value.tensor_id is None:
                raise RuntimeError(f"Input tensor does not have a tensor_id")
            if object_value.tensor_id not in TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR:
                if (
                    ttnn.database.query_output_tensor_by_tensor_id(ttnn.CONFIG.report_path, object_value.tensor_id)
                    is not None
                ):
                    logger.warning(
                        f"Intermediate tensor with tensor_id {object_value.tensor_id} (input index: {input_index}) is not found in the global golden tensors. Global golden will be skipped"
                    )
                    raise RuntimeError("Intermediate tensor is not found in the global golden tensors")
                else:
                    logger.warning(
                        f"Input tensor with tensor_id {object_value.tensor_id} (input index: {input_index})  is not found in the global golden tensors. Creating it from ttnn tensor."
                    )
                    golden_tensor = ttnn.to_torch(object_value)
            else:
                golden_tensor = TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR[object_value.tensor_id]
            input_index += 1
            return golden_tensor
        elif isinstance(object_value, ttnn.Shape):
            return tuple(object_value)
        elif isinstance(object_value, (list, tuple)):
            new_object_value = [recursive_preprocess_golden_function_inputs(element) for element in object_value]
            return type(object_value)(new_object_value)
        else:
            return object_value

    try:
        new_args = []
        for arg in function_args:
            new_arg = recursive_preprocess_golden_function_inputs(arg)
            new_args.append(new_arg)
        new_kwargs = {}
        for key, value in function_kwargs.items():
            new_value = recursive_preprocess_golden_function_inputs(value)
            new_kwargs[key] = new_value
        return new_args, new_kwargs
    except Exception as e:
        logger.warning(f"Failed to preprocess global golden function inputs: {e}")
        return None


def postprocess_global_golden_function_outputs(outputs, golden_outputs):
    import torch

    if isinstance(outputs, ttnn.Tensor):
        if not isinstance(golden_outputs, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(golden_outputs)}")
        outputs = [outputs]
        golden_outputs = [golden_outputs]
    elif isinstance(outputs, torch.Tensor):
        if not isinstance(golden_outputs, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(golden_outputs)}")
        outputs = [outputs]
        golden_outputs = [golden_outputs]
    else:
        if not isinstance(outputs, (list, tuple)):
            raise TypeError(f"Expected list or tuple, got {type(outputs)}")
        if not isinstance(golden_outputs, (list, tuple)):
            raise TypeError(f"Expected list or tuple, got {type(golden_outputs)}")

    for output, golden_output in zip(outputs, golden_outputs):
        if output.tensor_id is None:
            raise RuntimeError(f"Output tensor does not have a tensor_id")
        TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR[output.tensor_id] = golden_output


@dataclasses.dataclass
class FastOperation:
    python_fully_qualified_name: str
    function: Callable
    preprocess_golden_function_inputs: Callable
    golden_function: Callable
    postprocess_golden_function_outputs: Callable
    is_cpp_operation: bool
    is_experimental: bool

    @property
    def __name__(self):
        return self.python_fully_qualified_name

    def __gt__(self, other):
        return self.python_fully_qualified_name < other.python_fully_qualified_name

    def __hash__(self):
        return hash(self.python_fully_qualified_name)

    def __call__(self, *function_args, **function_kwargs):
        # Log operation info before execution
        pre_operation_state = log_operation_info(self.python_fully_qualified_name, function_args, function_kwargs)

        cq_id = None
        if "queue_id" in function_kwargs:
            cq_id = function_kwargs.pop("queue_id")
        elif "cq_id" in function_kwargs:
            cq_id = function_kwargs.pop("cq_id")

        if cq_id is None:
            result = self.function(*function_args, **function_kwargs)
        else:
            with command_queue(cq_id):
                result = self.function(*function_args, **function_kwargs)

        # Finalize operation logging after execution
        finalize_operation_logging(self.python_fully_qualified_name, pre_operation_state)

        return result

    def __post_init__(self):
        if self.function.__doc__ is None:
            return

        # Delete the signature line created by pybind11
        docstring_lines = self.function.__doc__.split("\n")
        op_name = self.python_fully_qualified_name.split(".")[-1]
        if f"{op_name}(" in docstring_lines[0]:
            docstring_lines.pop(0)
        self.__doc__ = "\n".join(docstring_lines)

        # # TEMP HACK to read docstring from the file
        # doc_folder = pathlib.Path(__file__).parent.parent.parent / "ops_docs"
        # doc_file = doc_folder / f"{self.python_fully_qualified_name}.md"
        # if doc_file.exists():
        #     with open(doc_file, "r") as f:
        #         self.__doc__ = f.read()


@dataclasses.dataclass
class Operation:
    python_fully_qualified_name: str
    function: Callable
    preprocess_golden_function_inputs: Callable
    golden_function: Callable
    postprocess_golden_function_outputs: Callable
    is_cpp_operation: bool
    is_experimental: bool

    @property
    def __name__(self):
        return self.python_fully_qualified_name

    def __gt__(self, other):
        return self.python_fully_qualified_name < other.python_fully_qualified_name

    def __hash__(self):
        return hash(self.python_fully_qualified_name)

    def __post_init__(self):
        function = self.function

        self.preprocess_golden_function_inputs = (
            self.preprocess_golden_function_inputs or default_preprocess_golden_function_inputs
        )
        self.postprocess_golden_function_outputs = (
            self.postprocess_golden_function_outputs or default_postprocess_golden_function_outputs
        )

        def set_output_tensor_id_decorator(function):
            @wraps(function)
            def call_wrapper(*function_args, **function_kwargs):
                output = function(*function_args, **function_kwargs)
                output_tensors = get_all_tensors(output)
                # Set new tensor id to store the outputs of in-place operations correctly
                set_tensor_id(output_tensors, force=True)
                return output

            return call_wrapper

        def duration_decorator(function):
            @wraps(function)
            def call_wrapper(*function_args, **function_kwargs):
                start = time.time()
                output = function(*function_args, **function_kwargs)
                end = time.time()
                duration = end - start
                return OutputWithDuration(output, duration)

            return call_wrapper

        def comparison_decorator(function):
            @wraps(function)
            def call_wrapper(*function_args, **function_kwargs):
                import torch

                if self.golden_function is not None:
                    local_golden_function_args, local_golden_function_kwargs = self.preprocess_golden_function_inputs(
                        function_args, function_kwargs
                    )
                    global_golden_function_args_and_kwargs = preprocess_global_golden_function_inputs(
                        function_args, function_kwargs
                    )

                function_return_value = function(*function_args, **function_kwargs)

                local_tensor_comparison_records = []
                global_tensor_comparison_records = []

                if self.golden_function is None:
                    logger.debug(
                        f"{self.python_fully_qualified_name}: Skipping comparison against CPU because golden_function is not provided"
                    )
                    return function_return_value, (
                        local_tensor_comparison_records,
                        [],
                        global_tensor_comparison_records,
                        [],
                    )

                if isinstance(function_return_value, OutputWithDuration):
                    output = function_return_value.output
                else:
                    output = function_return_value

                logger.debug(f"{self.python_fully_qualified_name}: Comparing against CPU")
                local_golden_function_output = self.golden_function(
                    *local_golden_function_args, **local_golden_function_kwargs
                )

                global_golden_function_output = None
                if global_golden_function_args_and_kwargs is not None:
                    global_golden_function_args, global_golden_function_kwargs = global_golden_function_args_and_kwargs
                    global_golden_function_output = self.golden_function(
                        *global_golden_function_args, **global_golden_function_kwargs
                    )

                if local_golden_function_output is not None:
                    set_tensor_id(local_golden_function_output)
                    local_tensor_comparison_records = compare_tensors_using_pcc(
                        self.python_fully_qualified_name,
                        local_golden_function_output,
                        output,
                        desired_pcc=ttnn.CONFIG.comparison_mode_pcc,
                        level="locally",
                        fail_on_bad_comparison=ttnn.CONFIG.comparison_mode_should_raise_exception,
                    )

                if global_golden_function_output is not None:
                    set_tensor_id(global_golden_function_output)
                    postprocess_global_golden_function_outputs(output, global_golden_function_output)
                    global_tensor_comparison_records = compare_tensors_using_pcc(
                        self.python_fully_qualified_name,
                        global_golden_function_output,
                        output,
                        desired_pcc=ttnn.CONFIG.comparison_mode_pcc,
                        level="globally",
                        fail_on_bad_comparison=ttnn.CONFIG.comparison_mode_should_raise_exception,
                    )

                if isinstance(local_golden_function_output, torch.Tensor):
                    local_golden_function_output = [local_golden_function_output]
                if isinstance(global_golden_function_output, torch.Tensor):
                    global_golden_function_output = [global_golden_function_output]

                return function_return_value, (
                    local_tensor_comparison_records,
                    local_golden_function_output,
                    global_tensor_comparison_records,
                    global_golden_function_output,
                )

            return call_wrapper

        def runtime_decorator(function):
            @wraps(function)
            def call_wrapper(*function_args, **function_kwargs):
                if ttnn.CONFIG.report_path is not None:
                    # If the database already exists, get the operation_id from the latest operation
                    latest_operation = ttnn.database.query_latest_operation(ttnn.CONFIG.report_path)
                    if latest_operation is not None:
                        operation_id = latest_operation.operation_id + 1
                        ttnn._ttnn.set_python_operation_id(operation_id)

                    latest_tensor = ttnn.database.query_latest_tensor(ttnn.CONFIG.report_path)
                    if latest_tensor is not None:
                        tensor_id = latest_tensor.tensor_id + 1
                        ttnn._ttnn.set_tensor_id(tensor_id)

                operation_id = ttnn._ttnn.get_python_operation_id()
                is_top_level_operation = len(OPERATION_CALL_STACK) == 1

                decorated_function = function

                if not is_top_level_operation:
                    return decorated_function(*function_args, **function_kwargs)

                cq_id = None
                if "queue_id" in function_kwargs:
                    cq_id = function_kwargs.pop("queue_id")
                elif "cq_id" in function_kwargs:
                    cq_id = function_kwargs.pop("cq_id")

                for hook in PRE_OPERATION_HOOKS:
                    hook_return_value = hook(self, function_args, function_kwargs)
                    if hook_return_value is not None:
                        raise RuntimeError(
                            f"Pre-operation hook {hook} returned {hook_return_value} but must return None"
                        )

                if ttnn.CONFIG.enable_logging and ttnn.CONFIG.enable_graph_report:
                    if not ttnn.tracer.is_tracing_enabled():
                        ttnn.tracer.enable_tracing()

                if ttnn.tracer.ENABLE_TRACER:
                    decorated_function = ttnn.tracer.trace_ttnn_operation(
                        self.python_fully_qualified_name, decorated_function
                    )

                if ttnn.CONFIG.enable_logging or ttnn.CONFIG.enable_comparison_mode:
                    input_tensors = get_all_tensors((function_args, function_kwargs))
                    set_tensor_id(input_tensors)
                    decorated_function = set_output_tensor_id_decorator(decorated_function)

                if ttnn.CONFIG.enable_logging:
                    devices = get_devices((function_args, function_kwargs))
                    for device in devices:
                        ttnn.synchronize_device(device)

                    logger.debug(f"Started {self.python_fully_qualified_name:50}")

                    if ttnn.CONFIG.report_path is not None:
                        cluster_descriptor_path = pathlib.Path(ttnn.CONFIG.report_path) / "cluster_descriptor.yaml"
                        if not cluster_descriptor_path.exists():
                            save_cluster_descriptor(str(cluster_descriptor_path))
                        ttnn.database.insert_operation(ttnn.CONFIG.report_path, operation_id, self, None)
                        ttnn.database.insert_stack_trace(
                            ttnn.CONFIG.report_path, operation_id, traceback.format_stack()
                        )
                        ttnn.database.insert_operation_arguments(
                            ttnn.CONFIG.report_path, operation_id, function_args, function_kwargs
                        )
                        ttnn.database.insert_input_tensors(ttnn.CONFIG.report_path, operation_id, input_tensors)

                    decorated_function = duration_decorator(decorated_function)

                if ttnn.CONFIG.enable_comparison_mode:
                    decorated_function = comparison_decorator(decorated_function)

                ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)

                if cq_id is None:
                    output = decorated_function(*function_args, **function_kwargs)
                else:
                    with command_queue(cq_id):
                        output = decorated_function(*function_args, **function_kwargs)

                captured_graph = ttnn.graph.end_graph_capture()

                local_tensor_comparison_records = []
                local_golden_function_output = []
                global_tensor_comparison_records = []
                global_golden_function_output = []
                if ttnn.CONFIG.enable_comparison_mode:
                    (
                        output,
                        (
                            local_tensor_comparison_records,
                            local_golden_function_output,
                            global_tensor_comparison_records,
                            global_golden_function_output,
                        ),
                    ) = output

                if ttnn.CONFIG.enable_logging:
                    for device in devices:
                        ttnn.synchronize_device(device)

                    output, duration = output.output, output.duration
                    logger.debug(f"Finished {self.python_fully_qualified_name:50}")

                    output_tensors = get_all_tensors(output)

                    if ttnn.CONFIG.report_path is not None:
                        ttnn.database.insert_devices(ttnn.CONFIG.report_path, devices)
                        ttnn.database.insert_operation(ttnn.CONFIG.report_path, operation_id, self, duration)
                        ttnn.database.insert_output_tensors(ttnn.CONFIG.report_path, operation_id, output_tensors)
                        ttnn.database.insert_tensor_comparison_records(
                            ttnn.CONFIG.report_path,
                            "local_tensor_comparison_records",
                            local_tensor_comparison_records,
                        )
                        if global_golden_function_output is not None:
                            ttnn.database.store_tensors(ttnn.CONFIG.report_path, local_golden_function_output)
                        ttnn.database.insert_tensor_comparison_records(
                            ttnn.CONFIG.report_path,
                            "global_tensor_comparison_records",
                            global_tensor_comparison_records,
                        )
                        if global_golden_function_output is not None:
                            ttnn.database.store_tensors(ttnn.CONFIG.report_path, global_golden_function_output)
                        ttnn.database.insert_buffers(ttnn.CONFIG.report_path, operation_id, devices)
                        if ttnn.CONFIG.enable_detailed_buffer_report:
                            ttnn.database.insert_buffer_pages(ttnn.CONFIG.report_path, operation_id, devices)

                        if ttnn.CONFIG.enable_graph_report:
                            ttnn.tracer.visualize(
                                ttnn.tracer.GRAPH_STACK[-1],
                                file_name=ttnn.CONFIG.report_path / ttnn.database.GRAPHS_PATH / f"{operation_id}.svg",
                            )
                            # ttnn.database.store_graph(operation_id, ttnn.tracer.GRAPH_STACK[-1])
                        ttnn.database.insert_captured_graph(ttnn.CONFIG.report_path, operation_id, captured_graph)

                for hook in POST_OPERATION_HOOKS:
                    hook_return_value = hook(self, function_args, function_kwargs, output)
                    if hook_return_value is not None:
                        raise RuntimeError(
                            f"Post-operation hook {hook} returned {hook_return_value} but must return None"
                        )

                return output

            return call_wrapper

        function = runtime_decorator(function)
        self.decorated_function = function

    def __call__(self, *function_args, **function_kwargs):
        try:
            pre_operation_state = None

            if not OPERATION_CALL_STACK:
                # Log operation info before execution (only for top-level calls)
                pre_operation_state = log_operation_info(
                    self.python_fully_qualified_name, function_args, function_kwargs
                )

            OPERATION_CALL_STACK.append(self.python_fully_qualified_name)
            output = self.decorated_function(*function_args, **function_kwargs)

            # Finalize operation logging after execution (only for top-level calls)
            if pre_operation_state:
                finalize_operation_logging(self.python_fully_qualified_name, pre_operation_state)

        finally:
            OPERATION_CALL_STACK.pop()
        return output

    __doc__ = property(lambda self: self.decorated_function.__doc__)


class RegisteredOperations:
    def __init__(self):
        self.operations = set()

    def __iter__(self):
        return iter(self.operations)

    def __contains__(self, operation):
        return operation in self.operations

    def add(self, operation, name):
        if operation in self.operations:
            raise RuntimeError(f'Operation with name "{name}" is already registered')
        self.operations.add(operation)


REGISTERED_OPERATIONS = RegisteredOperations()


def query_registered_operations(include_experimental=False):
    sorted_operations = sorted(REGISTERED_OPERATIONS)

    ttnn_operations = [
        operation
        for operation in sorted_operations
        if operation.python_fully_qualified_name.startswith("ttnn.")
        and not operation.python_fully_qualified_name.startswith("ttnn.experimental.")
    ]
    ttl_operations = [
        operation
        for operation in sorted_operations
        if operation.python_fully_qualified_name.startswith("ttnn.experimental.")
    ]
    if include_experimental:
        return ttnn_operations + ttl_operations
    else:
        return ttnn_operations


def dump_operations(csv_file, include_experimental=False):
    import csv
    import pandas as pd

    apis = query_registered_operations(include_experimental)

    def to_dict(obj):
        return {
            "python_fully_qualified_name": obj.python_fully_qualified_name,
            "function": str(obj.function),
            "preprocess_golden_function_inputs": str(obj.preprocess_golden_function_inputs),
            "golden_function": str(obj.golden_function),
            "postprocess_golden_function_outputs": str(obj.postprocess_golden_function_outputs),
            "is_cpp_operation": obj.is_cpp_operation,
            "is_experimental": obj.is_experimental,
        }

    df = pd.DataFrame([to_dict(obj) for obj in apis])
    df.sort_values(by=["is_experimental", "is_cpp_operation", "python_fully_qualified_name"], inplace=True)
    df["has_golden_function"] = df["golden_function"].apply(lambda golden_function: golden_function is not None)
    df = df[
        [
            "python_fully_qualified_name",
            "is_cpp_operation",
            "has_golden_function",
            "is_experimental",
        ]
    ]
    df.to_csv(csv_file, index=False)


def get_golden_function(operation):
    if operation.golden_function is None:
        raise RuntimeError(f"{operation} does not have a golden function")
    return operation.golden_function


def get_fallback_function(operation):
    golden_function = get_golden_function(operation)

    def fallback_function(*function_args, **function_kwargs):
        preprocess_inputs = operation.preprocess_golden_function_inputs or default_preprocess_golden_function_inputs
        postprocess_outputs = (
            operation.postprocess_golden_function_outputs or default_postprocess_golden_function_outputs
        )

        updated_function_args, updated_function_kwargs = preprocess_inputs(function_args, function_kwargs)
        output = golden_function(*updated_function_args, **updated_function_kwargs)
        output = postprocess_outputs(output, function_args, function_kwargs)

        return output

    return fallback_function


def attach_golden_function(
    operation,
    golden_function,
    *,
    preprocess_golden_function_inputs=None,
    postprocess_golden_function_outputs=None,
):
    operation.golden_function = golden_function
    operation.preprocess_golden_function_inputs = (
        preprocess_golden_function_inputs or default_preprocess_golden_function_inputs
    )
    operation.postprocess_golden_function_outputs = (
        postprocess_golden_function_outputs or default_postprocess_golden_function_outputs
    )


def create_module_if_not_exists(module_name):
    if module_name in sys.modules:
        return sys.modules[module_name]

    # Recursively create parent modules if they don't exist
    parent_module_name, _, child_module_name = module_name.rpartition(".")
    if parent_module_name:
        parent_module = create_module_if_not_exists(parent_module_name)
    else:
        parent_module = None

    # Create the module
    new_module = module_from_spec(ModuleSpec(module_name, None))

    if parent_module:
        setattr(parent_module, child_module_name, new_module)
    sys.modules[module_name] = new_module
    return new_module


def register_cpp_operation(target_module: types.ModuleType, func_name: str, function: Callable):
    operation_class = FastOperation if ttnn.CONFIG.enable_fast_runtime_mode else Operation

    operation = operation_class(
        python_fully_qualified_name=function.python_fully_qualified_name,
        function=function,
        golden_function=None,
        preprocess_golden_function_inputs=None,
        postprocess_golden_function_outputs=None,
        is_cpp_operation=True,
        is_experimental=False,
    )

    REGISTERED_OPERATIONS.add(operation, func_name)
    setattr(target_module, func_name, operation)

    return operation


def register_python_operation(
    *,
    name,
    is_experimental=False,
    is_method=False,
    golden_function=None,
    preprocess_golden_function_inputs=None,
    postprocess_golden_function_outputs=None,
    doc=None,
):
    python_fully_qualified_name = name

    def operation_decorator(function: Callable):
        is_cpp_operation = hasattr(function, "__ttnn_operation__")

        if is_cpp_operation:
            raise RuntimeError(f"{function} is a C++ operation, but it is being registered as a Python operation")
        # Disabling for now (See GH issue #18386)
        # elif not is_experimental and not is_method:
        #     logger.debug(f"Should {python_fully_qualified_name} be migrated to C++?")

        operation_class = FastOperation if ttnn.CONFIG.enable_fast_runtime_mode else Operation

        if not ttnn.CONFIG.enable_fast_runtime_mode:
            # Wrap function before attaching documentation to avoid errors
            if doc is not None:

                def doc_decorator(function):
                    @wraps(function)
                    def wrapper(*args, **kwargs):
                        return function(*args, **kwargs)

                    return wrapper

                function = doc_decorator(function)
                function.__doc__ = doc

        operation = operation_class(
            python_fully_qualified_name=python_fully_qualified_name,
            function=function,
            golden_function=golden_function,
            preprocess_golden_function_inputs=preprocess_golden_function_inputs,
            postprocess_golden_function_outputs=postprocess_golden_function_outputs,
            is_cpp_operation=False,
            is_experimental=is_experimental,
        )

        attach_golden_function(
            operation,
            golden_function,
            preprocess_golden_function_inputs=preprocess_golden_function_inputs,
            postprocess_golden_function_outputs=postprocess_golden_function_outputs,
        )

        if not is_method:  # Do not export methods
            module_path, _, func_name = python_fully_qualified_name.rpartition(".")
            if not module_path:
                raise RuntimeError("Module path have to have at least 2 tokens!")
            if not module_path.startswith("ttnn"):
                raise RuntimeError('Module path must start with "ttnn."')

            target_module = create_module_if_not_exists(module_path)

            REGISTERED_OPERATIONS.add(operation, python_fully_qualified_name)
            setattr(target_module, func_name, operation)

        # Wrap method appropriately in order to avoid errors
        if is_method:

            @wraps(operation)
            def method_call(self, *function_args, **function_kwargs):
                return operation(self, *function_args, **function_kwargs)

            return method_call

        return operation

    return operation_decorator


def register_ttl_operation_as_ttnn_operation(python_fully_qualified_name, function):
    function = register_python_operation(
        name=python_fully_qualified_name,
        is_experimental=True,
    )(function)
    return function


def save_cluster_descriptor(dest_path):
    temp_path = ttnn._ttnn.cluster.serialize_cluster_descriptor()

    if not temp_path:
        return None

    shutil.copy(temp_path, dest_path)
