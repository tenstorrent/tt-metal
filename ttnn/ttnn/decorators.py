# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
import dataclasses
from functools import wraps
import inspect
import os
import time
import traceback
import types

from loguru import logger

import ttnn
import ttnn.database


def compare_tensors_using_pcc(python_fully_qualified_name, golden_outputs, outputs, desired_pcc, level):
    import torch

    from models.utility_functions import comp_pcc

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
        commparison_record = ttnn.database.TensorComparisonRecord(
            tensor_id=output.tensor_id,
            golden_tensor_id=golden_output.tensor_id,
            matches=matches,
            desired_pcc=desired_pcc,
            actual_pcc=actual_pcc,
        )
        comparison_records.append(commparison_record)

        if not matches:
            logger.error(
                f"{python_fully_qualified_name}: Comparing output tensor {index} against CPU {level} failed: pcc is {actual_pcc} but should be >={desired_pcc}"
            )

    return comparison_records


PRE_OPERATION_HOOKS = []


@contextmanager
def register_pre_operation_hook(hook):
    """

    register_pre_operation_hook is a context manager that registers a pre-operation hook. The hook can be used to run custom code before the operation is executed.

    The hook takes in the following arguments:
    - operation: The operation that is being called
    - args: The arguments that are passed to the operation
    - kwargs: The keyword arguments that are passed to the operation

    The hook must return None.

    """

    global PRE_OPERATION_HOOKS
    PRE_OPERATION_HOOKS.append(hook)
    yield
    PRE_OPERATION_HOOKS.pop()


POST_OPERATION_HOOKS = []


@contextmanager
def register_post_operation_hook(hook):
    """

    register_post_operation_hook is a context manager that registers a post-operation hook. The hook can be used to run custom code after the operation is executed.

    The hook takes in the following arguments:
    - operation: The operation that is being called
    - args: The arguments that are passed to the operation
    - kwargs: The keyword arguments that are passed to the operation
    - output: The output of the operation

    The hook must return None.

    """

    global POST_OPERATION_HOOKS
    POST_OPERATION_HOOKS.append(hook)
    yield
    POST_OPERATION_HOOKS.pop()


def get_devices(object_value):
    devices = set()
    if isinstance(object_value, ttnn.Tensor):
        if ttnn.is_tensor_storage_on_device(object_value) and object_value.is_allocated():
            devices.add(object_value.device())
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


TENSOR_ID = 0


def set_tensor_id(tensor, force=False):
    import torch

    global TENSOR_ID
    if isinstance(tensor, (ttnn.Tensor, torch.Tensor)):
        if not force and hasattr(tensor, "tensor_id") and tensor.tensor_id is not None:
            return
        tensor.tensor_id = TENSOR_ID
        TENSOR_ID += 1
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
            return tuple(object_value.with_tile_padding())
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


def posprocess_global_golden_function_outputs(outputs, golden_outputs):
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
    function: callable
    preprocess_golden_function_inputs: callable
    golden_function: callable
    postprocess_golden_function_outputs: callable
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
        return self.function(*function_args, **function_kwargs)

    __doc__ = property(lambda self: self.function.__doc__)


@dataclasses.dataclass
class Operation:
    python_fully_qualified_name: str
    function: callable
    preprocess_golden_function_inputs: callable
    golden_function: callable
    postprocess_golden_function_outputs: callable
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

        def operation_history_decorator(function):
            @wraps(function)
            def call_wrapper(*function_args, **function_kwargs):
                original_operation_history_json = os.environ.get("OPERATION_HISTORY_JSON", None)
                os.environ["OPERATION_HISTORY_JSON"] = str(
                    ttnn.CONFIG.report_path / ttnn.database.OPERATION_HISTORY_JSON
                )
                output = function(*function_args, **function_kwargs)
                if hasattr(ttnn._ttnn.deprecated.operations, "dump_operation_history_to_json"):
                    ttnn._ttnn.deprecated.operations.dump_operation_history_to_json()
                if original_operation_history_json is not None:
                    os.environ["OPERATION_HISTORY_JSON"] = original_operation_history_json
                else:
                    del os.environ["OPERATION_HISTORY_JSON"]
                return output

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
                    )

                if global_golden_function_output is not None:
                    set_tensor_id(global_golden_function_output)
                    posprocess_global_golden_function_outputs(output, global_golden_function_output)
                    global_tensor_comparison_records = compare_tensors_using_pcc(
                        self.python_fully_qualified_name,
                        global_golden_function_output,
                        output,
                        desired_pcc=ttnn.CONFIG.comparison_mode_pcc,
                        level="globally",
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
                    previous_operation = ttnn.database.query_latest_operation(ttnn.CONFIG.report_path)
                    if previous_operation is not None:
                        operation_id = previous_operation.operation_id + 1
                        ttnn._ttnn.set_operation_id(operation_id)

                operation_id = ttnn._ttnn.get_operation_id()
                is_top_level_operation = len(OPERATION_CALL_STACK) == 1

                decorated_function = function

                if not is_top_level_operation:
                    return decorated_function(*function_args, **function_kwargs)

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
                        decorated_function = operation_history_decorator(decorated_function)
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

                output = decorated_function(*function_args, **function_kwargs)

                local_tensor_comparison_records = []
                local_golden_function_output = []
                global_tensor_comparison_records = []
                global_golden_function_output = []
                if ttnn.CONFIG.enable_comparison_mode:
                    output, (
                        local_tensor_comparison_records,
                        local_golden_function_output,
                        global_tensor_comparison_records,
                        global_golden_function_output,
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
                        ttnn.database.insert_buffers(ttnn.CONFIG.report_path, operation_id)
                        if ttnn.CONFIG.enable_detailed_buffer_report:
                            ttnn.database.insert_buffer_pages(ttnn.CONFIG.report_path, operation_id)

                        if ttnn.CONFIG.enable_graph_report:
                            ttnn.tracer.visualize(
                                ttnn.tracer.GRAPH_STACK[-1],
                                file_name=ttnn.CONFIG.report_path / ttnn.database.GRAPHS_PATH / f"{operation_id}.svg",
                            )
                            # ttnn.database.store_graph(operation_id, ttnn.tracer.GRAPH_STACK[-1])

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
            if not OPERATION_CALL_STACK:
                ttnn._ttnn.increment_operation_id()
            OPERATION_CALL_STACK.append(self.python_fully_qualified_name)
            output = self.decorated_function(*function_args, **function_kwargs)
        finally:
            OPERATION_CALL_STACK.pop()
        return output

    __doc__ = property(lambda self: self.decorated_function.__doc__)


REGISTERED_OPERATIONS = set()


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
    df = df[["python_fully_qualified_name", "is_cpp_operation", "has_golden_function", "is_experimental"]]
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
    operation, golden_function, *, preprocess_golden_function_inputs=None, postprocess_golden_function_outputs=None
):
    operation.golden_function = golden_function
    operation.preprocess_golden_function_inputs = (
        preprocess_golden_function_inputs or default_preprocess_golden_function_inputs
    )
    operation.postprocess_golden_function_outputs = (
        postprocess_golden_function_outputs or default_postprocess_golden_function_outputs
    )


def export_operation(python_fully_qualified_name, operation, is_method):
    """
    This function is used to export the operation to the ttnn module using the fully qualified name
    For example, an operation named "ttnn.add" would be exported as ttnn.add.
    Any level of nesting is possible.
    """
    global REGISTERED_OPERATIONS

    if operation in REGISTERED_OPERATIONS:
        raise RuntimeError(f'Operarion with name "{python_fully_qualified_name}" is already registered')

    REGISTERED_OPERATIONS.add(operation)

    # Do not export methods
    if is_method:
        return

    module_path = python_fully_qualified_name.split(".")  # "ttnn.add" -> ["ttnn", "add"]

    if len(module_path) < 2:
        raise RuntimeError("Module path have to have at least 2 tokens!")

    if module_path[0] != "ttnn":
        raise RuntimeError('Module path must start with "ttnn."')

    module_path = module_path[1:]  # ["ttnn", "add"] -> ["add"]

    def recursive_helper(module, path):
        current, *rest = path
        if not rest:
            setattr(module, current, operation)
            return

        current_module = getattr(module, current, types.ModuleType("module_name"))
        setattr(module, current, current_module)
        recursive_helper(current_module, rest)

    recursive_helper(ttnn, module_path)


def register_cpp_operation():
    def operation_decorator(function: callable):
        is_cpp_operation = hasattr(function, "__ttnn_operation__")

        if not is_cpp_operation:
            raise RuntimeError(f"{function} is not a C++ operation)")

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

        export_operation(function.python_fully_qualified_name, operation, is_method=False)
        return operation

    return operation_decorator


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

    def operation_decorator(function: callable):
        is_cpp_operation = hasattr(function, "__ttnn_operation__")

        if is_cpp_operation:
            raise RuntimeError(f"{function} is a C++ operation, but it is being registered as a Python operation")
        elif not is_experimental and not is_method:
            logger.warning(f"Should {python_fully_qualified_name} be migrated to C++?")

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
        export_operation(python_fully_qualified_name, operation, is_method)

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
