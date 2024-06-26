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


def document_input_tensors(python_fully_qualified_name, function, validate_input_tensors):
    signature = inspect.signature(validate_input_tensors)
    arguments = {arg_name: None for arg_name in signature.parameters}
    arguments["operation_name"] = python_fully_qualified_name
    tensor_names = {
        arg_name: None for arg_name in arguments if arg_name not in {"operation_name", "args", "kwargs", "_"}
    }

    tensor_schemas = []

    def document_validate_input_tensor(_):
        nonlocal tensor_schemas

        def wrapper(*_, **kwargs):
            tensor_schemas.append(kwargs)

        return wrapper

    original_validate_input_tensor = ttnn.validate_input_tensor
    ttnn.validate_input_tensor = document_validate_input_tensor(ttnn.validate_input_tensor)
    try:
        validate_input_tensors(**arguments)
    except:
        pass
    ttnn.validate_input_tensor = original_validate_input_tensor

    if not tensor_schemas:
        # Handle the case for the functions without input tensors
        return

    if len(tensor_names) != len(tensor_schemas):
        raise RuntimeError(f"Expected {len(tensor_names)} tensor_schemas, got {len(tensor_schemas)}")

    doc = function.__doc__ if function.__doc__ is not None else ""
    for tensor_name, tensor_schema in zip(tensor_names, tensor_schemas):
        doc = f"{doc}\n    .. list-table:: {tensor_name}\n\n"

        for index, arg_name in enumerate(tensor_schema.keys()):
            arg_name = arg_name.replace("_", " ")
            bullet_point = f"* -" if index == 0 else "  -"
            doc = f"{doc}        {bullet_point} {arg_name}\n"

        for index, value in enumerate(tensor_schema.values()):
            if isinstance(value, (list, tuple)):

                def to_string(object_value):
                    try:
                        if object_value is None:
                            return ""
                        elif isinstance(object_value, ttnn.DataType):
                            return f"ttnn.{object_value.name.lower()}"
                        elif isinstance(object_value, ttnn.Layout):
                            return f"ttnn.{object_value.name}_LAYOUT"
                        else:
                            return f"{object_value}"
                    except Exception as e:
                        return f"{object_value}"

                value = f"{', '.join([to_string(element) for element in value])}"
            bullet_point = f"* -" if index == 0 else "  -"
            doc = f"{doc}        {bullet_point} {value}\n"

    function.__doc__ = f"{doc}\n"


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
class Operation:
    python_fully_qualified_name: str
    function: callable
    validate_input_tensors: callable
    preprocess_golden_function_inputs: callable
    golden_function: callable
    postprocess_golden_function_outputs: callable
    is_cpp_function: bool
    is_experimental: bool

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

        if self.validate_input_tensors is not None:
            document_input_tensors(self.python_fully_qualified_name, function, self.validate_input_tensors)

        def validate_decorator(function):
            @wraps(function)
            def call_wrapper(*function_args, **function_kwargs):
                if self.validate_input_tensors is not None:
                    self.validate_input_tensors(self.python_fully_qualified_name, *function_args, **function_kwargs)
                return function(*function_args, **function_kwargs)

            return call_wrapper

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
                original_operation_history_csv = os.environ.get("OPERATION_HISTORY_CSV", None)
                os.environ["OPERATION_HISTORY_CSV"] = str(ttnn.CONFIG.report_path / ttnn.database.OPERATION_HISTORY_CSV)
                output = function(*function_args, **function_kwargs)
                if hasattr(ttnn._tt_lib.operations, "dump_operation_history_to_csv"):
                    ttnn._tt_lib.operations.dump_operation_history_to_csv()
                if original_operation_history_csv is not None:
                    os.environ["OPERATION_HISTORY_CSV"] = original_operation_history_csv
                else:
                    del os.environ["OPERATION_HISTORY_CSV"]
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

        def fallback_to_golden_function_decorator(function):
            def golden_function(*function_args, **function_kwargs):
                updated_function_args, updated_function_kwargs = self.preprocess_golden_function_inputs(
                    function_args, function_kwargs
                )
                output = self.golden_function(*updated_function_args, **updated_function_kwargs)
                output = self.postprocess_golden_function_outputs(output, function_args, function_kwargs)
                return output

            @wraps(function)
            def call_wrapper(*function_args, **function_kwargs):
                ran_golden_function = False
                try:
                    output = function(*function_args, **function_kwargs)
                except NotImplementedError:
                    ran_golden_function = True
                    logger.warning(
                        f"{self.python_fully_qualified_name}: falling back to CPU due to NotImplementedError"
                    )
                    output = golden_function(*function_args, **function_kwargs)
                except Exception as e:
                    ran_golden_function = True
                    exception_message = "\n".join(str(e).split("\n")[:3])
                    logger.warning(
                        f"{self.python_fully_qualified_name}: falling back to CPU due to {exception_message}"
                    )
                    output = golden_function(*function_args, **function_kwargs)
                if ttnn.CONFIG.throw_exception_on_fallback and ran_golden_function:
                    raise RuntimeError(
                        f"Fallbacks are disabled, but {self.python_fully_qualified_name} used a fallback"
                    )
                return output

            return call_wrapper

        def runtime_decorator(function):
            @wraps(function)
            def call_wrapper(*function_args, **function_kwargs):
                operation_id = ttnn._ttnn.get_operation_id()
                is_top_level_operation = len(OPERATION_CALL_STACK) == 1

                decorated_function = function
                decorated_function = validate_decorator(decorated_function)

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

    __name__ = property(lambda self: self.python_fully_qualified_name)
    __doc__ = property(lambda self: self.decorated_function.__doc__)


REGISTERED_APIS = set()


def query_registered_operations(include_experimental=False):
    sorted_operations = sorted(REGISTERED_APIS)

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


OPERATION_TO_GOLDEN_FUNCTION = {}
OPERATION_TO_FALLBACK_FUNCTION = {}


def get_golden_function(operation):
    return OPERATION_TO_GOLDEN_FUNCTION[operation]


def get_fallback_function(operation):
    return OPERATION_TO_FALLBACK_FUNCTION[operation]


def register_operation(
    *,
    name=None,
    is_experimental=False,
    is_method=False,
    validate_input_tensors=None,
    golden_function=None,
    preprocess_golden_function_inputs=None,
    postprocess_golden_function_outputs=None,
    doc=None,
):
    def operation_decorator(function: callable):
        global REGISTERED_APIS
        global OPERATION_TO_GOLDEN_FUNCTION
        global OPERATION_TO_FALLBACK_FUNCTION

        def fallback_function(*function_args, **function_kwargs):
            preprocess_inputs = preprocess_golden_function_inputs or default_preprocess_golden_function_inputs
            postprocess_outputs = postprocess_golden_function_outputs or default_postprocess_golden_function_outputs

            updated_function_args, updated_function_kwargs = preprocess_inputs(function_args, function_kwargs)
            output = golden_function(*updated_function_args, **updated_function_kwargs)
            output = postprocess_outputs(output, function_args, function_kwargs)

            return output

        if ttnn.CONFIG.enable_fast_runtime_mode:
            OPERATION_TO_GOLDEN_FUNCTION[function] = golden_function
            OPERATION_TO_FALLBACK_FUNCTION[function] = fallback_function
            # Wrap functions before attaching name to avoid errors
            if hasattr(function, "python_fully_qualified_name"):

                def name_decorator(function):
                    @wraps(function)
                    def wrapper(*args, **kwargs):
                        return function(*args, **kwargs)

                    return wrapper

                python_fully_qualified_name = function.python_fully_qualified_name
                function = name_decorator(function)
                function.__name__ = python_fully_qualified_name
            return function

        is_cpp_function = hasattr(function, "__ttnn__")

        python_fully_qualified_name = name
        if is_cpp_function:
            if doc is not None:
                raise RuntimeError(f"Registering {name}: documentation for C++ functiomn has to be set from C++")
            if python_fully_qualified_name is not None:
                raise RuntimeError(f"Registering {name}: name is not allowed for ttnn functions")
            python_fully_qualified_name = function.python_fully_qualified_name  # Replace C++ name with python

        # Wrap functions before attaching documentation to avoid errors
        if doc is not None:

            def doc_decorator(function):
                @wraps(function)
                def wrapper(*args, **kwargs):
                    return function(*args, **kwargs)

                return wrapper

            function = doc_decorator(function)
            function.__doc__ = doc

        operation = Operation(
            python_fully_qualified_name=python_fully_qualified_name,
            function=function,
            validate_input_tensors=validate_input_tensors,
            golden_function=golden_function,
            preprocess_golden_function_inputs=preprocess_golden_function_inputs,
            postprocess_golden_function_outputs=postprocess_golden_function_outputs,
            is_cpp_function=is_cpp_function or is_experimental,
            is_experimental=is_experimental,
        )

        if is_method:

            @wraps(operation)
            def method_call(self, *function_args, **function_kwargs):
                return operation(self, *function_args, **function_kwargs)

            api = method_call
        else:
            api = operation

        if api in REGISTERED_APIS:
            raise RuntimeError(f"{api} is already registered")
        REGISTERED_APIS.add(api)

        OPERATION_TO_GOLDEN_FUNCTION[api] = golden_function
        OPERATION_TO_FALLBACK_FUNCTION[api] = fallback_function

        return api

    return operation_decorator


def register_ttl_operation_as_ttnn_operation(python_fully_qualified_name, function):
    function = register_operation(
        name=python_fully_qualified_name,
        is_experimental=True,
    )(function)
    return function
