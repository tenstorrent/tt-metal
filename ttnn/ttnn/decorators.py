# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
import dataclasses
from functools import wraps
import inspect
import time

from loguru import logger

import ttnn


def calculate_pcc(golden_outputs, outputs, desired_pcc):
    import torch

    from models.utility_functions import comp_pcc

    if isinstance(outputs, ttnn.Tensor):
        if not isinstance(golden_outputs, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(golden_outputs)}")
        outputs = [outputs]
        golden_outputs = [golden_outputs]
    else:
        if not isinstance(outputs, (list, tuple)):
            raise TypeError(f"Expected list or tuple, got {type(outputs)}")
        if not isinstance(golden_outputs, (list, tuple)):
            raise TypeError(f"Expected list or tuple, got {type(golden_outputs)}")

    for golden_output, output in zip(golden_outputs, outputs):
        output = ttnn.to_torch(output)
        passed, actual_pcc = comp_pcc(golden_output, output, desired_pcc)
        if not passed:
            return False, actual_pcc
    return True, actual_pcc


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


def document_input_tensors(name, function, validate_input_tensors):
    signature = inspect.signature(validate_input_tensors)
    arguments = {arg_name: None for arg_name in signature.parameters}
    arguments["operation_name"] = name
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

                def to_string(object):
                    try:
                        if object is None:
                            return ""
                        elif isinstance(object, ttnn.DataType):
                            return f"ttnn.{object.name.lower()}"
                        elif isinstance(object, ttnn.Layout):
                            return f"ttnn.{object.name}_LAYOUT"
                        else:
                            return f"{object}"
                    except Exception as e:
                        return f"{object}"

                value = f"{', '.join([to_string(element) for element in value])}"
            bullet_point = f"* -" if index == 0 else "  -"
            doc = f"{doc}        {bullet_point} {value}\n"

    function.__doc__ = f"{doc}\n"


def get_devices(object):
    devices = set()
    if isinstance(object, ttnn.Tensor):
        if ttnn.is_tensor_storage_on_device(object) and object.is_allocated():
            devices.add(object.device())
    elif isinstance(object, ttnn.Device):
        devices.add(object)
    elif isinstance(object, (list, tuple)):
        for element in object:
            devices |= get_devices(element)
    elif isinstance(object, dict):
        for value in object.values():
            devices |= get_devices(value)
    return devices


def get_tensors(object):
    tensors = []
    if isinstance(object, ttnn.Tensor):
        tensors.append(object)
    elif isinstance(object, (list, tuple)):
        for element in object:
            tensors += get_tensors(element)
    elif isinstance(object, dict):
        for value in object.values():
            tensors += get_tensors(value)
    return tensors


REGISTERED_OPERATIONS = set()


def query_operations(include_experimental=False):
    sorted_operations = sorted(REGISTERED_OPERATIONS)

    ttnn_operations = [
        operation
        for operation in sorted_operations
        if operation.name.startswith("ttnn.") and not operation.name.startswith("ttnn.experimental.")
    ]
    ttl_operations = [operation for operation in sorted_operations if operation.name.startswith("ttnn.experimental.")]
    if include_experimental:
        return ttnn_operations + ttl_operations
    else:
        return ttnn_operations


OPERATION_ID = 0
OPERATION_CALL_STACK = []


@dataclasses.dataclass
class Operation:
    name: str
    function: callable
    validate_input_tensors: callable
    compute_golden: callable
    is_cpp_function: bool
    fallback: callable

    def __gt__(self, other):
        return self.name < other.name

    def __hash__(self):
        return hash(self.name)

    def __post_init__(self):
        function = self.function

        if not self.is_cpp_function or self.fallback is not None:
            document_input_tensors(self.name, function, self.validate_input_tensors)

        def validate_decorator(function):
            @wraps(function)
            def call_wrapper(*function_args, **function_kwargs):
                if not self.is_cpp_function:
                    self.validate_input_tensors(self.name, *function_args, **function_kwargs)
                return function(*function_args, **function_kwargs)

            return call_wrapper

        def run_and_compare(function, *function_args, **function_kwargs):
            if self.compute_golden is not None:
                logger.debug(f"{self.name}: Comparing against PyTorch")
                golden_output = self.compute_golden(*function_args, **function_kwargs)
            else:
                logger.debug(f"{self.name}: Skipping comparison against PyTorch because compute_golden is not provided")
                golden_output = None

            output = function(*function_args, **function_kwargs)

            matches = None
            actual_pcc = None
            if golden_output is not None:
                matches, actual_pcc = calculate_pcc(golden_output, output, desired_pcc=ttnn.CONFIG.comparison_mode_pcc)

            return output, matches, actual_pcc

        def fallback_decorator(function):
            @wraps(function)
            def call_wrapper(*function_args, **function_kwargs):
                used_fallback = False
                try:
                    output = function(*function_args, **function_kwargs)
                except NotImplementedError:
                    used_fallback = True
                    logger.warning(f"{self.name}: falling back to CPU due to NotImplementedError")
                    output = self.fallback(*function_args, **function_kwargs)
                except Exception as e:
                    used_fallback = True
                    exception_message = "\n".join(str(e).split("\n")[:3])
                    logger.warning(f"{self.name}: falling back to CPU due to {exception_message}")
                    output = self.fallback(*function_args, **function_kwargs)

                if ttnn.CONFIG.throw_exception_on_fallback and used_fallback:
                    raise RuntimeError(f"Fallbacks are disabled, but {self.name} used a fallback")
                return output

            return call_wrapper

        if self.fallback is not None:
            function = fallback_decorator(function)

        def runtime_decorator(function):
            @wraps(function)
            def call_wrapper(*function_args, **function_kwargs):
                operation_id = OPERATION_ID
                is_top_level_operation = len(OPERATION_CALL_STACK) == 1

                if is_top_level_operation and ttnn.CONFIG.enable_logging and ttnn.CONFIG.enable_graph_report:
                    if not ttnn.tracer.is_tracing_enabled():
                        ttnn.tracer.enable_tracing()

                decorated_function = validate_decorator(function)

                if is_top_level_operation and ttnn.tracer.ENABLE_TRACER:
                    decorated_function = ttnn.tracer.trace_ttnn_operation(self.name, decorated_function)

                if is_top_level_operation:
                    for hook in PRE_OPERATION_HOOKS:
                        hook_return_value = hook(self, function_args, function_kwargs)
                        if hook_return_value is not None:
                            raise RuntimeError(
                                f"Pre-operation hook {hook} returned {hook_return_value} but must return None"
                            )

                if is_top_level_operation and ttnn.CONFIG.enable_logging:
                    start = time.time()
                    logger.debug(f"Started {self.name:50}")

                    input_tensors = get_tensors((function_args, function_kwargs))
                    ttnn.database.insert_input_tensors(operation_id, input_tensors)
                    if ttnn.CONFIG.enable_tensor_report:
                        (ttnn.CONFIG.reports_path / "input_tensors" / f"{operation_id}").mkdir(
                            parents=True, exist_ok=True
                        )
                        for index, tensor in enumerate(input_tensors):
                            ttnn.dump_tensor(
                                ttnn.CONFIG.reports_path / "input_tensors" / f"{operation_id}" / f"{index}.bin",
                                ttnn.from_device(tensor),
                            )

                if is_top_level_operation and ttnn.CONFIG.enable_comparison_mode:
                    output, matches_golden, actual_pcc = run_and_compare(
                        decorated_function, *function_args, **function_kwargs
                    )
                else:
                    matches_golden = None
                    actual_pcc = None
                    output = decorated_function(*function_args, **function_kwargs)

                if is_top_level_operation and ttnn.CONFIG.enable_logging:
                    devices = get_devices((function_args, function_kwargs))
                    for device in devices:
                        ttnn.synchronize_device(device)

                    end = time.time()
                    duration = end - start
                    logger.debug(f"Finished {self.name:50} in {duration:30} seconds")

                    ttnn.database.insert_devices(devices)

                    output_tensors = get_tensors(output)

                    ttnn.database.insert_operation(
                        self, operation_id, duration, matches_golden, ttnn.CONFIG.comparison_mode_pcc, actual_pcc
                    )
                    ttnn.database.insert_output_tensors(operation_id, output_tensors)
                    ttnn.database.insert_buffers(operation_id)

                    if ttnn.CONFIG.enable_graph_report:
                        ttnn.tracer.visualize(
                            ttnn.tracer.GRAPH_STACK[-1],
                            file_name=ttnn.CONFIG.reports_path / "graphs" / f"{operation_id}.svg",
                        )

                        """
                        codegen_reports = ttnn.CONFIG.reports_path / "codegen"
                        codegen_reports.mkdir(parents=True, exist_ok=True)
                        with open(ttnn.CONFIG.reports_path / "codegen" / f"{operation_id}.py", "w") as f:
                            f.write(ttnn.tracer.codegen(output))
                        """

                    if ttnn.CONFIG.enable_tensor_report:
                        (ttnn.CONFIG.reports_path / "output_tensors" / f"{operation_id}").mkdir(
                            parents=True, exist_ok=True
                        )
                        for index, tensor in enumerate(output_tensors):
                            ttnn.dump_tensor(
                                ttnn.CONFIG.reports_path / "output_tensors" / f"{operation_id}" / f"{index}.bin",
                                ttnn.from_device(tensor),
                            )

                if is_top_level_operation:
                    for hook in POST_OPERATION_HOOKS:
                        hook_return_value = hook(self, function_args, function_kwargs, output)
                        if hook_return_value is not None:
                            raise RuntimeError(
                                f"Post-operation hook {hook} returned {hook_return_value} but must return None"
                            )

                if matches_golden is not None and not matches_golden:
                    logger.error(f"{self.name}: Comparing against PyTorch failed")

                return output

            return call_wrapper

        if not ttnn.CONFIG.enable_fast_runtime_mode:
            # If fast runtime mode is enabled during import time, then don't decorate the original function
            function = runtime_decorator(function)

        self.decorated_function = function

    def __call__(self, *function_args, **function_kwargs):
        # If fast runtime mode is enabled during runtime, then only run the original function
        if ttnn.CONFIG.enable_fast_runtime_mode:
            return self.function(*function_args, **function_kwargs)

        global OPERATION_ID
        try:
            OPERATION_CALL_STACK.append(self.name)
            output = self.decorated_function(*function_args, **function_kwargs)
        finally:
            OPERATION_CALL_STACK.pop()
            OPERATION_ID += 1
        return output

    __doc__ = property(lambda self: self.decorated_function.__doc__)


def register_operation(
    *, name, validate_input_tensors=None, compute_golden=None, is_cpp_function=False, fallback=None, is_method=False
):
    if is_cpp_function:
        if fallback is not None:
            if validate_input_tensors is None:
                raise RuntimeError(
                    f"Registering {name}: validate_input_tensors is required for cpp functions with fallbacks"
                )
        else:
            if validate_input_tensors is not None:
                raise RuntimeError(
                    f"Registering {name}: validate_input_tensors is not supported for cpp functions without fallbacks because the input tensors are validated in C++"
                )
    else:
        if validate_input_tensors is None:
            raise RuntimeError(f"Registering {name}: validate_input_tensors is required for non-cpp functions")

    def operation_decorator(function: callable):
        global REGISTERED_OPERATIONS

        operation = Operation(
            name=name,
            function=function,
            validate_input_tensors=validate_input_tensors,
            compute_golden=compute_golden,
            is_cpp_function=is_cpp_function,
            fallback=fallback,
        )

        if operation in REGISTERED_OPERATIONS:
            raise RuntimeError(f"{operation} is already registered")
        REGISTERED_OPERATIONS.add(operation)

        if is_method:

            @wraps(operation)
            def method_call(self, *args, **kwargs):
                return operation(self, *args, **kwargs)

            return method_call

        return operation

    return operation_decorator


def register_ttl_operation_as_ttnn_operation(name, function):
    function = register_operation(
        name=name,
        is_cpp_function=True,
    )(function)
    return function
