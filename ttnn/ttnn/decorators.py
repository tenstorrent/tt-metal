# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
import dataclasses
from functools import wraps
import inspect
import time

from loguru import logger

import ttnn


def compare(torch_outputs, outputs, pcc):
    import torch

    from models.utility_functions import comp_pcc

    if isinstance(outputs, ttnn.Tensor):
        if not isinstance(torch_outputs, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(torch_outputs)}")
        outputs = [outputs]
        torch_outputs = [torch_outputs]
    else:
        if not isinstance(outputs, (list, tuple)):
            raise TypeError(f"Expected list or tuple, got {type(outputs)}")
        if not isinstance(torch_outputs, (list, tuple)):
            raise TypeError(f"Expected list or tuple, got {type(torch_outputs)}")

    matches = True
    last_message = None
    for torch_output, output in zip(torch_outputs, outputs):
        output = ttnn.to_torch(output)
        passed, last_message = comp_pcc(torch_output, output, pcc)
        matches &= passed
    return matches, last_message


ENABLE_VALIDATE_DECORATOR = True
ENABLE_DEBUG_DECORATOR = False
PEARSON_CORRELATION_COEFFICIENT = 0.9999
USE_TORCH_OUTPUT_IF_MISMATCHES = False


@contextmanager
def disable_validate_decorator():
    global ENABLE_VALIDATE_DECORATOR
    ENABLE_VALIDATE_DECORATOR = False
    yield
    ENABLE_VALIDATE_DECORATOR = True


@contextmanager
def enable_debug_decorator():
    ttnn.decorators.ENABLE_DEBUG_DECORATOR = True
    yield
    ttnn.decorators.ENABLE_DEBUG_DECORATOR = False


@contextmanager
def override_pcc_of_debug_decorator(value):
    global PEARSON_CORRELATION_COEFFICIENT
    old_value = PEARSON_CORRELATION_COEFFICIENT
    PEARSON_CORRELATION_COEFFICIENT = value
    yield
    PEARSON_CORRELATION_COEFFICIENT = old_value


def convert_torch_output_to_be_like_ttnn_output(torch_output, output):
    torch_output = ttnn.from_torch(torch_output, dtype=output.dtype, layout=output.layout)
    if ttnn.is_tensor_storage_on_device(output):
        torch_output = ttnn.to_device(torch_output, output.device())
    return torch_output


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


def get_devices(arg):
    devices = set()
    if isinstance(arg, ttnn.Tensor):
        if ttnn.is_tensor_storage_on_device(arg) and arg.is_allocated():
            devices.add(arg.device())
    elif isinstance(arg, (list, tuple)):
        for element in arg:
            devices |= get_devices(element)
    elif isinstance(arg, dict):
        for value in arg.values():
            devices |= get_devices(value)
    return devices


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
    torch_function: callable
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

        def debug_decorator(function):
            @wraps(function)
            def call_wrapper(*function_args, **function_kwargs):
                if self.torch_function is not None:
                    logger.info(f"{self.name} : Comparing against PyTorch")
                    torch_output = self.torch_function(*function_args, **function_kwargs)
                else:
                    logger.info(
                        f"{self.name} : Skipping comparison against PyTorch because torch_function is not provided"
                    )
                    torch_output = None

                output = function(*function_args, **function_kwargs)

                if torch_output is not None:
                    matches, last_message = compare(torch_output, output, pcc=PEARSON_CORRELATION_COEFFICIENT)
                    if not matches:
                        if USE_TORCH_OUTPUT_IF_MISMATCHES:
                            logger.warning(f"{self.name}: Comparing against PyTorch failed, using PyTorch output")
                            if not isinstance(output, ttnn.Tensor):
                                raise TypeError(f"Expected Tensor, got {type(output)}")
                            output = convert_torch_output_to_be_like_ttnn_output(torch_output, output)
                        else:
                            if isinstance(output, ttnn.Tensor):
                                output = ttnn.to_torch(output)
                            elif isinstance(output, (list, tuple)):
                                output = [ttnn.to_torch(tensor) for tensor in output]
                            else:
                                raise TypeError(f"Expected Tensor, list or tuple, got {type(output)}")
                            raise RuntimeError(
                                f"{self.name}: Comparing against PyTorch failed with: {last_message} compared: {torch_output} vs {output}"
                            )

                return output

            return call_wrapper

        def fallback_decorator(function):
            @wraps(function)
            def call_wrapper(*function_args, **function_kwargs):
                try:
                    return function(*function_args, **function_kwargs)
                except NotImplementedError:
                    logger.warning(f"{self.name}: falling back to torch due to NotImplementedError")
                    return self.fallback(*function_args, **function_kwargs)
                except Exception as e:
                    exception_message = "\n".join(str(e).split("\n")[:3])
                    logger.warning(f"{self.name}: falling back to torch due to {exception_message}")
                    return self.fallback(*function_args, **function_kwargs)

            return call_wrapper

        if self.fallback is not None:
            function = fallback_decorator(function)

        def runtime_decorator(function):
            @wraps(function)
            def call_wrapper(*function_args, **function_kwargs):
                is_top_level_operation = len(OPERATION_CALL_STACK) == 1

                decorated_function = function
                if ENABLE_VALIDATE_DECORATOR:
                    decorated_function = validate_decorator(decorated_function)

                if ENABLE_DEBUG_DECORATOR:
                    decorated_function = debug_decorator(decorated_function)

                if is_top_level_operation and ttnn.tracer.ENABLE_TRACER:
                    decorated_function = ttnn.tracer.trace_ttnn_operation(self.name, decorated_function)

                if is_top_level_operation:
                    for hook in PRE_OPERATION_HOOKS:
                        hook_return_value = hook(self, function_args, function_kwargs)
                        if hook_return_value is not None:
                            raise RuntimeError(
                                f"Pre-operation hook {hook} returned {hook_return_value} but must return None"
                            )

                if is_top_level_operation and ttnn.ENABLE_LOGGING:
                    start = time.time()
                    logger.info(f"Started {self.name:50}")

                output = decorated_function(*function_args, **function_kwargs)

                if is_top_level_operation and ttnn.ENABLE_LOGGING:
                    for device in get_devices((function_args, function_kwargs)):
                        ttnn.synchronize_device(device)
                    end = time.time()
                    duration = end - start
                    logger.info(f"Finished {self.name:50} in {duration:30} seconds")

                    if ttnn.SQLITE_CONNECTION is not None:
                        cursor = ttnn.SQLITE_CONNECTION.cursor()
                        cursor.execute(f"INSERT INTO operations VALUES ({OPERATION_ID}, '{self.name}')")
                        for buffer_page in ttnn.get_buffer_pages():
                            cursor.execute(
                                f"INSERT INTO buffers VALUES ({OPERATION_ID}, {buffer_page.address}, {buffer_page.device_id}, {buffer_page.core_y}, {buffer_page.core_x}, {buffer_page.page_index}, {buffer_page.page_address}, {buffer_page.page_size}, {buffer_page.buffer_type.value})"
                            )
                        ttnn.SQLITE_CONNECTION.commit()

                if is_top_level_operation:
                    for hook in POST_OPERATION_HOOKS:
                        hook_return_value = hook(self, function_args, function_kwargs, output)
                        if hook_return_value is not None:
                            raise RuntimeError(
                                f"Post-operation hook {hook} returned {hook_return_value} but must return None"
                            )

                return output

            return call_wrapper

        if not ttnn.ENABLE_FAST_RUNTIME_MODE:
            function = runtime_decorator(function)

        self.decorated_function = function

    def __call__(self, *function_args, **function_kwargs):
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
    *, name, validate_input_tensors=None, torch_function=None, is_cpp_function=False, fallback=None, is_method=False
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
            torch_function=torch_function,
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
