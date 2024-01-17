# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from loguru import logger


def compare(torch_outputs, outputs, pcc):
    import ttnn
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
        shape = torch_output.shape
        slices = [slice(0, dim) for dim in shape]

        output = ttnn.from_device(output)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
        output = ttnn.to_torch(output)
        output = output[slices]

        passed, last_message = comp_pcc(torch_output, output, pcc)
        matches &= passed
    return matches, last_message


ENABLE_DEBUG_DECORATOR = False
USE_TORCH_OUTPUT_IF_MISMATCHES = False


def convert_torch_output_to_be_like_ttnn_output(torch_output, output):
    import ttnn

    torch_output = ttnn.from_torch(torch_output, dtype=output.dtype, layout=output.layout)
    if ttnn.has_storage_type_of(output, ttnn.DEVICE_STORAGE_TYPE):
        torch_output = ttnn.to_device(torch_output, output.device)
    return torch_output


def decorate_operation(*, torch_function=None, pcc=0.99, name=None):
    def operation_decorator(function):
        function_name = name or function.__name__

        def debug_decorator(function):
            def call_wrapper(*function_args, **function_kwargs):
                if torch_function is not None:
                    logger.info(f"{function_name} : Comparing against PyTorch")

                if torch_function is not None:
                    torch_output = torch_function(*function_args, **function_kwargs)
                else:
                    torch_output = None

                output = function(*function_args, **function_kwargs)

                if torch_output is not None:
                    matches, last_message = compare(torch_output, output, pcc)
                    if not matches:
                        import ttnn

                        if USE_TORCH_OUTPUT_IF_MISMATCHES:
                            logger.warning(f"{function_name}: Comparing against PyTorch failed, using PyTorch output")
                            if not isinstance(output, ttnn.Tensor):
                                raise TypeError(f"Expected ttnn.Tensor, got {type(output)}")
                            output = convert_torch_output_to_be_like_ttnn_output(torch_output, output)
                        else:
                            output = ttnn.from_device(output)
                            output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
                            output = ttnn.to_torch(output)
                            raise RuntimeError(
                                f"{function_name}: Comparing against PyTorch failed with: {last_message} compared: {torch_output} vs {output}"
                            )

                return output

            return call_wrapper

        @wraps(function)
        def call_wrapper(*function_args, **function_kwargs):
            decorated_function = function
            if ENABLE_DEBUG_DECORATOR:
                decorated_function = debug_decorator(decorated_function)
            return decorated_function(*function_args, **function_kwargs)

        return call_wrapper

    return operation_decorator
