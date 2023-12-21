# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger


def compare(torch_outputs, outputs, pcc):
    import ttnn
    import torch
    from tests.ttnn.utils_for_testing import assert_with_pcc

    if isinstance(torch_outputs, torch.Tensor):
        torch_outputs = [torch_outputs]
    assert isinstance(torch_outputs, (list, tuple))

    if isinstance(outputs, ttnn.Tensor):
        outputs = [outputs]
    assert isinstance(outputs, (list, tuple))

    for torch_output, output in zip(torch_outputs, outputs):
        shape = torch_output.shape
        slices = [slice(0, dim) for dim in shape]

        output = ttnn.from_device(output)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
        output = ttnn.to_torch(output)
        output = output[slices]

        assert_with_pcc(torch_output, output, pcc)


ENABLE_DEBUG_DECORATOR = False


def debug_decorator(torch_function=None, pcc=0.99, name=None):
    def decorator(function):
        function_name = name or function.__name__

        def call_wrapper(*args, **kwargs):
            if not ENABLE_DEBUG_DECORATOR:
                return function(*args, **kwargs)

            logger.info(function_name, ": Comparing with PyTorch" if torch_function is not None else "")
            outputs = function(*args, **kwargs)
            if torch_function is not None:
                torch_outputs = torch_function(*args, **kwargs)
                compare(torch_outputs, outputs, pcc)
            return outputs

        return call_wrapper

    return decorator
