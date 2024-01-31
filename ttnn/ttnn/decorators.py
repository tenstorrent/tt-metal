# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from functools import wraps
import inspect

from loguru import logger

import ttnn.core as ttnn


ENABLE_VALIDATE_DECORATOR = True
ENABLE_DEBUG_DECORATOR = False
USE_TORCH_OUTPUT_IF_MISMATCHES = False


@contextmanager
def disable_validate_decorator():
    global ENABLE_VALIDATE_DECORATOR
    ENABLE_VALIDATE_DECORATOR = False
    yield
    ENABLE_VALIDATE_DECORATOR = True


PEARSON_CORRELATION_COEFFICIENT = 0.9999


@contextmanager
def override_pearson_correlation_coefficient(value):
    global PEARSON_CORRELATION_COEFFICIENT
    old_value = PEARSON_CORRELATION_COEFFICIENT
    PEARSON_CORRELATION_COEFFICIENT = value
    yield
    PEARSON_CORRELATION_COEFFICIENT = old_value


def convert_torch_output_to_be_like_ttnn_output(torch_output, output):
    torch_output = ttnn.from_torch(torch_output, dtype=output.dtype, layout=output.layout)
    if ttnn.has_storage_type_of(output, ttnn.DEVICE_STORAGE_TYPE):
        torch_output = ttnn.to_device(torch_output, output.device)
    return torch_output
