"""Reference implementations of affine quantization operations."""

from __future__ import annotations

from typing import Any

import numpy as np


def _round(value: np.ndarray) -> np.ndarray:
    # numpy.rint is the usual tensor-kernel round-to-nearest-even operation.
    return np.rint(value)


def _saturate(values: np.ndarray, dtype: Any) -> np.ndarray:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.uint8):
        values = np.clip(values, 0, 255)
    elif dtype == np.dtype(np.int8):
        values = np.clip(values, -128, 127)
    else:
        raise TypeError("quantized output dtype must be np.uint8 or np.int8")
    return values.astype(dtype)


def quantize(input_tensor: Any, scale: Any, zero_point: Any = 0,
             dtype: Any = np.uint8) -> np.ndarray:
    """Quantize ``input_tensor`` using affine scale and zero point."""
    values = _round(np.asarray(input_tensor, dtype=np.float32) / scale + zero_point)
    return _saturate(values, dtype)


def requantize(input_tensor: Any, input_scale: Any, output_scale: Any,
               input_zero_point: Any = 0, output_zero_point: Any = 0,
               dtype: Any = np.uint8) -> np.ndarray:
    """Convert an already quantized/real tensor to a new affine scale."""
    values = _round(
        (np.asarray(input_tensor, dtype=np.float32) - input_zero_point)
        * input_scale / output_scale + output_zero_point
    )
    return _saturate(values, dtype)
