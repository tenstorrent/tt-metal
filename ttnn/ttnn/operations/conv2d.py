# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

from typing import Tuple, Union, Dict, Optional
import warnings
import math
import ttnn
from ttnn.operations.activations import get_golden_function_for_activation

SlidingWindowParallelConfig = ttnn._ttnn.operations.sliding_window.ParallelConfig
Conv2dConfig = ttnn._ttnn.operations.conv.Conv2dConfig
PaddingMode = ttnn._ttnn.operations.conv.PaddingMode

# TODO: Remove Conv2dSliceConfig and update all relevant models & tests
Conv2dSliceConfig = ttnn._ttnn.operations.sliding_window.Op2DSliceConfig
Conv2dDRAMSliceHeight = Conv2dSliceConfig.SliceTypeEnum.DRAMSliceHeight
Conv2dDRAMSliceWidth = Conv2dSliceConfig.SliceTypeEnum.DRAMSliceWidth
Conv2dL1Full = Conv2dSliceConfig.SliceTypeEnum.L1Full
Conv2dL1FullSliceConfig = Conv2dSliceConfig(slice_type=Conv2dL1Full)

Op2DSliceConfig = ttnn._ttnn.operations.sliding_window.Op2DSliceConfig
Op2DDRAMSliceHeight = Op2DSliceConfig.SliceTypeEnum.DRAMSliceHeight
Op2DDRAMSliceWidth = Op2DSliceConfig.SliceTypeEnum.DRAMSliceWidth
Op2DL1Full = Op2DSliceConfig.SliceTypeEnum.L1Full
Op2DL1FullSliceConfig = Op2DSliceConfig(slice_type=Op2DL1Full)


def get_conv_output_dim(input, window, stride=1, pad=0, dilation=1):
    """
    Returns the output dimension of a convolution operation.
    """
    return (input + (2 * pad) - dilation * (window - 1) - 1) // stride + 1


# NOTE: the conv2d op and the prepare_conv_weights / prepare_conv_bias /
# prepare_conv_transpose2d_* op wrappers + golden function were nuked for the
# agent-regen baseline. Conv2dConfig / PaddingMode / get_conv_output_dim and the
# slice-config aliases above are retained as shared infra.

__all__ = []
