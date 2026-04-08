# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

SlidingWindowParallelConfig = ttnn._ttnn.operations.sliding_window.ParallelConfig

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


__all__ = []
