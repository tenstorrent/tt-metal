// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotate_half.hpp"

#include "device/rotate_half_device_operation.hpp"

namespace ttnn::operations::transformer {

Tensor RotateHalfOperation::operator()(const Tensor& input_tensor, const MemoryConfig& output_mem_config)
{
    TT_FATAL(input_tensor.get_legacy_shape()[-1] % (TILE_WIDTH * 2) == 0, "Input X dim must be divisible into tiles");
    tt::tt_metal::Shape pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape());
    FormatParams input_format_params = {.pad_shape=pad_shape, .pad_value=0.0, .target_layout=Layout::TILE};
    return operation::run_with_autoformat(RotateHalf{output_mem_config}, {input_tensor}, {input_format_params}, {Layout::TILE}).at(0);
}


}  // namespace ttnn::operations::transformer
