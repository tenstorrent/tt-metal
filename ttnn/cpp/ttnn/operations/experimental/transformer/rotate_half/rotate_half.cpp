// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotate_half.hpp"

#include "device/rotate_half_device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

Tensor RotateHalfOperation::invoke(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config)
{
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE,
            fmt::format("Input tensor must be on device. Current storage type: {}.",
                        static_cast<int>(input_tensor.storage_type())));

    TT_FATAL(input_tensor.get_legacy_shape()[-1] % (TILE_WIDTH * 2) == 0,
            fmt::format("Input X dimension ({}) must be divisible by {} for tiling.",
                        input_tensor.get_legacy_shape()[-1],
                        TILE_WIDTH * 2));

    tt::tt_metal::Shape pad_shape = ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape());
    ttnn::operations::experimental::auto_format::FormatParams input_format_params = {.pad_shape=pad_shape, .pad_value=0.0, .target_layout=Layout::TILE};
    return operation::run_with_autoformat(RotateHalf{memory_config.value_or(input_tensor.memory_config())}, {input_tensor}, {input_format_params}, {Layout::TILE}).at(0);
}


}  // namespace ttnn::operations::experimental::transformer
