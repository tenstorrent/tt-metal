// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotate_half.hpp"

#include "device/rotate_half_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"

namespace ttnn::operations::experimental::transformer {

Tensor RotateHalfOperation::invoke(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    using namespace tt::constants;

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Input tensor must be on device. Current storage type: {}.",
        static_cast<int>(input_tensor.storage_type()));

    TT_FATAL(
        input_tensor.padded_shape()[-1] % (TILE_WIDTH * 2) == 0,
        "Input X dimension ({}) must be divisible by {} for tiling.",
        input_tensor.padded_shape()[-1],
        TILE_WIDTH * 2);

    using namespace ttnn::operations::experimental::auto_format;
    Tensor formatted_input = AutoFormat::format_input_tensor(input_tensor, 0, Layout::TILE);
    return tt::tt_metal::operation::run(
               RotateHalf{memory_config.value_or(input_tensor.memory_config())}, {formatted_input})
        .at(0);
}

}  // namespace ttnn::operations::experimental::transformer
