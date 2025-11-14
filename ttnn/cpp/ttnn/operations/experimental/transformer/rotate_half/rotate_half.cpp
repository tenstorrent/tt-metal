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

    // Format input: device operation requires TILE layout, but API accepts both TILE and ROW_MAJOR
    auto format_input = [](const Tensor& input) -> Tensor {
        if (input.layout() == Layout::TILE) {
            return input;
        } else {
            Shape tile_aligned_shape =
                data_movement::compute_padded_shape(input.padded_shape(), TILE_HEIGHT, TILE_WIDTH);

            PadValue pad_value_variant;
            if (input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::FLOAT32) {
                pad_value_variant = 0.0f;
            } else {
                pad_value_variant = (uint32_t)0;
            }
            return ttnn::tilize_with_val_padding(input, tile_aligned_shape, pad_value_variant, input.memory_config());
        }
    };

    Tensor formatted_input = format_input(input_tensor);

    return tt::tt_metal::operation::run(
               RotateHalf{memory_config.value_or(input_tensor.memory_config())}, {formatted_input})
        .at(0);
}

}  // namespace ttnn::operations::experimental::transformer
