// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotate_half.hpp"

#include "device/rotate_half_device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
// TODO(nuked-op tilize_with_val_padding): header removed

namespace ttnn::experimental {

Tensor rotate_half(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    using namespace tt::constants;
    using tt::tt_metal::PadValue;

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Input tensor must be on device. Current storage type: {}.",
        static_cast<int>(input_tensor.storage_type()));

    TT_FATAL(
        input_tensor.padded_shape()[-1] % (TILE_WIDTH * 2) == 0,
        "Input X dimension ({}) must be divisible by {} for tiling.",
        input_tensor.padded_shape()[-1],
        TILE_WIDTH * 2);

    auto padded_shape = ttnn::operations::data_movement::pad_to_tile_shape(input_tensor.padded_shape());
    (void)padded_shape;
    // TODO(nuked-op tilize_with_val_padding): restore real call
    Tensor formatted_input = input_tensor;
    return ttnn::prim::rotate_half(formatted_input, memory_config.value_or(input_tensor.memory_config()));
}

}  // namespace ttnn::experimental
