// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotate_half.hpp"

#include <boost/container/vector.hpp>
#include <fmt/base.h>
#include <algorithm>
#include <vector>

#include "device/rotate_half_device_operation.hpp"
#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer_constants.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/shape_base.hpp>
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/shape/shape.hpp"

namespace ttnn::operations::experimental::transformer {

Tensor RotateHalfOperation::invoke(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Input tensor must be on device. Current storage type: {}.",
        static_cast<int>(input_tensor.storage_type()));

    TT_FATAL(
        input_tensor.get_padded_shape()[-1] % (tt::constants::TILE_WIDTH * 2) == 0,
        "Input X dimension ({}) must be divisible by {} for tiling.",
        input_tensor.get_padded_shape()[-1],
        tt::constants::TILE_WIDTH * 2);

    ttnn::Shape pad_shape =
        ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(input_tensor.get_padded_shape());
    ttnn::operations::experimental::auto_format::FormatParams input_format_params = {
        .pad_shape = pad_shape, .pad_value = 0.0, .target_layout = Layout::TILE};
    return tt::tt_metal::operation::run_with_autoformat(
               RotateHalf{memory_config.value_or(input_tensor.memory_config())},
               {input_tensor},
               {input_format_params},
               {Layout::TILE})
        .at(0);
}

}  // namespace ttnn::operations::experimental::transformer
