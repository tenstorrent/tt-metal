// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_pad.hpp"
#include "device/fill_pad_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include <utility>
#include "ttnn/operations/copy/typecast/typecast.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor FillPadOperation::invoke(
    const ttnn::Tensor& input_tensor, float fill_value, const std::optional<ttnn::MemoryConfig>& memory_config) {
    // if padded shape == logical shape for last 2 dims no padding should be present, and no fill pad is necessary
    uint32_t padded_height =
        tt::div_up(input_tensor.logical_shape()[-2], tt::constants::TILE_HEIGHT) * tt::constants::TILE_HEIGHT;
    uint32_t padded_width =
        tt::div_up(input_tensor.logical_shape()[-1], tt::constants::TILE_HEIGHT) * tt::constants::TILE_HEIGHT;
    if (padded_width == input_tensor.logical_shape()[-1] && padded_height == input_tensor.logical_shape()[-2]) {
        return input_tensor;
    }
    auto mutable_input_tensor = input_tensor;
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
    if (input_tensor.dtype() == DataType::BFLOAT8_B) {
        mutable_input_tensor = ttnn::typecast(mutable_input_tensor, DataType::BFLOAT16);
    }
    // if input_tensor is rank > 3, then we need to reshape it to rank 3 such that the last 2 dims are the same
    if (mutable_input_tensor.logical_shape().rank() > 3) {
        ttnn::Shape original_shape = mutable_input_tensor.logical_shape();

        uint32_t third_dim = 1;
        for (uint32_t i = 0; i < original_shape.rank() - 2; i++) {
            third_dim *= original_shape[i];
        }

        ttnn::Shape new_shape = ttnn::Shape{std::array<uint32_t, 3>{third_dim, original_shape[-2], original_shape[-1]}};
        auto reshaped_tensor = ttnn::reshape(mutable_input_tensor, new_shape);

        reshaped_tensor = ttnn::prim::fill_pad(reshaped_tensor, fill_value, output_memory_config);
        return ttnn::reshape(reshaped_tensor, original_shape);
    }
    auto output_tensor = ttnn::prim::fill_pad(mutable_input_tensor, fill_value, output_memory_config);
    if (input_tensor.dtype() == DataType::BFLOAT8_B) {
        return ttnn::typecast(output_tensor, DataType::BFLOAT8_B);
    }
    return output_tensor;
}

}  // namespace ttnn::operations::data_movement
