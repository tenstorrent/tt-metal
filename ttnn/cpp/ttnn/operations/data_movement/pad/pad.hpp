// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include <ranges>
#include "ttnn/decorators.hpp"


namespace ttnn {
namespace operations::data_movement {

// We overload over Array1D-8D
#define PAD_OVERLOAD_DIM(ShapeType) \
    static ttnn::Tensor invoke(uint8_t, const ttnn::Tensor&, const ShapeType&, const ShapeType&, const float, const bool, const std::optional<MemoryConfig>&); \
    static ttnn::Tensor invoke(const ttnn::Tensor&, const ShapeType&, const ShapeType&, const float, const std::optional<MemoryConfig>&); \
    static ttnn::Tensor invoke(const ttnn::Tensor&, const ShapeType&, const ShapeType&, const float);

struct ExecutePad {
    PAD_OVERLOAD_DIM(tt::tt_metal::Array1D)
    PAD_OVERLOAD_DIM(tt::tt_metal::Array2D)
    PAD_OVERLOAD_DIM(tt::tt_metal::Array3D)
    PAD_OVERLOAD_DIM(tt::tt_metal::Array4D)
    PAD_OVERLOAD_DIM(tt::tt_metal::Array5D)
    PAD_OVERLOAD_DIM(tt::tt_metal::Array6D)
    PAD_OVERLOAD_DIM(tt::tt_metal::Array7D)
    PAD_OVERLOAD_DIM(tt::tt_metal::Array8D)

    // This function signature is similar to pytorch's signature
    // Any rank tensor supported
    static ttnn::Tensor invoke(uint8_t queue_id,
                                   const ttnn::Tensor& input_tensor,
                                   const std::vector<std::pair<uint32_t, uint32_t>>& padding,
                                   const float value,
                                   const bool use_multicore,
                                   const std::optional<MemoryConfig>& memory_config_arg);
};

}  // namespace operations::data_movement

constexpr auto pad = ttnn::register_operation_with_auto_launch_op<"ttnn::pad", ttnn::operations::data_movement::ExecutePad>();

}  // namespace ttnn
