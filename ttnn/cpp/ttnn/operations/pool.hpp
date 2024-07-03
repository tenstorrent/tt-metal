// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tt_dnn/op_library/pool/average_pool.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core.hpp"

namespace ttnn {
namespace operations {
namespace pool {

namespace detail {
inline const std::array<ttnn::TensorSchema, 1> input_tensor_schemas() {
    return {ttnn::TensorSchema{
        4,  // min rank
        4,  // max rank
        {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::uint16, ttnn::uint32},
        {ttnn::TILE_LAYOUT},
        true,   // can_be_on_device
        false,  // can_be_on_cpu
        false,  // can_be_scalar
        false   // is_optional}
    }};
}
}  // namespace detail

struct GlobalAveragePool2D {
    static const std::array<TensorSchema, 1> input_tensor_schemas() { return detail::input_tensor_schemas(); }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return std::forward_as_tuple(input_tensor);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& input,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<DataType>& output_dtype = std::nullopt) {
        auto memory_config = memory_config_arg.value_or(input.memory_config());
        auto result = tt::tt_metal::average_pool_2d(input, memory_config, output_dtype);
        return result;
    }
};
}  // namespace pool
}  // namespace operations

constexpr auto global_avg_pool2d =
    ttnn::register_operation<ttnn::operations::pool::GlobalAveragePool2D>("ttnn::global_avg_pool2d");

}  // namespace ttnn
