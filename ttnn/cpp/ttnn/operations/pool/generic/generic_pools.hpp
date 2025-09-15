// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/core.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"

#include "device/pool_op.hpp"

namespace ttnn {
namespace operations::pool {

struct MaxPoolWithIndicesResult {
    Tensor output;
    Tensor indices;
};

struct MaxPool2DOp {
    static std::variant<Tensor, MaxPoolWithIndicesResult> invoke(
        const Tensor& input_tensor,
        uint32_t batch_size,
        uint32_t input_h,
        uint32_t input_w,
        uint32_t channels,
        std::array<uint32_t, 2> kernel_size,
        std::array<uint32_t, 2> stride,
        std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
        std::array<uint32_t, 2> dilation,
        bool ceil_mode = false,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        std::optional<const TensorMemoryLayout> applied_shard_scheme = std::nullopt,
        bool in_place_halo = false,
        bool deallocate_input = false,
        bool reallocate_halo_output = true,
        bool return_indices = false,
        DataType dtype = DataType::BFLOAT16,
        Layout output_layout = Layout::ROW_MAJOR);
};
struct AvgPool2DOp {
    static Tensor invoke(
        const Tensor& input_tensor,
        uint32_t batch_size,
        uint32_t input_h,
        uint32_t input_w,
        uint32_t channels,
        std::array<uint32_t, 2> kernel_size,
        std::array<uint32_t, 2> stride,
        std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
        bool ceil_mode = false,
        bool count_include_pad = true,
        std::optional<int32_t> divisor_override = std::nullopt,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        std::optional<const TensorMemoryLayout> applied_shard_scheme = std::nullopt,
        bool in_place_halo = false,
        bool deallocate_input = false,
        bool reallocate_halo_output = true,
        DataType dtype = DataType::BFLOAT16,
        Layout output_layout = Layout::ROW_MAJOR);
};

}  // namespace operations::pool

constexpr auto max_pool2d = ttnn::register_operation<"ttnn::max_pool2d", operations::pool::MaxPool2DOp>();
constexpr auto avg_pool2d = ttnn::register_operation<"ttnn::avg_pool2d", operations::pool::AvgPool2DOp>();

}  // namespace ttnn
