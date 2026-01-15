// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/sliding_window/op_slicing/op_slicing.hpp"
namespace ttnn {
namespace operations::pool {

using op_slicing::Op2DSliceConfig;

struct MaxPoolWithIndicesResult {
    Tensor output;
    Tensor indices;
};

struct MaxPool2DOp {
    static std::vector<Tensor> invoke(
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
        const std::optional<Op2DSliceConfig>& dram_slice_config = std::nullopt,
        std::optional<const TensorMemoryLayout> applied_shard_scheme = std::nullopt,
        bool deallocate_input = false,
        bool reallocate_halo_output = true,
        bool return_indices = false,
        DataType dtype = DataType::BFLOAT16,
        Layout output_layout = Layout::ROW_MAJOR,
        bool config_tensor_in_dram = false);
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
        const std::optional<Op2DSliceConfig>& dram_slice_config = std::nullopt,
        std::optional<const TensorMemoryLayout> applied_shard_scheme = std::nullopt,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
        bool deallocate_input = false,
        bool reallocate_halo_output = true,
        DataType dtype = DataType::BFLOAT16,
        Layout output_layout = Layout::ROW_MAJOR,
        bool config_tensor_in_dram = false);
};

}  // namespace operations::pool

constexpr auto max_pool2d = ttnn::register_operation<"ttnn::max_pool2d", operations::pool::MaxPool2DOp>();
constexpr auto avg_pool2d = ttnn::register_operation<"ttnn::avg_pool2d", operations::pool::AvgPool2DOp>();

}  // namespace ttnn
