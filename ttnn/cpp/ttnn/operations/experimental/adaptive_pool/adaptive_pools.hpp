// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/core.hpp"
#include "ttnn/operations/sliding_window/op_slicing/op_slicing.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include "ttnn/operations/pool/generic/device/pool_op.hpp"

namespace ttnn::operations::experimental::adaptive_pool {
struct AdaptiveAvgPool2DOp {
    static Tensor invoke(
        const Tensor& input_tensor,
        uint32_t batch_size,
        uint32_t input_h,
        uint32_t input_w,
        uint32_t channels,
        std::array<uint32_t, 2> output_size,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        const std::optional<op_slicing::Op2DSliceConfig>& dram_slice_config = std::nullopt,
        std::optional<const TensorMemoryLayout> applied_shard_scheme = std::nullopt,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
        bool deallocate_input = false,
        bool reallocate_output = true);
};

struct AdaptiveMaxPool2DOp {
    static Tensor invoke(
        const Tensor& input_tensor,
        uint32_t batch_size,
        uint32_t input_h,
        uint32_t input_w,
        uint32_t channels,
        std::array<uint32_t, 2> output_size,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        const std::optional<op_slicing::Op2DSliceConfig>& dram_slice_config = std::nullopt,
        std::optional<const TensorMemoryLayout> applied_shard_scheme = std::nullopt,
        bool deallocate_input = false,
        bool reallocate_output = true);
};

}  // namespace ttnn::operations::experimental::adaptive_pool

namespace ttnn {
constexpr auto adaptive_avg_pool2d = ttnn::
    register_operation<"ttnn::adaptive_avg_pool2d", operations::experimental::adaptive_pool::AdaptiveAvgPool2DOp>();
constexpr auto adaptive_max_pool2d = ttnn::
    register_operation<"ttnn::adaptive_max_pool2d", operations::experimental::adaptive_pool::AdaptiveMaxPool2DOp>();

}  // namespace ttnn
