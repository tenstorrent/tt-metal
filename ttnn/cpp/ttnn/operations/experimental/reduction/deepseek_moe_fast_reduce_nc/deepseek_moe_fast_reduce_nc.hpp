// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::experimental::reduction {

struct DeepseekMoEFastReduceNCOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input,
        tt::stl::Span<const int32_t> dims,
        const std::optional<const ttnn::Tensor>& output,
        const ttnn::MemoryConfig& memory_config,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config);
};

}  // namespace operations::experimental::reduction

namespace experimental::reduction {

constexpr auto deepseek_moe_fast_reduce_nc = ttnn::register_operation<
    "ttnn::experimental::deepseek_moe_fast_reduce_nc",
    ttnn::operations::experimental::reduction::DeepseekMoEFastReduceNCOperation>();

}  // namespace experimental::reduction
}  // namespace ttnn
