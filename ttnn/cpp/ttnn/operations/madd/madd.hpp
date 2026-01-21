// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::madd {

struct MAdd {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& a,
        const ttnn::Tensor& b,
        const ttnn::Tensor& c,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);
};
}  // namespace ttnn::operations::madd

namespace ttnn {
constexpr auto madd = ttnn::register_operation<"ttnn::madd", ttnn::operations::madd::MAdd>();
}  // namespace ttnn
