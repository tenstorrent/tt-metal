// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::moreh::moreh_bmm {
struct MorehBMM {
    static Tensor invoke(
        const Tensor& input,
        const Tensor& mat2,
        const std::optional<Tensor>& output,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_bmm

namespace ttnn {
constexpr auto moreh_bmm = ttnn::register_operation<"ttnn::moreh_bmm", ttnn::operations::moreh::moreh_bmm::MorehBMM>();
}  // namespace ttnn
