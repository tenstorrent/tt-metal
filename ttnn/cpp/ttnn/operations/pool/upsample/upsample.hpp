// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace upsample {

struct ExecuteUpSample {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        std::variant<int, tt::tt_metal::Array2D> scale_factor,
        const std::string& mode = std::string("nearest"),
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);
};
}  // namespace upsample
}  // namespace operations
constexpr auto upsample = ttnn::register_operation<"ttnn::upsample", ttnn::operations::upsample::ExecuteUpSample>();
}  // namespace ttnn
