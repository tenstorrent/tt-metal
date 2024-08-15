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
        std::variant<int, tt::tt_metal::Array2D, tt::tt_metal::Array3D, tt::tt_metal::Array4D> scale_factor,
        std::optional<MemoryConfig> output_mem_config = std::nullopt);
};
} // upsample
} // operations
constexpr auto upsample = ttnn::register_operation_with_auto_launch_op<"ttnn::upsample", ttnn::operations::upsample::ExecuteUpSample>();
} // upsample
