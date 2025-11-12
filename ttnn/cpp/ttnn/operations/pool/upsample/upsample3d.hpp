// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace upsample {

struct ExecuteUpSample3D {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        std::variant<int, std::array<int, 3>> scale_factor,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

}  // namespace upsample
}  // namespace operations

constexpr auto upsample3d =
    ttnn::register_operation<"ttnn::upsample3d", ttnn::operations::upsample::ExecuteUpSample3D>();

}  // namespace ttnn
