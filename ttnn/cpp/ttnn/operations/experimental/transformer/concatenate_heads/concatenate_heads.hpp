// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include "device/concatenate_heads_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental::transformer {

struct ConcatenateHeadsOperation {
    static ttnn::Tensor invoke(
        const Tensor& input_tensor,
        const CoreCoord& compute_with_storage_grid_size,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return ttnn::prim::concatenate_heads(
            input_tensor, compute_with_storage_grid_size, memory_config, optional_output_tensor);
    }
};

}  // namespace operations::experimental::transformer

namespace experimental {

constexpr auto concatenate_heads = ttnn::register_operation<
    "ttnn::experimental::concatenate_heads",
    ttnn::operations::experimental::transformer::ConcatenateHeadsOperation>();

}  // namespace experimental

}  // namespace ttnn
