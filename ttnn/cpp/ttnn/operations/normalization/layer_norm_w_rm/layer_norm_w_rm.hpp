// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include "ttnn/decorators.hpp"
#include "device/layer_norm_w_rm_device_operation.hpp"

namespace ttnn {
namespace operations {
namespace layer_norm_w_rm {

struct ExecuteLayerNormWRm {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input,
        const ttnn::Tensor& gamma,
        const ttnn::Tensor& beta,
        float epsilon = 1e-5f,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace layer_norm_w_rm
}  // namespace operations

// Register the operation
constexpr auto layer_norm_w_rm =
    ttnn::register_operation<"ttnn::layer_norm_w_rm", ttnn::operations::layer_norm_w_rm::ExecuteLayerNormWRm>();

}  // namespace ttnn