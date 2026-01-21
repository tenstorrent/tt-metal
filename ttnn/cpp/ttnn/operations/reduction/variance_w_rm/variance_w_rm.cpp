// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "variance_w_rm.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::variance_w_rm {

using namespace tt;
using namespace tt::tt_metal;

ttnn::Tensor ExecuteVarianceWRm::invoke(const ttnn::Tensor& input, const std::optional<MemoryConfig>& memory_config) {
    // Call the primitive device operation
    // Unwrap optional memory_config, defaulting to input tensor's memory config
    return ttnn::prim::variance_w_rm(input, memory_config.value_or(input.memory_config()));
}

}  // namespace ttnn::operations::variance_w_rm
