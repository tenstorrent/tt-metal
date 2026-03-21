// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::offset_cumsum {

std::array<ttnn::Tensor, 2> offset_cumsum(
    const ttnn::Tensor& input_tensor,
    uint32_t cluster_axis,
    uint32_t num_links,
    const ttnn::MemoryConfig& memory_config);

}  // namespace ttnn::operations::experimental::deepseek_prefill::offset_cumsum

namespace ttnn {
using operations::experimental::deepseek_prefill::offset_cumsum::offset_cumsum;
}  // namespace ttnn
