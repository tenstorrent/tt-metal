// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "iterative_topk.hpp"
#include "device/iterative_topk_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::iterative_topk {

std::array<Tensor, 2> iterative_topk(
    const Tensor& input, uint32_t k, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::prim::iterative_topk(input, k, output_mem_config);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::iterative_topk
