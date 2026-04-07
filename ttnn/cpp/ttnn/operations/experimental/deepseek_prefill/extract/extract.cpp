// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "extract.hpp"
#include "device/extract_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::extract {

ttnn::Tensor extract(
    const ttnn::Tensor& global_tensor,
    const ttnn::Tensor& start,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    uint32_t local_expert_id,
    uint32_t max_dispatched_tokens_per_expert) {
    return ttnn::prim::prefill_extract(
        global_tensor, start, counts, global_expert_idx_table, local_expert_id, max_dispatched_tokens_per_expert);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::extract
