// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::insert {

ttnn::Tensor insert(
    const ttnn::Tensor& global_tensor,
    const ttnn::Tensor& local_tensor,
    const ttnn::Tensor& start,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    uint32_t local_expert_id);

}  // namespace ttnn::operations::experimental::deepseek_prefill::insert

namespace ttnn {
using operations::experimental::deepseek_prefill::insert::insert;
}  // namespace ttnn
