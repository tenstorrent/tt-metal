// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::insert {

struct InsertParams {
    // Index into global_expert_idx_table. The kernel looks up
    //   global_expert_id = global_expert_idx_table[local_expert_id]
    // at runtime and uses the result to index start / counts.
    uint32_t local_expert_id;

    static constexpr auto attribute_names = std::forward_as_tuple("local_expert_id");

    auto attribute_values() const { return std::forward_as_tuple(local_expert_id); }
};

struct InsertInputs {
    Tensor global_tensor;
    Tensor local_tensor;
    Tensor start;
    Tensor counts;
    // 1D (or 2D with first dim == 1) UINT32 DRAM-interleaved tensor mapping
    // local_expert_id -> global_expert_id.
    Tensor global_expert_idx_table;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::insert
