// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct NlpCreateQkvHeadsDecodeParams {
    uint32_t num_q_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    bool overlap_qk_coregrid;
    bool input_on_subcoregrids;
    std::optional<uint32_t> slice_size;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct NlpCreateQkvHeadsDecodeInputs {
    Tensor input_tensor;
    std::optional<Tensor> batch_offset;
};

}  // namespace ttnn::experimental::prim
