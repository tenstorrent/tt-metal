// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct GdnGatesParams {
    uint32_t sequence;    // T (tile aligned)
    uint32_t rank;        // rank of gab (3: [1,T,W], 4: [1,1,T,W]); the outputs get the same rank
    uint32_t num_heads;   // <= 32
    uint32_t a_col_tile;  // tile column of a in gab
    uint32_t b_col_tile;  // tile column of b in gab
    float beta_scale;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct GdnGatesInputs {
    Tensor gab;
    Tensor dt_bias;
    Tensor a_neg;
};

}  // namespace ttnn::experimental::prim
