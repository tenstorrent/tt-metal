// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct GgufMatmulParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct GgufMatmulInputs {
    Tensor input;   // [..., M, K] BFLOAT16 TILE, M <= 32
    Tensor weight;  // [K, N] GGUF dtype (from_gguf), DRAM interleaved
};

}  // namespace ttnn::prim
