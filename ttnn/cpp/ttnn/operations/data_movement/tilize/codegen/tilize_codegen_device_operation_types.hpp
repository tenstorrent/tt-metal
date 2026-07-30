// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Mirrors manifest cache_key_fields for tilize codegen exactly.
struct TilizeCodegenParams {
    uint32_t NC = 0;
    uint32_t Ht = 0;
    uint32_t Wt = 0;
    tt::tt_metal::DataType input_dtype;
    tt::tt_metal::DataType output_dtype;
    tt::tt_metal::MemoryConfig input_mem_config;
    tt::tt_metal::MemoryConfig output_mem_config;
    bool use_multicore = false;
    bool use_low_perf = false;
    bool preserve_logical_shape = false;
};

struct TilizeCodegenInputs {
    Tensor input_tensor;
};

}  // namespace ttnn::prim
