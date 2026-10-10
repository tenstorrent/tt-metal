// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::per_token_cast_to_fp8 {

struct PerTokenCastToFp8Params {
    tt::tt_metal::MemoryConfig output_memory_config;
    bool round_scale_to_power_of_two;
    uint32_t scale_block_w;      // elements per scale; divides the 128-element I/O chunk (32, 64 or 128)
    bool round_to_nearest_even;  // RNE to e4m3 in SFPU before the (truncating) packer
};

struct PerTokenCastToFp8Inputs {
    const Tensor& input_tensor;
};

}  // namespace ttnn::experimental::prim::per_token_cast_to_fp8
