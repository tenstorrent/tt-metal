// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::per_token_cast_to_fp8 {

struct PerTokenCastToFp8Params {
    tt::tt_metal::MemoryConfig output_memory_config;
    bool round_scale_to_power_of_two;
    // false: the e4m3 packer truncates (rounds toward zero). true: round half away from zero.
    bool round_to_nearest = false;
};

struct PerTokenCastToFp8Inputs {
    const Tensor& input_tensor;
};

}  // namespace ttnn::experimental::prim::per_token_cast_to_fp8
