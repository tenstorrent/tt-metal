// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::per_token_cast_back {

struct PerTokenCastBackParams {
    tt::tt_metal::DataType output_dtype;
    tt::tt_metal::MemoryConfig output_memory_config;
    // When true, the compute kernel narrows the fp32 scale to bf16 on-device and runs the broadcast
    // multiply in bf16 (HiFi2); when false (default), the scale stays fp32 (HiFi4).
    bool compute_is_bf16 = false;
};

struct PerTokenCastBackInputs {
    const Tensor& input_e4m3;
    const Tensor& input_scale;
};

}  // namespace ttnn::experimental::prim::per_token_cast_back
