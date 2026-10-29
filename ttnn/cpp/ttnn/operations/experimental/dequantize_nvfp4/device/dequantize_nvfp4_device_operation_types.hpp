// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../dequantize_nvfp4.hpp"

namespace ttnn::experimental::prim {
struct Nvfp4Params {
    float global_scale;
    operations::experimental::Nvfp4ScaleFormat scale_format;
    DataType dtype;
    Layout output_layout;
    bool fp32_dest_acc_en;
    MemoryConfig memory_config;
    bool experimental_fast;
    bool fpu_tensor_scale;
    bool transpose_output;
};
struct Nvfp4Inputs {
    Tensor packed;
    Tensor scales;
};
}  // namespace ttnn::experimental::prim
