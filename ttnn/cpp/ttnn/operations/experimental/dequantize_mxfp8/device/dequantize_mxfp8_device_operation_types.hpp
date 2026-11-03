// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../dequantize_mxfp8.hpp"

namespace ttnn::experimental::prim {
struct Mxfp8Params {
    DataType dtype;
    Layout output_layout;
    bool fp32_dest_acc_en;
    MemoryConfig memory_config;
};
struct Mxfp8Inputs {
    Tensor packed;
    Tensor scales;
};
}  // namespace ttnn::experimental::prim
