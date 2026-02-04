// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct FillPadParams {
    float fill_value;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct FillPadInputs {
    Tensor input;
};

}  // namespace ttnn::prim
