// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct FillPadParams {
    float fill_value;
    tt::tt_metal::MemoryConfig output_mem_config;
    // When true, fill_value carries the raw bit pattern of an int32 fill value (used for int32 pad
    // sentinels that are not float-representable). When false (default), int32 fill_value is decoded
    // numerically via static_cast<int32_t>.
    bool fill_value_is_packed_bits = false;
};

struct FillPadInputs {
    Tensor input;
};

}  // namespace ttnn::prim
