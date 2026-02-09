// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::fill_rm {

struct FillRmParams {
    uint32_t N{0};
    uint32_t C{0};
    uint32_t H{0};
    uint32_t W{0};
    uint32_t hFill{0};
    uint32_t wFill{0};
    float val_hi{0.0f};
    float val_lo{0.0f};
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct FillRmInputs {
    Tensor input;
};

}  // namespace ttnn::operations::data_movement::fill_rm
