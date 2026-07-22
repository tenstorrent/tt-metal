// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct RepeatInterleaveCodegenParams {
    uint32_t rep_dim{};
    uint32_t num_repeats{};
    uint32_t lower_pages{};
    uint32_t rep_dim_pages{};
    uint32_t total_out_pages{};
    uint32_t stick_size{};
    uint32_t stick_size_out{};
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct RepeatInterleaveCodegenInputs {
    Tensor input;
};

}  // namespace ttnn::prim
