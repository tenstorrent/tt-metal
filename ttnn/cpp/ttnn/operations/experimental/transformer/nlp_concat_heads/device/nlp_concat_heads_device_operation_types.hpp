// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"

namespace ttnn::experimental::prim {

struct NlpConcatHeadsParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    // Interleaved input and output with more than one head and fewer tile rows than cores: one work unit per
    // (tile row, head) instead of per tile row. Same output. Default off.
    bool head_split = false;
};

}  // namespace ttnn::experimental::prim
