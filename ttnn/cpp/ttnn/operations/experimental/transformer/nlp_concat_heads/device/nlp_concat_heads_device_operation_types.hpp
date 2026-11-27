// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"

namespace ttnn::experimental::prim {

struct NlpConcatHeadsParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    // Interleaved input and output only: one program serves every row count (the rows are a runtime word).
    bool runtime_rows = false;
};

}  // namespace ttnn::experimental::prim
