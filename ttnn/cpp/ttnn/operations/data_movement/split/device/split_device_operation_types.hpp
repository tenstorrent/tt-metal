// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct SplitParams {
    int num_splits{};
    int dim{};
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct SplitInputs {
    Tensor input;
};

}  // namespace ttnn::prim
