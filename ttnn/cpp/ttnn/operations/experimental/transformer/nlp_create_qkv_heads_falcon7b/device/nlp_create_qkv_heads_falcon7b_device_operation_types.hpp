// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct NlpCreateQkvHeadsFalcon7bParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

// Return types using named structs for Q, K, V heads
struct NlpCreateQkvHeadsFalcon7bResult {
    Tensor q;
    Tensor k;
    Tensor v;
};

struct NlpCreateQkvHeadsFalcon7bResultSpec {
    TensorSpec q;
    TensorSpec k;
    TensorSpec v;
};

}  // namespace ttnn::experimental::prim
