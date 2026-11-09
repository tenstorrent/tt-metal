// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct CreateQkvHeadsFalcon7bParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

// Return types using named structs for Q, K, V heads
struct CreateQkvHeadsFalcon7bResult {
    Tensor q;
    Tensor k;
    Tensor v;
};

struct CreateQkvHeadsFalcon7bResultSpec {
    tt::tt_metal::TensorSpec q;
    tt::tt_metal::TensorSpec k;
    tt::tt_metal::TensorSpec v;
};

}  // namespace ttnn::experimental::prim
