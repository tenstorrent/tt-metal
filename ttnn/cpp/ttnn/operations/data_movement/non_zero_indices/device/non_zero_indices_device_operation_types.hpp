// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct NonzeroParams {
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct NonzeroInputs {
    Tensor input;
};

using NonzeroResult = std::tuple<Tensor, Tensor>;
using NonzeroResultSpec = std::tuple<TensorSpec, TensorSpec>;

}  // namespace ttnn::prim
