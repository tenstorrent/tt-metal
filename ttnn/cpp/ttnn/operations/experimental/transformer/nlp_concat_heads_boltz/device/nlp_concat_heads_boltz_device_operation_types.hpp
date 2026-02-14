// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct NLPConcatHeadsBoltzParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct NLPConcatHeadsBoltzInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::experimental::prim
