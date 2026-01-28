// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct NlpConcatHeadsDecodeParams {
    uint32_t num_heads{};
    bool on_subcoregrids{};
    std::optional<CoreRangeSet> sub_core_grids;
};

struct NlpConcatHeadsDecodeInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::experimental::prim
