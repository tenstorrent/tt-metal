// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::nlp_concat_heads_decode {

struct NlpConcatHeadsDecodeParams {
    uint32_t num_heads{};
    bool on_subcoregrids{};
    std::optional<CoreRangeSet> sub_core_grids;
};

struct NlpConcatHeadsDecodeInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::nlp_concat_heads_decode
