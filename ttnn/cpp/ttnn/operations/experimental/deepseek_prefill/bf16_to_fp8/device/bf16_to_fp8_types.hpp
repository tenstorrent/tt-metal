// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::bf16_to_fp8 {

struct Bf16ToFp8Params {
    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const { return std::forward_as_tuple(); }
};

struct Bf16ToFp8Inputs {
    Tensor input_tensor;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::bf16_to_fp8
