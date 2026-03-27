// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MaskedBincountParams {
    const uint32_t n_routed_experts;
    const uint32_t num_experts_per_token;
};

struct MaskedBincountInputs {
    const Tensor& input_tensor;
    const Tensor& expert_mask;
};

}  // namespace ttnn::experimental::prim
