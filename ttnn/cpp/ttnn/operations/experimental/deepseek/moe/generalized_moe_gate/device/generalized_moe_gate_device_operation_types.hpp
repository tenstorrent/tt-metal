// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek::moe::generalized_moe_gate {

struct operation_attributes_t {
    float eps{};
    float scaling_factor{};
    bool enable_sigmoid{};
    uint32_t topk{8};            // number of experts to keep (k). ONLY 4, 6, 8 supported (enforced in validate; the
                                 // finalize rank-mask + tests cover exactly these); 8 = full top-8.
    bool output_softmax{false};  // false = linear normalize (score/Σ); true = softmax over the selected top-k.
    bool grouped{false};         // false = ungrouped global top-k (the generalized kernel path). true = DeepSeek
                                 // grouped gate (8 groups × 32 → top-2-sum → top-4 groups → top-8): single 256-block,
                                 // forced top-8 + linear renorm (topk/output_softmax ignored). Selects the kernel's
                                 // grouped `#else` path via the GMG_UNGROUPED_TOP8=0 compile define (see builder).
};

struct tensor_args_t {
    const Tensor& input_tensor;
    const Tensor& bias_tensor;
    const Tensor& input_indices_tensor;
    const Tensor& output_tensor;
    const Tensor& output_indices_tensor;
};

using tensor_return_value_t = std::tuple<Tensor, Tensor>;

using spec_return_value_t = std::tuple<TensorSpec, TensorSpec>;

}  // namespace ttnn::operations::experimental::deepseek::moe::generalized_moe_gate
