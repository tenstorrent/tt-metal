// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::transformer::rotary_embedding_llama_fused_qk {

struct RotaryEmbeddingLlamaFusedQkParams {
    tt::tt_metal::MemoryConfig q_output_mem_config;
    tt::tt_metal::MemoryConfig k_output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    bool row_major_QK{};
};

struct RotaryEmbeddingLlamaFusedQkInputs {
    Tensor q_input;
    Tensor k_input;
    Tensor cos;
    Tensor sin;
    Tensor trans_mat;
};

using tensor_return_value_t = std::tuple<Tensor, Tensor>;

using spec_return_value_t = std::tuple<TensorSpec, TensorSpec>;

}  // namespace ttnn::operations::experimental::transformer::rotary_embedding_llama_fused_qk
