// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/global_circular_buffer.hpp>
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::transformer::sdpa {

struct operation_attributes_t {
    const std::optional<float> scale;
    const tt::tt_metal::MemoryConfig output_mem_config;
    const std::optional<SDPAProgramConfig> program_config;
    const bool is_causal;
    const std::optional<int64_t> chunk_start_idx;
    const DeviceComputeKernelConfig compute_kernel_config;
    const bool use_mla;
    const std::optional<uint32_t> head_dim_v;
    const std::optional<uint32_t> sliding_window_size;
};

struct tensor_args_t {
    const Tensor& q;
    const Tensor& k;
    std::optional<Tensor> v;
    std::optional<Tensor> attn_mask;
    std::optional<Tensor> page_table;
    std::optional<Tensor> attention_sink;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::transformer::sdpa
