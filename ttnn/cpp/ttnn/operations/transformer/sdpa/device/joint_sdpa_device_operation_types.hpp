// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::operations::transformer::sdpa::joint_sdpa {

struct operation_attributes_t {
    std::string joint_strategy;
    float scale = 0.0f;
    tt::tt_metal::MemoryConfig output_memory_config;
    std::optional<SDPAProgramConfig> program_config;
    DeviceComputeKernelConfig compute_kernel_config;

    std::uint32_t get_q_chunk_size() const { return program_config.has_value() ? program_config->q_chunk_size : 32; }

    std::uint32_t get_k_chunk_size() const { return program_config.has_value() ? program_config->k_chunk_size : 32; }
};

struct tensor_args_t {
    Tensor input_q;
    Tensor input_k;
    Tensor input_v;
    Tensor joint_q;
    Tensor joint_k;
    Tensor joint_v;
};

struct tensor_return_value_t {
    Tensor output;
    Tensor joint_output;
};

struct spec_return_value_t {
    TensorSpec output;
    TensorSpec joint_output;
};

}  // namespace ttnn::operations::transformer::sdpa::joint_sdpa
