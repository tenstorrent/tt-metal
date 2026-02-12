// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include <tuple>

namespace ttnn::prim {

struct JointSDPAParams {
    std::string joint_strategy;
    float scale = 0.0f;
    tt::tt_metal::MemoryConfig output_memory_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;
    DeviceComputeKernelConfig compute_kernel_config;

    std::uint32_t get_q_chunk_size() const { return program_config.has_value() ? program_config->q_chunk_size : 32; }

    std::uint32_t get_k_chunk_size() const { return program_config.has_value() ? program_config->k_chunk_size : 32; }

    static constexpr auto attribute_names = std::forward_as_tuple(
        "joint_strategy", "scale", "output_memory_config", "program_config", "compute_kernel_config");
    auto attribute_values() const {
        return std::forward_as_tuple(
            joint_strategy, scale, output_memory_config, program_config, compute_kernel_config);
    }
};

struct JointSDPAInputs {
    Tensor input_q;
    Tensor input_k;
    Tensor input_v;
    Tensor joint_q;
    Tensor joint_k;
    Tensor joint_v;

    static constexpr auto attribute_names =
        std::forward_as_tuple("input_q", "input_k", "input_v", "joint_q", "joint_k", "joint_v");
    auto attribute_values() const {
        return std::forward_as_tuple(input_q, input_k, input_v, joint_q, joint_k, joint_v);
    }
};

struct JointSDPAResult {
    Tensor output;
    Tensor joint_output;
};

struct JointSDPAResultSpec {
    TensorSpec output;
    TensorSpec joint_output;
};

}  // namespace ttnn::prim
