// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::prim {

struct JointSDPAParams {
    std::string joint_strategy;
    float scale = 0.0f;
    tt::tt_metal::MemoryConfig output_memory_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;
    // Precision recipe (device/recipe/); resolved to STANDARD by default before dispatch.
    std::optional<ttnn::operations::transformer::SDPAPrecision> precision;
    // Recipe calls: the L1 their layout may use (recipe_l1_budget), set before the program-cache lookup.
    uint64_t recipe_l1_budget = 0;

    std::uint32_t get_q_chunk_size() const { return program_config.has_value() ? program_config->q_chunk_size : 32; }

    std::uint32_t get_k_chunk_size() const { return program_config.has_value() ? program_config->k_chunk_size : 32; }
};

struct JointSDPAInputs {
    Tensor input_q;
    Tensor input_k;
    Tensor input_v;
    Tensor joint_q;
    Tensor joint_k;
    Tensor joint_v;
};

// Index constants for JointSDPAResult vector
constexpr size_t JOINT_SDPA_OUTPUT_IDX = 0;
constexpr size_t JOINT_SDPA_JOINT_OUTPUT_IDX = 1;

// JointSDPAResult is a vector of 2 tensors: [output, joint_output]
using JointSDPAResult = Tensors;

// JointSDPAResultSpec is a vector of 2 TensorSpecs: [output, joint_output]
using JointSDPAResultSpec = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttnn::prim
