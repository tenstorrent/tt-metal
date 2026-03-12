// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <utility>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::prim {

/**
 * Profile params for ring_joint_sdpa - simplified version without all_gather.
 *
 * Key differences from RingJointSDPAParams:
 * - No all_gather_operation_attributes or all_gather_tensor_args
 * - No ccl_core_grid_offset (no CCL workers to coordinate with)
 * - Added ring_index to specify which device we're simulating
 */
struct RingJointSDPAProfileParams {
    std::optional<std::string> joint_strategy;
    std::optional<float> scale;
    bool is_causal = false;
    bool is_balanced = false;
    std::size_t logical_n = 0;
    std::size_t ring_size = 0;
    std::size_t ring_index = 0;  // Which device position we're simulating
    tt::tt_metal::MemoryConfig output_memory_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;
    DeviceComputeKernelConfig compute_kernel_config;

    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        if (joint_strategy.has_value()) {
            attrs.emplace_back("joint_strategy", joint_strategy.value());
        }
        attrs.emplace_back("is_causal", is_causal);
        attrs.emplace_back("is_balanced", is_balanced);
        attrs.emplace_back("logical_n", logical_n);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("ring_index", ring_index);
        attrs.emplace_back("output_memory_config", output_memory_config);
        attrs.emplace_back("compute_kernel_config", compute_kernel_config);
        if (scale.has_value()) {
            attrs.emplace_back("scale", scale);
        }
        if (program_config.has_value()) {
            attrs.emplace_back("program_config", program_config);
        }
        return attrs;
    }

    std::uint32_t get_q_chunk_size() const { return program_config.has_value() ? program_config->q_chunk_size : 32; }

    std::uint32_t get_k_chunk_size() const { return program_config.has_value() ? program_config->k_chunk_size : 32; }

    std::string get_joint_strategy() const { return joint_strategy.value_or("rear"); }
};

/**
 * Profile input tensors - adds gathered_k/v as explicit pre-staged inputs.
 *
 * Key difference from RingJointSDPAInputs:
 * - gathered_k/v are provided externally (not built by all_gather)
 * - They contain KV from all ring devices in arrival order
 */
struct RingJointSDPAProfileInputs {
    Tensor input_q;     // Local Q for this device [B x NH x local_N x DH]
    Tensor input_k;     // Local K for this device [B x NH x local_N x DH]
    Tensor input_v;     // Local V for this device [B x NH x local_N x DH]
    Tensor gathered_k;  // Pre-staged full K in arrival order [B x NH x N x DH]
    Tensor gathered_v;  // Pre-staged full V in arrival order [B x NH x N x DH]
    std::optional<Tensor> joint_q;
    std::optional<Tensor> joint_k;
    std::optional<Tensor> joint_v;
};

// Reuse result types from ring_joint_sdpa
struct RingJointSDPAProfileResult {
    Tensor output;
    std::optional<Tensor> joint_output;
    Tensor lse_output;
};

struct RingJointSDPAProfileResultSpec {
    TensorSpec output;
    std::optional<TensorSpec> joint_output;
    TensorSpec lse_output;
};

}  // namespace ttnn::prim
