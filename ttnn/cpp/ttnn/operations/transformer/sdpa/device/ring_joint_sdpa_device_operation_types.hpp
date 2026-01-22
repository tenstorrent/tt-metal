// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <utility>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_fusion.hpp"

namespace ttnn::prim {

struct RingJointSDPAParams {
    std::string joint_strategy;
    std::optional<float> scale;
    std::size_t logical_n = 0;
    std::size_t ring_size = 0;
    tt::tt_metal::MemoryConfig output_memory_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    experimental::prim::RingAttentionAllGatherAsyncParams all_gather_operation_attributes;
    experimental::prim::RingAttentionAllGatherAsyncInputs all_gather_tensor_args;
    CoreCoord ccl_core_grid_offset;

    // We need a constructor, because all_gather_struct is not default initializable.
    RingJointSDPAParams(
        std::string joint_strategy,
        std::optional<float> scale,
        std::size_t logical_n,
        std::size_t ring_size,
        tt::tt_metal::MemoryConfig output_memory_config,
        std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
        DeviceComputeKernelConfig compute_kernel_config,
        experimental::prim::RingAttentionAllGatherAsyncParams all_gather_operation_attributes,
        experimental::prim::RingAttentionAllGatherAsyncInputs all_gather_tensor_args,
        CoreCoord ccl_core_grid_offset) :
        joint_strategy(std::move(joint_strategy)),
        scale(scale),
        logical_n(logical_n),
        ring_size(ring_size),
        output_memory_config(std::move(output_memory_config)),
        program_config(std::move(program_config)),
        compute_kernel_config(compute_kernel_config),
        all_gather_operation_attributes(std::move(all_gather_operation_attributes)),
        all_gather_tensor_args(std::move(all_gather_tensor_args)),
        ccl_core_grid_offset(ccl_core_grid_offset) {}

    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("joint_strategy", joint_strategy);
        attrs.emplace_back("logical_n", logical_n);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("output_memory_config", output_memory_config);
        attrs.emplace_back("compute_kernel_config", compute_kernel_config);
        attrs.emplace_back("ccl_core_grid_offset", ccl_core_grid_offset);
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
};

struct RingJointSDPAInputs {
    Tensor input_q;
    Tensor input_k;
    Tensor input_v;
    Tensor joint_q;
    Tensor joint_k;
    Tensor joint_v;
    Tensor gathered_k;
    Tensor gathered_v;
};

struct RingJointSDPAResult {
    Tensor output;
    Tensor joint_output;
    Tensor lse_output;
};

struct RingJointSDPAResultSpec {
    TensorSpec output;
    TensorSpec joint_output;
    TensorSpec lse_output;
};

}  // namespace ttnn::prim
