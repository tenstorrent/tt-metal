// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_fusion.hpp"

namespace ttnn::prim {

struct ExpRingJointSDPAParams {
    std::string joint_strategy;
    std::optional<float> scale;
    std::size_t logical_n = 0;
    std::size_t ring_size = 0;
    tt::tt_metal::MemoryConfig output_memory_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    // Flattened CCL (all-gather) params
    int32_t dim;
    uint32_t num_links;
    ttnn::ccl::Topology topology;
    std::vector<GlobalSemaphore> semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy;
    uint32_t cluster_axis;
    CoreCoord ccl_core_grid_offset;

    ExpRingJointSDPAParams(
        std::string joint_strategy,
        std::optional<float> scale,
        std::size_t logical_n,
        std::size_t ring_size,
        tt::tt_metal::MemoryConfig output_memory_config,
        std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
        DeviceComputeKernelConfig compute_kernel_config,
        int32_t dim,
        uint32_t num_links,
        ttnn::ccl::Topology topology,
        std::vector<GlobalSemaphore> semaphore,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
        ttnn::ccl::CoreAllocationStrategy core_allocation_strategy,
        uint32_t cluster_axis,
        CoreCoord ccl_core_grid_offset) :
        joint_strategy(std::move(joint_strategy)),
        scale(scale),
        logical_n(logical_n),
        ring_size(ring_size),
        output_memory_config(std::move(output_memory_config)),
        program_config(std::move(program_config)),
        compute_kernel_config(compute_kernel_config),
        dim(dim),
        num_links(num_links),
        topology(topology),
        semaphore(std::move(semaphore)),
        sub_device_id(sub_device_id),
        core_allocation_strategy(core_allocation_strategy),
        cluster_axis(cluster_axis),
        ccl_core_grid_offset(ccl_core_grid_offset) {}

    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("joint_strategy", joint_strategy);
        attrs.emplace_back("logical_n", logical_n);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("output_memory_config", output_memory_config);
        attrs.emplace_back("compute_kernel_config", compute_kernel_config);
        attrs.emplace_back("dim", dim);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("cluster_axis", cluster_axis);
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

struct ExpRingJointSDPAInputs {
    Tensor input_q;
    Tensor input_k;
    Tensor input_v;
    Tensor joint_q;
    Tensor joint_k;
    Tensor joint_v;
    Tensor gathered_k;
    Tensor gathered_v;
};

// Index constants for ExpRingJointSDPAResult vector
constexpr size_t EXP_RING_JOINT_SDPA_OUTPUT_IDX = 0;
constexpr size_t EXP_RING_JOINT_SDPA_JOINT_OUTPUT_IDX = 1;
constexpr size_t EXP_RING_JOINT_SDPA_STATS_OUTPUT_IDX = 2;

// ExpRingJointSDPAResult is a vector of 3 tensors: [output, joint_output, stats_output]
using ExpRingJointSDPAResult = Tensors;

// ExpRingJointSDPAResultSpec is a vector of 3 TensorSpecs: [output, joint_output, stats_output]
using ExpRingJointSDPAResultSpec = std::vector<TensorSpec>;

}  // namespace ttnn::prim
