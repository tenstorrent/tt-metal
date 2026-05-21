// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/core_coord.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Attributes for the fused Wan2.2 distributed RMSNorm device op.
// Combines: per-row RMSNorm pre stats, ring all-gather of stats across the TP
// cluster axis, post normalization, optional head-split, optional RoPE, and
// optional output-dtype cast — all in a single program with L1-resident input.
struct WanFusedDistributedRmsnormParams {
    float epsilon;
    uint32_t num_heads_per_device;

    // Output dtype override (defaults to input dtype if unset).
    std::optional<DataType> dtype;
    MemoryConfig output_mem_config;

    // CCL config
    uint32_t cluster_axis;
    uint32_t num_links;
    uint32_t ring_size;
    ttnn::ccl::Topology topology;
    std::vector<GlobalSemaphore> multi_device_global_semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;

    DeviceComputeKernelConfig compute_kernel_config;

    WanFusedDistributedRmsnormParams(
        float epsilon,
        uint32_t num_heads_per_device,
        std::optional<DataType> dtype,
        MemoryConfig output_mem_config,
        uint32_t cluster_axis,
        uint32_t num_links,
        uint32_t ring_size,
        ttnn::ccl::Topology topology,
        std::vector<GlobalSemaphore> multi_device_global_semaphore,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
        DeviceComputeKernelConfig compute_kernel_config) :
        epsilon(epsilon),
        num_heads_per_device(num_heads_per_device),
        dtype(dtype),
        output_mem_config(std::move(output_mem_config)),
        cluster_axis(cluster_axis),
        num_links(num_links),
        ring_size(ring_size),
        topology(topology),
        multi_device_global_semaphore(std::move(multi_device_global_semaphore)),
        sub_device_id(sub_device_id),
        compute_kernel_config(compute_kernel_config) {}

    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("epsilon", epsilon);
        attrs.emplace_back("num_heads_per_device", num_heads_per_device);
        attrs.emplace_back("dtype", dtype);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("cluster_axis", cluster_axis);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("compute_kernel_config", compute_kernel_config);
        return attrs;
    }
};

struct WanFusedDistributedRmsnormInputs {
    Tensor input;
    std::optional<const Tensor> weight;
    std::optional<const Tensor> transformation_mat;
    std::optional<const Tensor> rope_cos;
    std::optional<const Tensor> rope_sin;
    // Persistent buffer for the AG of stats (optional; allocated internally if null).
    std::optional<Tensor> persistent_output_buffer;
};

}  // namespace ttnn::experimental::prim
