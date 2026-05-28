// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt_stl/reflection.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct AllGatherRMSNormParams {
    const float eps;
    const MemoryConfig output_mem_config;
    const DeviceComputeKernelConfig compute_kernel_config;
    const std::optional<DataType> dtype;
    const ttnn::ccl::Topology topology;
    const uint32_t num_links;
    const uint32_t ring_size;
    const uint32_t cluster_axis;
    const bool has_beta;
    const GlobalSemaphore semaphore;  // Not default-constructible.
    const std::optional<tt::tt_metal::SubDeviceId> sub_device_id;

    // Explicit constructor is required because GlobalSemaphore is not default-constructible.
    AllGatherRMSNormParams(
        float eps,
        MemoryConfig output_mem_config,
        DeviceComputeKernelConfig compute_kernel_config,
        std::optional<DataType> dtype,
        ttnn::ccl::Topology topology,
        uint32_t num_links,
        uint32_t ring_size,
        uint32_t cluster_axis,
        bool has_beta,
        GlobalSemaphore semaphore,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id) :
        eps(eps),
        output_mem_config(std::move(output_mem_config)),
        compute_kernel_config(compute_kernel_config),
        dtype(dtype),
        topology(topology),
        num_links(num_links),
        ring_size(ring_size),
        cluster_axis(cluster_axis),
        has_beta(has_beta),
        semaphore(std::move(semaphore)),
        sub_device_id(sub_device_id) {}

    // Attributes consumed by the default device-operation hash + reflection/logging.
    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("eps", eps);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("compute_kernel_config", compute_kernel_config);
        attrs.emplace_back("dtype", dtype);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("cluster_axis", cluster_axis);
        attrs.emplace_back("has_beta", has_beta);
        attrs.emplace_back("semaphore", semaphore);
        return attrs;
    }
};

struct AllGatherRMSNormInputs {
    Tensor input;
    std::optional<const Tensor> residual_input_tensor;
    std::optional<const Tensor> weight;  // gamma
    std::optional<const Tensor> bias;    // beta
    std::optional<Tensor> preallocated_stats;
};

}  // namespace ttnn::prim
