// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/core_coord.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::fused::normalization {

struct operation_attributes_t {
    const float eps;
    const MemoryConfig output_mem_config;
    uint32_t subblock_wt;
    uint32_t block_wt;
    bool inplace;
    tt::tt_metal::CoreCoord grid_size;
    const DeviceComputeKernelConfig compute_kernel_config;
    const std::optional<DataType> dtype;
    const ttnn::ccl::Topology topology;
    const uint32_t num_links;
    const uint32_t ring_size;
    const GlobalSemaphore semaphore;  // Not default constructible
    const std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    const uint32_t cluster_axis;
    const bool use_noc1_only;

    // Constructor required because GlobalSemaphore is not default constructible
    operation_attributes_t(
        float eps,
        MemoryConfig output_mem_config,
        uint32_t subblock_wt,
        uint32_t block_wt,
        bool inplace,
        tt::tt_metal::CoreCoord grid_size,
        DeviceComputeKernelConfig compute_kernel_config,
        std::optional<DataType> dtype,
        ttnn::ccl::Topology topology,
        uint32_t num_links,
        uint32_t ring_size,
        GlobalSemaphore semaphore,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
        uint32_t cluster_axis,
        bool use_noc1_only) :
        eps(eps),
        output_mem_config(std::move(output_mem_config)),
        subblock_wt(subblock_wt),
        block_wt(block_wt),
        inplace(inplace),
        grid_size(grid_size),
        compute_kernel_config(compute_kernel_config),
        dtype(dtype),
        topology(topology),
        num_links(num_links),
        ring_size(ring_size),
        semaphore(std::move(semaphore)),
        sub_device_id(sub_device_id),
        cluster_axis(cluster_axis),
        use_noc1_only(use_noc1_only) {}

    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("eps", eps);
        attrs.emplace_back("subblock_wt", subblock_wt);
        attrs.emplace_back("block_wt", block_wt);
        attrs.emplace_back("compute_kernel_config", compute_kernel_config);
        attrs.emplace_back("dtype", dtype);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("semaphore", semaphore);
        attrs.emplace_back("cluster_axis", cluster_axis);
        return attrs;
    }
};

struct tensor_args_t {
    Tensor input;
    std::optional<const Tensor> residual_input_tensor;
    std::optional<const Tensor> weight;
    std::optional<const Tensor> stats;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::operations::fused::normalization
