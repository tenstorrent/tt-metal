// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <utility>

#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"

namespace ttnn::operations::fused::normalization {

tt::tt_metal::operation::ProgramWithCallbacks frmsnorm_multi_core_sharded(
    const Tensor& a,
    const std::optional<const Tensor>& b,      // residual
    const std::optional<const Tensor>& gamma,  // weight
    const std::optional<const Tensor>& stats,  // stats
    Tensor& output,
    float eps,
    CoreCoord compute_grid_size,
    uint32_t subblock_wt,
    uint32_t block_wt,
    DeviceComputeKernelConfig compute_kernel_config,
    // New Parameters
    IDevice* target_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ::ttnn::ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    bool use_noc1_only);

tt::tt_metal::operation::ProgramWithCallbacks frmsnorm_post_multi_core_sharded(
    const Tensor& a,
    const std::optional<const Tensor>& gamma,  // weight
    const std::optional<const Tensor>& stats,  // stats
    Tensor& output,
    float eps,
    CoreCoord compute_grid_size,
    uint32_t subblock_wt,
    uint32_t block_wt,
    DeviceComputeKernelConfig compute_kernel_config,
    const GlobalSemaphore& semaphore,
    uint32_t ring_size,
    uint32_t num_links);

struct RMSAllGather {
    float eps;
    MemoryConfig output_mem_config;
    ttnn::operations::normalization::LayerNormProgramConfig program_config;
    const DeviceComputeKernelConfig compute_kernel_config;
    std::optional<DataType> dtype;
    const ttnn::ccl::Topology topology;
    const uint32_t num_links;
    const uint32_t ring_size;
    const GlobalSemaphore semaphore;
    const std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    const uint32_t cluster_axis = 0;
    bool use_noc1_only = false;

    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program_at(
        const ttnn::MeshCoordinate& mesh_coordinate,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;
    RMSAllGather(
        float eps,
        MemoryConfig output_mem_config,
        ttnn::operations::normalization::LayerNormProgramConfig program_config,
        const DeviceComputeKernelConfig compute_kernel_config,
        std::optional<DataType> dtype,
        ::ttnn::ccl::Topology topology,
        const uint32_t num_links,
        const uint32_t ring_size,
        GlobalSemaphore semaphore,
        std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        uint32_t cluster_axis,
        bool use_noc1_only) :
        eps(eps),
        output_mem_config(std::move(output_mem_config)),
        program_config(program_config),
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
        attrs.emplace_back("program_config", program_config);
        attrs.emplace_back("compute_kernel_config", compute_kernel_config);
        attrs.emplace_back("dtype", dtype);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("semaphore", semaphore);
        attrs.emplace_back("cluster_axis", cluster_axis);

        return attrs;
    }
    tt::tt_metal::operation::Hash compute_program_hash(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
};

RMSAllGather create_rms_struct(
    const Tensor& input_tensor,
    uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<IDevice*>& devices,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphores,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    float epsilon,
    ttnn::operations::normalization::LayerNormProgramConfig program_config,
    DeviceComputeKernelConfig compute_kernel_config,
    std::optional<DataType> dtype);

}  // namespace ttnn::operations::fused::normalization
