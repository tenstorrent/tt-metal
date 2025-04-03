// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"

namespace ttnn::operations::fused::normalization {

tt::tt_metal::operation::ProgramWithCallbacks frmsnorm_pre_multi_core_sharded(
    const Tensor& a,
    const std::optional<const Tensor>& b,  // residual
    Tensor& output,
    float eps,
    CoreCoord compute_grid_size,
    uint32_t subblock_wt,
    uint32_t block_wt,
    DeviceComputeKernelConfig compute_kernel_config,
    // New Parameters
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);

tt::tt_metal::operation::ProgramWithCallbacks frmsnorm_post_multi_core_sharded(
    const Tensor& a,
    const std::optional<const Tensor>& gamma,  // weight
    const std::optional<const Tensor>& stats,  // stats
    Tensor& output,
    float eps,
    CoreCoord compute_grid_size,
    uint32_t subblock_wt,
    uint32_t block_wt,
    DeviceComputeKernelConfig compute_kernel_config);

struct RMSAllGather {
    float eps;
    MemoryConfig output_mem_config;
    ttnn::operations::normalization::LayerNormProgramConfig program_config;
    const DeviceComputeKernelConfig compute_kernel_config;
    std::optional<DataType> dtype;
    const ttnn::ccl::Topology topology;
    const bool is_pre;
    const uint32_t num_links;
    const uint32_t ring_size;
    const uint32_t ring_index;
    const GlobalSemaphore semaphore;
    const std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    std::optional<IDevice*> forward_device;
    std::optional<IDevice*> backward_device;
    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;
    RMSAllGather(
        float eps,
        MemoryConfig output_mem_config,
        ttnn::operations::normalization::LayerNormProgramConfig program_config,
        const DeviceComputeKernelConfig compute_kernel_config,
        std::optional<DataType> dtype,
        ccl::Topology topology,
        const bool is_pre,
        uint32_t num_links,
        uint32_t ring_size,
        uint32_t ring_index,
        GlobalSemaphore semaphore,
        std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        std::optional<IDevice*> forward_device,
        std::optional<IDevice*> backward_device) :
        eps(eps),
        output_mem_config(output_mem_config),
        program_config(program_config),
        compute_kernel_config(compute_kernel_config),
        dtype(dtype),
        topology(topology),
        is_pre(is_pre),
        num_links(num_links),
        ring_size(ring_size),
        ring_index(ring_index),
        semaphore(semaphore),
        sub_device_id(sub_device_id),
        forward_device(forward_device),
        backward_device(backward_device) {}

    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("eps", eps);
        attrs.emplace_back("program_config", program_config);
        attrs.emplace_back("compute_kernel_config", compute_kernel_config);
        attrs.emplace_back("dtype", dtype);
        attrs.emplace_back("is_pre", is_pre);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("ring_index", ring_index);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("semaphore", semaphore);

        return attrs;
    }
    const tt::tt_metal::operation::Hash compute_program_hash(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
};

RMSAllGather create_rms_struct(
    const Tensor& input_tensor,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<IDevice*>& devices,
    const ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphores,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    float epsilon,
    const ttnn::operations::normalization::LayerNormProgramConfig program_config,
    const DeviceComputeKernelConfig compute_kernel_config,
    std::optional<DataType> dtype,
    const bool is_pre);

}  // namespace ttnn::operations::fused::normalization
