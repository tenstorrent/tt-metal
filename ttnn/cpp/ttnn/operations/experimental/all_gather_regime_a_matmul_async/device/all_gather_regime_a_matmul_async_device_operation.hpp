// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "all_gather_regime_a_matmul_async_device_operation_types.hpp"
#include "all_gather_regime_a_matmul_async_program_factory.hpp"

namespace ttnn::experimental::prim {

// Fused all-gather(in0 on K) + regime_a_matmul device operation (Phase A, DRAM-staged). D=1 never reaches
// here (the public op delegates to regime_a_matmul); this handles D>1.
struct AllGatherRegimeAMatmulAsyncDeviceOperation {
    using operation_attributes_t = AllGatherRegimeAMatmulAsyncParams;
    using tensor_args_t = AllGatherRegimeAMatmulAsyncInputs;
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<AllGatherRegimeAMatmulAsyncProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    // Custom program-cache hash over stable scalar/shape fields only. The default reflection hash visits
    // fabric/semaphore/width-shard members that carry a CoreRangeSet the hasher cannot serialize.
    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const Tensor& weight_tensor,
        const std::optional<const RegimeAMatmulConfig>& config,
        uint32_t d,
        std::optional<uint32_t> cluster_axis,
        ttnn::ccl::Topology topology,
        uint32_t num_links,
        uint32_t num_workers_per_link,
        uint32_t num_buffers_per_channel,
        std::vector<GlobalSemaphore> multi_device_global_semaphore,
        std::optional<GlobalSemaphore> barrier_semaphore,
        std::optional<Tensor> persistent_output_buffer,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<const DataType> dtype,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> all_gather_regime_a_matmul_async(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const experimental::prim::RegimeAMatmulConfig>& config,
    uint32_t d,
    std::optional<uint32_t> cluster_axis,
    ttnn::ccl::Topology topology,
    uint32_t num_links,
    uint32_t num_workers_per_link,
    uint32_t num_buffers_per_channel,
    std::vector<GlobalSemaphore> multi_device_global_semaphore,
    std::optional<GlobalSemaphore> barrier_semaphore,
    std::optional<Tensor> persistent_output_buffer,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config);

}  // namespace ttnn::prim
