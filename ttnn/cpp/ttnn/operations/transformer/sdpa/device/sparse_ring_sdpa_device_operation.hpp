// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

#include "ttnn/operations/transformer/sdpa/device/sparse_ring_sdpa_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa/device/sparse_ring_sdpa_program_factory.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::prim {

struct SparseRingSDPADeviceOperation {
    using operation_attributes_t = SparseRingSDPAParams;
    using tensor_args_t = SparseRingSDPAInputs;
    using spec_return_value_t = SparseRingSDPAResultSpec;
    using tensor_return_value_t = SparseRingSDPAResult;
    using program_factory_t = std::variant<SparseRingSDPAMeshWorkloadFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<Tensors> create_op_performance_model(
        const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors);
};

SparseRingSDPAResult sparse_ring_sdpa(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& block_mask,
    const ttnn::Tensor& pooled_k,
    const ttnn::Tensor& pooled_v,
    ttnn::Tensor& persistent_output_buffer_k,
    ttnn::Tensor& persistent_output_buffer_v,
    std::size_t logical_n,
    ttnn::operations::transformer::SDPAProgramConfig program_config,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    CoreCoord ccl_core_grid_offset,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
    std::optional<float> scale = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy = ttnn::ccl::CoreAllocationStrategy::ROW_MAJOR);

}  // namespace ttnn::prim
