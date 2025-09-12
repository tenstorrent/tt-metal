// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"

#include "ttnn/run_operation.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace ttnn {

using ccl::EriscDatamoverBuilder;
struct AllGatherParams {
    std::vector<IDevice*> devices;
    const uint32_t dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const MemoryConfig output_mem_config;
    const ccl::Topology topology;
    const GlobalSemaphore semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    std::optional<uint32_t> cluster_axis;

    AllGatherParams(
        std::vector<IDevice*> devices,
        uint32_t dim,
        uint32_t num_links,
        uint32_t ring_size,
        MemoryConfig output_mem_config,
        ccl::Topology topology,
        GlobalSemaphore semaphore,
        std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        std::optional<uint32_t> cluster_axis) :
        devices(std::move(devices)),
        dim(dim),
        num_links(num_links),
        ring_size(ring_size),
        output_mem_config(std::move(output_mem_config)),
        topology(topology),
        semaphore(std::move(semaphore)),
        sub_device_id(sub_device_id),
        cluster_axis(cluster_axis) {}
};

struct LlamaAllGatherMatmulAsync {
    /* All Gather Replicate Params */
    const ttnn::AllGatherParams all_gather_params;

    /* Matmul Params */
    const operations::matmul::Matmul matmul_struct;

    /* Physical Devices this op runs on*/
    std::vector<IDevice*> devices;

    /* General */
    void validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& optional_output_tensors) const;

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

    tt::tt_metal::operation::Hash compute_program_hash(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple("matmul_struct", "devices");
    auto attribute_values() const { return std::forward_as_tuple(this->matmul_struct, this->devices); }
};

// llama All Gather MM Variants
tt::tt_metal::operation::ProgramWithCallbacks llama_all_gather_matmul_async_sharded(
    const Tensor& input_tensor,
    const Tensor& input1,
    Tensor& output_tensor,
    const Tensor& intermediate_tensor,
    const Tensor& aggregated_tensor,
    IDevice* target_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb);

tt::tt_metal::operation::ProgramWithCallbacks matmul_multi_core_agmm_fusion_helper(
    tt::tt_metal::Program& program,
    const Tensor& input0,
    const std::vector<Tensor>& input_tensors_b,
    const std::optional<const Tensor>& bias,
    const std::vector<Tensor>& output_tensors,
    bool bcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);

namespace operations {
namespace experimental {
namespace ccl {

Tensor llama_all_gather_matmul_async(
    const Tensor& input_tensor,
    const Tensor& input1,
    const Tensor& intermediate_tensor,
    int32_t dim,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<MemoryConfig>& ag_memory_config = std::nullopt,
    const std::optional<MemoryConfig>& mm_memory_config = std::nullopt,
    std::optional<size_t> num_preferred_links = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb = std::nullopt);

LlamaAllGatherMatmulAsync create_llama_all_gather_matmul_async_struct(
    const ttnn::AllGatherParams& all_gather_params,
    const operations::matmul::Matmul& matmul_struct,
    const std::vector<IDevice*>& devices);

}  // namespace ccl
}  // namespace experimental

namespace llama_matmul {

tt::tt_metal::operation::ProgramWithCallbacks matmul_multi_core_agmm_fusion_helper(
    tt::tt_metal::Program& program,
    const Tensor& a,
    const std::vector<Tensor>& b_tensors,
    const std::optional<const Tensor>& bias,
    const std::vector<Tensor>& output_tensors,
    bool broadcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const matmul::MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);

}  // namespace llama_matmul

}  // namespace operations

}  // namespace ttnn

namespace llama_agmm_fusion_helpers {
void override_program_parameters(
    const ttnn::operations::matmul::matmul_mcast_1d_common_override_variables_t& override_variables,
    const void* operation,
    tt::tt_metal::Program& program,
    const std::vector<tt::tt_metal::Tensor>& input_tensors,
    const std::vector<std::optional<const tt::tt_metal::Tensor>>& optional_input_tensors,
    const std::vector<tt::tt_metal::Tensor>& output_tensors);

}  // namespace llama_agmm_fusion_helpers
