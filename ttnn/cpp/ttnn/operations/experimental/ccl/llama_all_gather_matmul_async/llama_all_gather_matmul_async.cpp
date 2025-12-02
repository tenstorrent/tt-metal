// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/llama_all_gather_matmul_async.hpp"

#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteAllGatherMatmulAsync::invoke(
    const ttnn::Tensor& input0,
    const ttnn::Tensor& input1,
    const ttnn::Tensor& intermediate_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<MemoryConfig>& ag_memory_config,
    const std::optional<MemoryConfig>& mm_memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const DataType> dtype,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb) {
    tt::tt_fabric::Topology usable_topology = ::ttnn::ccl::get_usable_topology(input0, topology, cluster_axis);

    const auto& mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(),
        "all-gather-replicate invoked with cluster_axis API on >2D mesh, which is currently unsupported");
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    int32_t rank = input0.logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input0);

    operations::matmul::Matmul matmul_struct = operations::matmul::create_matmul_struct(
        input0,
        input1,
        /*parameters=*/
        operations::matmul::Matmul{
            program_config,
            /*bcast_batch=*/std::nullopt,
            mm_memory_config.value_or(input0.memory_config()),
            dtype.value_or(input0.dtype()),
            compute_kernel_config,
            /*untilize_out=*/false,
            /*user_core_coord=*/std::nullopt,
            /*activation=*/std::nullopt,
            /*user_run_batched=*/false,
            /*transpose_a=*/false,
            /*transpose_b=*/false,
            /*output_tile=*/std::nullopt,
            /*global_cb=*/global_cb});

    auto output_tensors = ttnn::prim::llama_all_gather_matmul_async(
        input0,
        input1,
        intermediate_tensor,
        devices,
        gather_dim,
        num_preferred_links.has_value() ? num_preferred_links.value() : 1,
        num_devices,
        ag_memory_config.value_or(input0.memory_config()),
        usable_topology,
        multi_device_global_semaphore,
        matmul_struct,
        sub_device_id,
        cluster_axis);

    output_tensors.aggregated.deallocate(true);
    return output_tensors.mm;
}

}  // namespace ttnn::operations::experimental::ccl
