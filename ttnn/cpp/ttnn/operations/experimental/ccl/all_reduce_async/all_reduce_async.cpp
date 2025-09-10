// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_async.hpp"

#include "ttnn/operations/experimental/ccl/reduce_scatter_async/device/reduce_scatter_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_command_processor_async/device/all_gather_command_processor_async_op.hpp"
#include "device/all_reduce_async_op.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/experimental/ccl/all_broadcast_async/device/all_broadcast_async_op.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::ccl {

uint32_t finding_scatter_dim(const ttnn::Shape& input_tensor_padded_shape, const Layout& layout, size_t num_workers) {
    // iterate until we find a dimension that is divisible by num_workers
    TT_FATAL(input_tensor_padded_shape.size() == 4, "Expected input tensor to have 4 dimensions");
    if (layout == Layout::TILE) {
        ttnn::Shape input_tensor_shape_in_tiles{
            input_tensor_padded_shape[0],
            input_tensor_padded_shape[1],
            input_tensor_padded_shape[2] / tt::constants::TILE_HEIGHT,
            input_tensor_padded_shape[3] / tt::constants::TILE_WIDTH};
        for (uint32_t dim = 0; dim < 4; ++dim) {
            if (input_tensor_shape_in_tiles[dim] % num_workers == 0) {
                log_debug(
                    tt::LogOp,
                    "Found scatter dimension {} for input tensor with padded shape {}",
                    dim,
                    input_tensor_padded_shape);
                return dim;
            }
        }
    } else {
        for (uint32_t dim = 0; dim < 4; ++dim) {
            if (input_tensor_padded_shape[dim] % num_workers == 0) {
                log_debug(
                    tt::LogOp,
                    "Found scatter dimension {} for input tensor with padded shape {}",
                    dim,
                    input_tensor_padded_shape);
                return dim;
            }
        }
    }

    return input_tensor_padded_shape.size();
}

Tensor strided_reduce(
    const ttnn::Tensor& gathered_tensor,
    int reduce_dim,
    uint32_t num_devices,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt) {
    const auto& input_shape = gathered_tensor.logical_shape();
    int rank = input_shape.size();

    // 1. Reshape to expose the device dimension
    uint32_t dim_to_split_size = input_shape[reduce_dim];
    TT_FATAL(
        dim_to_split_size % num_devices == 0,
        "Gathered dimension size ({}) must be divisible by the number of devices ({}).",
        dim_to_split_size,
        num_devices);
    uint32_t local_dim_size = dim_to_split_size / num_devices;

    // if row major convert first to tile layout
    auto input_tensor = gathered_tensor;
    bool is_rm = (gathered_tensor.layout() == Layout::ROW_MAJOR);
    if (is_rm) {
        input_tensor = ttnn::to_layout(gathered_tensor, Layout::TILE);
    }

    bool do_typecast = input_tensor.dtype() == DataType::BFLOAT8_B && reduce_dim == 2;
    if (do_typecast) {
        input_tensor = ttnn::typecast(input_tensor, DataType::BFLOAT16);
    }
    ttnn::SmallVector<uint32_t> reshape_dims_vec;
    for (int i = 0; i < rank; ++i) {
        if (i == reduce_dim) {
            reshape_dims_vec.push_back(num_devices);
            reshape_dims_vec.push_back(local_dim_size);
        } else {
            reshape_dims_vec.push_back(input_shape[i]);
        }
    }
    ttnn::Shape reshaped_shape(reshape_dims_vec);
    auto reshaped_tensor = ttnn::reshape(input_tensor, reshaped_shape);

    // 2. Transpose to bring the device dimension next to the data to be reduced
    // Shape is now [N, num_devices, local_rows, H, W]
    // We want to sum over num_devices, so we transpose it with local_rows
    // New shape: [N, local_rows, num_devices, H, W]
    int device_dim = reduce_dim;
    int local_rows_dim = reduce_dim + 1;
    auto transposed_tensor = ttnn::transpose(reshaped_tensor, device_dim, local_rows_dim);

    // 3. Reduce along the device dimension (which is now at `local_rows_dim`)
    auto reduced_tensor = ttnn::sum(transposed_tensor, local_rows_dim, false, memory_config);

    // The shape of reduced_tensor is now [N, local_rows, H, W], which is the desired final shape.
    if (do_typecast) {
        reduced_tensor = ttnn::typecast(reduced_tensor, DataType::BFLOAT8_B);
    }
    if (is_rm) {
        return ttnn::to_layout(reduced_tensor, Layout::ROW_MAJOR);
    }
    return reduced_tensor;
}

ttnn::Tensor ExecuteAllReduceAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const uint32_t num_devices,
    const std::vector<GlobalSemaphore>& barrier_semaphores,
    const std::vector<GlobalSemaphore>& rs_global_semaphores,
    const std::vector<GlobalSemaphore>& ag_global_semaphores,
    ttnn::operations::reduction::ReduceType math_op,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());
    uint32_t dim = finding_scatter_dim(
        input_tensor.padded_shape(),
        input_tensor.layout(),
        ttnn::ccl::get_active_physical_devices(input_tensor).size());
    auto composite_dim = (dim == input_tensor.padded_shape().size()) ? dim - 1 : dim;
    bool composite_all_gather =
        composite_common::use_composite_all_gather(input_tensor, composite_dim, out_memory_config);
    bool composite_reduce_scatter =
        composite_common::use_composite_reduce_scatter(input_tensor, composite_dim, std::nullopt);

    if (composite_all_gather || composite_reduce_scatter || (dim != composite_dim)) {
        // All reduce = all gather + local reduce
        auto gather_tensor = composite_common::composite_all_gather(
            input_tensor,
            composite_dim,
            num_preferred_links.value_or(1),
            out_memory_config,
            worker_subdevice_id_opt,
            std::nullopt);
        auto sum_tensor =
            strided_reduce(gather_tensor, static_cast<int>(composite_dim), num_devices, out_memory_config);
        return sum_tensor;
    }
    // Reduce scatter + all gather
    bool use_llama_sharded = composite_common::use_all_gather_async_llama_sharded(input_tensor, out_memory_config);
    ttnn::Tensor scattered_tensor = ttnn::operations::experimental::ccl::reduce_scatter_minimal_async(
        input_tensor,
        std::nullopt,
        dim,
        rs_global_semaphores,
        barrier_semaphores[0],
        num_preferred_links.value_or(1),
        out_memory_config,
        std::nullopt,
        topology,
        worker_subdevice_id_opt);
    return ttnn::operations::experimental::ccl::all_gather_async(
        scattered_tensor,
        dim,
        ag_global_semaphores,
        num_preferred_links.value_or(1),
        out_memory_config,
        topology,
        worker_subdevice_id_opt,
        false,
        use_llama_sharded,
        barrier_semaphores[1]);
}

ttnn::Tensor ExecuteAllReduceAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const std::vector<GlobalSemaphore>& barrier_semaphores,
    const std::vector<GlobalSemaphore>& rs_global_semaphores,
    const std::vector<GlobalSemaphore>& ag_global_semaphores,
    ttnn::operations::reduction::ReduceType math_op,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());
    const auto& mesh_view = mesh_device.get_view();
    std::vector<IDevice*> devices =
        (cluster_axis == 0) ? mesh_view.get_devices_on_column(0) : mesh_view.get_devices_on_row(0);
    uint32_t dim = finding_scatter_dim(input_tensor.padded_shape(), input_tensor.layout(), devices.size());
    auto composite_dim = (dim == input_tensor.padded_shape().size()) ? dim - 1 : dim;
    bool composite_all_gather =
        composite_common::use_composite_all_gather(input_tensor, composite_dim, out_memory_config);
    bool composite_reduce_scatter =
        composite_common::use_composite_reduce_scatter(input_tensor, composite_dim, cluster_axis);

    if (composite_all_gather || composite_reduce_scatter || (dim != composite_dim)) {
        // All reduce = all gather + local reduce
        auto gather_tensor = composite_common::composite_all_gather(
            input_tensor,
            composite_dim,
            num_preferred_links.value_or(1),
            out_memory_config,
            worker_subdevice_id_opt,
            cluster_axis);

        auto sum_tensor =
            strided_reduce(gather_tensor, static_cast<int>(composite_dim), devices.size(), out_memory_config);

        return sum_tensor;
    }
    // Reduce scatter + all gather
    bool use_llama_sharded = composite_common::use_all_gather_async_llama_sharded(input_tensor, out_memory_config);
    ttnn::Tensor scattered_tensor = ttnn::operations::experimental::ccl::reduce_scatter_minimal_async(
        input_tensor,
        std::nullopt,
        dim,
        rs_global_semaphores,
        barrier_semaphores[0],
        num_preferred_links.value_or(1),
        out_memory_config,
        std::nullopt,
        topology,
        worker_subdevice_id_opt,
        cluster_axis);
    return ttnn::operations::experimental::ccl::all_gather_async(
        scattered_tensor,
        dim,
        cluster_axis,
        mesh_device,
        topology,
        ag_global_semaphores,
        std::nullopt,
        out_memory_config,
        num_preferred_links.value_or(1),
        worker_subdevice_id_opt,
        false,
        use_llama_sharded,
        barrier_semaphores[1]);
}

ttnn::Tensor ExecuteAllReduceAsync::invoke(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& buffer_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<const DataType> dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt,
    bool use_noc1_only,
    bool use_optimal_ccl_for_llama) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());
    return ttnn::operations::experimental::ccl::all_reduce_async(
        input_tensor,
        buffer_tensor,
        cluster_axis,
        mesh_device,
        topology,
        multi_device_global_semaphore,
        dtype,
        out_memory_config,
        num_preferred_links,
        worker_subdevice_id_opt,
        use_noc1_only,
        use_optimal_ccl_for_llama);
}

std::vector<ttnn::Tensor> ExecuteAllReduceAsync::invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    ttnn::Tensor& buffer_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const std::optional<const DataType> dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt,
    bool use_noc1_only,
    bool use_optimal_ccl_for_llama) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensors.at(0).memory_config());
    return ttnn::operations::experimental::ccl::all_reduce_async(
        input_tensors,
        buffer_tensor,
        cluster_axis,
        mesh_device,
        topology,
        multi_device_global_semaphore,
        dtype,
        out_memory_config,
        num_preferred_links,
        worker_subdevice_id_opt,
        use_noc1_only,
        use_optimal_ccl_for_llama);
}

}  // namespace ttnn::operations::experimental::ccl
