// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "all_reduce_async.hpp"

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "device/all_reduce_async_device_operation.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::experimental::ccl {

namespace detail {
uint32_t finding_scatter_dim(const ttnn::Tensor& input_tensor, size_t num_workers) {
    // iterate until we find a dimension that is divisible by num_workers

    const auto& padded_shape = input_tensor.padded_shape();
    const auto layout = input_tensor.layout();
    const auto rank = padded_shape.rank();

    TT_FATAL(rank >= 2, "Expected input tensor to be of at least rank 2");

    ttnn::SmallVector<uint32_t> shape_vec(padded_shape.cbegin(), padded_shape.cend());
    if (layout == Layout::TILE) {
        const auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
        auto tile_shape_it = tile_shape.crbegin();
        std::for_each(
            shape_vec.rbegin(), shape_vec.rbegin() + 2, [&tile_shape_it](auto& x) { x /= *(tile_shape_it++); });
    }
    auto dim_it = std::find_if(
        shape_vec.crbegin(), shape_vec.crend(), [num_workers](const auto& x) { return x % num_workers == 0; });

    auto end_it = shape_vec.crend();
    return (dim_it == end_it) ? rank : end_it - dim_it - 1;  // forward index
}

// True 2D mesh when both mesh axes have more than one device.
bool is_true_2d_mesh(const ttnn::Tensor& input_tensor, tt::tt_fabric::Topology topology) {
    if (topology != tt::tt_fabric::Topology::Mesh && topology != tt::tt_fabric::Topology::Torus) {
        return false;
    }
    const auto mesh_shape = input_tensor.device()->shape();
    return mesh_shape.dims() >= 2 && mesh_shape[0] > 1 && mesh_shape[1] > 1;
}
}  // namespace detail

Tensor local_sum(
    const ttnn::Tensor& gathered_tensor,
    int reduce_dim,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt) {
    // if row major convert first to tile layout
    auto input_tensor = gathered_tensor;
    bool is_rm = (gathered_tensor.layout() == Layout::ROW_MAJOR);
    if (is_rm) {
        input_tensor = ttnn::to_layout(gathered_tensor, Layout::TILE);
    }

    bool do_typecast = false;
    // moreh_sum does not support bfloat8_b
    if (input_tensor.dtype() == DataType::BFLOAT8_B) {
        // cast up to bfloat16 prior to sum
        do_typecast = true;
        input_tensor = ttnn::typecast(input_tensor, DataType::BFLOAT16);
    }

    auto sum_tensor = ttnn::moreh_sum(
        input_tensor,
        reduce_dim,
        /* keep_dim */ true,
        /* output */ std::nullopt,
        memory_config,
        /* device kernel config */ std::nullopt);

    if (do_typecast) {
        // cast back down to bfloat8_b
        sum_tensor = ttnn::typecast(sum_tensor, DataType::BFLOAT8_B);
    }
    if (is_rm) {
        return ttnn::to_layout(sum_tensor, Layout::ROW_MAJOR);
    }
    return sum_tensor;
}

// moreh sum does not support float32 datatye
Tensor local_sum_float32(
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
    auto sum_tensor = ttnn::sum(transposed_tensor, local_rows_dim, false, memory_config);
    if (is_rm) {
        return ttnn::to_layout(sum_tensor, Layout::ROW_MAJOR);
    }
    return sum_tensor;
}

ttnn::Tensor ExecuteAllReduceAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const uint32_t num_devices,
    const std::vector<GlobalSemaphore>& barrier_semaphores,
    const std::vector<GlobalSemaphore>& rs_global_semaphores,
    const std::vector<GlobalSemaphore>& ag_global_semaphores,
    ttnn::operations::reduction::ReduceType /*math_op*/,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) {
    topology = ::ttnn::ccl::get_usable_topology(input_tensor, topology, std::nullopt);
    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());
    bool input_is_sharded = input_tensor.memory_config().is_sharded();
    uint32_t dim =
        detail::finding_scatter_dim(input_tensor, ttnn::ccl::get_active_physical_devices(input_tensor).size());

    auto padded_tensor = input_tensor;
    auto initial_shape = input_tensor.logical_shape();
    auto composite_dim = (dim == padded_tensor.padded_shape().size()) ? 0 : dim;
    bool composite_all_gather =
        composite_common::use_composite_all_gather(padded_tensor, composite_dim, out_memory_config);
    bool composite_reduce_scatter =
        composite_common::use_composite_reduce_scatter(padded_tensor, composite_dim, std::nullopt);

    // when input is sharded, shard specs are not compatible with the intermediate tensor shapes of the composite ops
    // convert to interleaved in this case
    auto interleaved_tensor = padded_tensor;
    bool change_mem_config = input_is_sharded;
    if (change_mem_config) {
        MemoryConfig working_memory_config{TensorMemoryLayout::INTERLEAVED, input_tensor.memory_config().buffer_type()};
        interleaved_tensor = ttnn::sharded_to_interleaved(padded_tensor, working_memory_config, std::nullopt);
    }

    const bool composite_for_2d_mesh = tt::tt_fabric::GetFabricConfig() == tt::tt_fabric::FabricConfig::FABRIC_2D &&
                                       detail::is_true_2d_mesh(input_tensor, topology);

    if (composite_all_gather || composite_reduce_scatter || (dim != composite_dim) || composite_for_2d_mesh) {
        log_debug(tt::LogOp, "Using composite all gather + local reduce");

        // All reduce = all gather + local reduce
        composite_dim = 0;
        auto reshaped_tensor = ttnn::reshape(
            interleaved_tensor,
            ttnn::Shape({1, initial_shape[0] * initial_shape[1], initial_shape[2], initial_shape[3]}));
        interleaved_tensor.deallocate();
        auto gather_tensor = composite_common::composite_all_gather(
            reshaped_tensor,
            composite_dim,
            num_preferred_links.value_or(1),
            out_memory_config,
            worker_subdevice_id_opt,
            std::nullopt);
        reshaped_tensor.deallocate();

        bool is_float32 = (input_tensor.dtype() == DataType::FLOAT32);
        auto sum_tensor =
            is_float32
                ? local_sum_float32(gather_tensor, static_cast<int>(composite_dim), num_devices, out_memory_config)
                : local_sum(gather_tensor, static_cast<int>(composite_dim), out_memory_config);
        gather_tensor.deallocate();

        return ttnn::reshape(sum_tensor, initial_shape);
    }
    // Reduce scatter + all gather
    bool use_llama_sharded = composite_common::use_all_gather_async_llama_sharded(padded_tensor, out_memory_config);
    padded_tensor.deallocate();
    log_debug(tt::LogOp, "Using reduce scatter + all gather");
    ttnn::Tensor scattered_tensor = ttnn::experimental::reduce_scatter_minimal_async(
        interleaved_tensor,
        std::nullopt,
        dim,
        rs_global_semaphores,
        barrier_semaphores[0],
        num_preferred_links.value_or(1),
        change_mem_config ? std::nullopt : std::optional<MemoryConfig>(out_memory_config),
        std::nullopt,
        topology,
        worker_subdevice_id_opt);
    interleaved_tensor.deallocate();
    auto gathered = ttnn::prim::all_gather_async(
        scattered_tensor,
        /*persistent_output_buffer*/ std::nullopt,
        dim,
        ag_global_semaphores,
        num_preferred_links.value_or(1),
        change_mem_config ? std::nullopt : std::optional<MemoryConfig>(out_memory_config),
        topology,
        worker_subdevice_id_opt,
        /*cluster_axis*/ std::nullopt,
        /*use_optimal_ccl_for_llama*/ false,
        use_llama_sharded,
        barrier_semaphores[1],
        /*chunks_per_sync*/ std::nullopt,
        /*num_workers_per_link*/ std::nullopt,
        /*num_buffers_per_channel*/ std::nullopt,
        /*reverse_order*/ false,
        /*sub_core_grid*/ std::nullopt,
        /*mesh_device*/ nullptr);
    scattered_tensor.deallocate();
    if (change_mem_config) {
        gathered = ttnn::to_memory_config(gathered, out_memory_config, std::nullopt);
    }
    return gathered;
}

ttnn::Tensor ExecuteAllReduceAsync::invoke(
    const ttnn::Tensor& input_tensor,
    std::optional<uint32_t> cluster_axis,
    MeshDevice& mesh_device,
    const std::optional<std::vector<GlobalSemaphore>>& barrier_semaphores,
    const std::optional<std::vector<GlobalSemaphore>>& rs_global_semaphores,
    const std::optional<std::vector<GlobalSemaphore>>& ag_global_semaphores,
    ttnn::operations::reduction::ReduceType /*math_op*/,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<ttnn::ccl::Topology> topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) {
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());
    bool input_is_sharded = input_tensor.memory_config().is_sharded();
    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    uint32_t dim = detail::finding_scatter_dim(input_tensor, num_devices);
    auto padded_tensor = input_tensor;
    auto initial_shape = input_tensor.logical_shape();

    // convert sharded tensors to interleaved because the shard specs are not compatible with composite intermediates
    bool change_mem_config = input_is_sharded;
    auto interleaved_tensor = padded_tensor;
    if (change_mem_config) {
        MemoryConfig working_memory_config{TensorMemoryLayout::INTERLEAVED, input_tensor.memory_config().buffer_type()};
        interleaved_tensor = ttnn::sharded_to_interleaved(padded_tensor, working_memory_config, std::nullopt);
    }

    // logic for taking the AG+local reduce code path
    auto composite_dim = (dim == padded_tensor.padded_shape().size()) ? 0 : dim;
    bool composite_all_gather =
        composite_common::use_composite_all_gather(padded_tensor, composite_dim, out_memory_config);
    bool composite_reduce_scatter =
        composite_common::use_composite_reduce_scatter(padded_tensor, composite_dim, cluster_axis);
    const bool composite_for_2d_mesh = tt::tt_fabric::GetFabricConfig() == tt::tt_fabric::FabricConfig::FABRIC_2D &&
                                       detail::is_true_2d_mesh(input_tensor, topology_);

    if (composite_all_gather || composite_reduce_scatter || (dim != composite_dim) || composite_for_2d_mesh) {
        log_debug(tt::LogOp, "Using composite all gather + local reduce");
        // All reduce = all gather + local reduce
        composite_dim = 0;

        ttnn::SmallVector<uint32_t> ag_shape_vec(initial_shape.rank());
        std::copy(initial_shape.cbegin() + 2, initial_shape.cend(), ag_shape_vec.begin() + 2);
        ag_shape_vec[0] = 1;
        ag_shape_vec[1] = initial_shape[0] * initial_shape[1];

        auto reshaped_tensor = ttnn::reshape(interleaved_tensor, ttnn::Shape(ag_shape_vec));
        interleaved_tensor.deallocate();
        auto gather_tensor = composite_common::composite_all_gather(
            reshaped_tensor,
            composite_dim,
            num_preferred_links.value_or(1),
            change_mem_config ? std::nullopt : std::optional<MemoryConfig>(out_memory_config),
            worker_subdevice_id_opt,
            cluster_axis);
        reshaped_tensor.deallocate();

        bool is_float32 = (input_tensor.dtype() == DataType::FLOAT32);
        auto sum_tensor =
            is_float32
                ? local_sum_float32(gather_tensor, static_cast<int>(composite_dim), num_devices, out_memory_config)
                : local_sum(gather_tensor, static_cast<int>(composite_dim), out_memory_config);
        gather_tensor.deallocate();

        return ttnn::reshape(sum_tensor, initial_shape);
    }

    // Reduce scatter + all gather
    bool use_llama_sharded = composite_common::use_all_gather_async_llama_sharded(padded_tensor, out_memory_config);
    padded_tensor.deallocate();
    log_debug(tt::LogOp, "Using reduce scatter + all gather");
    ttnn::Tensor scattered_tensor;
    if (rs_global_semaphores.has_value() && barrier_semaphores.has_value()) {
        scattered_tensor = ttnn::experimental::reduce_scatter_minimal_async(
            interleaved_tensor,
            std::nullopt,
            dim,
            rs_global_semaphores.value(),
            barrier_semaphores.value()[0],
            num_preferred_links.value_or(1),
            change_mem_config ? std::nullopt : std::optional<MemoryConfig>(out_memory_config),
            std::nullopt,
            topology_,
            worker_subdevice_id_opt,
            cluster_axis);
    } else {
        scattered_tensor = ttnn::reduce_scatter(
            interleaved_tensor,
            dim,
            cluster_axis,
            worker_subdevice_id_opt,
            out_memory_config,
            std::nullopt,
            std::nullopt,
            num_preferred_links,
            topology_,
            std::nullopt,
            std::nullopt,
            std::nullopt);
    }
    interleaved_tensor.deallocate();
    ttnn::Tensor gathered;
    if (ag_global_semaphores.has_value() && barrier_semaphores.has_value()) {
        TT_FATAL(barrier_semaphores.value().size() == 2, "Barrier semaphores must be of size 2");
        TT_FATAL(cluster_axis.has_value(), "Cluster axis is required for all gather");
        gathered = ttnn::prim::all_gather_async(
            scattered_tensor,
            /*persistent_output_buffer*/ std::nullopt,
            dim,
            ag_global_semaphores.value(),
            num_preferred_links.value_or(1),
            change_mem_config ? std::nullopt : std::optional<MemoryConfig>(out_memory_config),
            topology_,
            worker_subdevice_id_opt,
            cluster_axis.value(),
            /*use_optimal_ccl_for_llama*/ false,
            use_llama_sharded,
            barrier_semaphores.value()[1],
            /*chunks_per_sync*/ std::nullopt,
            /*num_workers_per_link*/ std::nullopt,
            /*num_buffers_per_channel*/ std::nullopt,
            /*reverse_order*/ false,
            /*sub_core_grid*/ std::nullopt,
            /*mesh_device*/ &mesh_device);
    } else {
        gathered = ttnn::all_gather(
            scattered_tensor,
            dim,
            cluster_axis,
            worker_subdevice_id_opt,
            out_memory_config,
            std::nullopt,
            num_preferred_links,
            topology_);
    }
    scattered_tensor.deallocate();
    if (change_mem_config) {
        gathered = ttnn::to_memory_config(gathered, out_memory_config, std::nullopt);
    }
    return gathered;
}

ttnn::Tensor ExecuteAllReduceAsync::invoke(
    const ttnn::Tensor& input_tensor,
    uint32_t cluster_axis,
    ttnn::operations::reduction::ReduceType math_op,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<ttnn::MemoryConfig>& /*memory_config*/,
    std::optional<size_t> num_preferred_links,
    std::optional<ttnn::ccl::Topology> topology) {
    auto topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Mesh device is required");
    return ExecuteAllReduceAsync::invoke(
        input_tensor,
        cluster_axis,
        *mesh_device,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        math_op,
        std::nullopt,
        topology_,
        num_preferred_links,
        subdevice_id);
}

ttnn::Tensor ExecuteAllReduceAsync::invoke(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& buffer_tensor,
    const uint32_t cluster_axis,
    MeshDevice& mesh_device,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<const DataType> dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt,
    bool use_noc1_only,
    bool use_optimal_ccl_for_llama) {
    topology = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());

    log_debug(tt::LogOp, "Using minimal all_reduce_async");
    return ttnn::prim::all_reduce_async(
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
    MeshDevice& mesh_device,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const std::optional<const DataType> dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt,
    bool use_noc1_only,
    bool use_optimal_ccl_for_llama) {
    topology = ::ttnn::ccl::get_usable_topology(input_tensors.at(0), topology, cluster_axis);
    MemoryConfig out_memory_config = memory_config.value_or(input_tensors.at(0).memory_config());

    log_debug(tt::LogOp, "Using minimal all_reduce_async with multiple tensors");
    std::vector<ttnn::Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        output_tensors.push_back(ttnn::prim::all_reduce_async(
            input_tensors[i],
            buffer_tensor,
            cluster_axis,
            mesh_device,
            topology,
            multi_device_global_semaphore.global_semaphores[i],
            dtype,
            out_memory_config,
            num_preferred_links,
            worker_subdevice_id_opt,
            use_noc1_only,
            use_optimal_ccl_for_llama));
    }
    return output_tensors;
}

}  // namespace ttnn::operations::experimental::ccl
