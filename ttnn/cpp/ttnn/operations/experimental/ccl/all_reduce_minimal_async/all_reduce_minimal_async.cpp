// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_minimal_async.hpp"

#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/experimental/ccl/all_broadcast_async/device/all_broadcast_async_op.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::ccl {

bool using_composite_reduce_scatter(
    const ttnn::Tensor& input_tensor, const int32_t dim, std::optional<uint32_t> cluster_axis) {
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    uint32_t tile_width = tile_shape[1];

    int32_t rank = input_tensor.logical_shape().rank();
    int32_t scatter_dim = (dim < 0) ? rank + dim : dim;

    uint32_t num_devices;
    if (cluster_axis.has_value()) {
        auto mesh_device = input_tensor.device();
        const auto& mesh_view = mesh_device->get_view();
        num_devices = (cluster_axis.value() == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    } else {
        num_devices = ttnn::ccl::get_active_physical_devices(input_tensor).size();
    }

    // Must scatter evenly
    auto input_shape = input_tensor.logical_shape();
    if (input_shape[scatter_dim] % num_devices != 0) {
        return false;
    }

    // Use composite for row major tensors
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        return true;
    }

    // Use composite if scattering on a dim that isn't 3
    if (scatter_dim != 3) {
        // printf("composite reduce scatter due to scatter dim not 3\n");
        return true;
    }

    // Use composite if tiled and scattering on padded dim 3
    auto output_shape = input_shape;
    output_shape[scatter_dim] /= num_devices;
    if (scatter_dim == 3 && output_shape[scatter_dim] % tile_width != 0) {
        // printf("composite reduce scatter due to scattering on padded dim 3\n");
        return true;
    }

    return false;
}

bool using_all_gather_async_llama_sharded(const Tensor& input_tensor, const MemoryConfig& output_mem_config) {
    auto input_tensor_shape = input_tensor.padded_shape();
    auto input_tensor_page_layout = input_tensor.layout();
    auto input_tensor_memory_config = input_tensor.memory_config();
    bool input_is_sharded = input_tensor_memory_config.shard_spec().has_value();
    bool output_is_sharded = output_mem_config.shard_spec().has_value();

    log_trace(tt::LogOp, "[select_version] input_tensor_shape: {}", input_tensor_shape);
    log_trace(tt::LogOp, "[select_version] input_tensor_memory_config: {}", input_tensor_memory_config);
    log_trace(tt::LogOp, "[select_version] output_mem_config: {}", output_mem_config);

    log_trace(tt::LogOp, "[select_version] input_is_sharded: {}", input_is_sharded);
    log_trace(tt::LogOp, "[select_version] output_is_sharded: {}", output_is_sharded);

    // Check for minimal sharded case
    if (input_is_sharded && output_is_sharded) {
        uint32_t input_shard_num_cores = input_tensor_memory_config.shard_spec()->grid.num_cores();
        uint32_t output_shard_num_cores = output_mem_config.shard_spec()->grid.num_cores();

        log_trace(tt::LogOp, "[select_version] input_shard_num_cores: {}", input_shard_num_cores);
        log_trace(tt::LogOp, "[select_version] output_shard_num_cores: {}", output_shard_num_cores);

        log_trace(
            tt::LogOp,
            "[select_version] input_tensor_memory_config.shard_spec()->shape: {}",
            input_tensor_memory_config.shard_spec()->shape);
        log_trace(
            tt::LogOp,
            "[select_version] output_mem_config.shard_spec()->shape: {}",
            output_mem_config.shard_spec()->shape);

        // Check for llama post binary mult+silu case
        if (input_tensor_shape[0] == 1 && input_tensor_shape[1] == 1 && input_tensor_shape[2] == 32 &&
            input_tensor_shape[3] == 960 && input_tensor_memory_config.buffer_type() == BufferType::L1 &&
            output_mem_config.buffer_type() == BufferType::L1 &&
            input_tensor_memory_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            output_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            input_tensor_memory_config.shard_spec()->shape[0] == 32 &&
            input_tensor_memory_config.shard_spec()->shape[1] == 32 && output_mem_config.shard_spec()->shape[0] == 32 &&
            output_mem_config.shard_spec()->shape[1] == 160 && input_shard_num_cores == 30 &&
            output_shard_num_cores == 24) {
            log_trace(
                tt::LogOp,
                "Matching conditions for Llama post binary mult+silu, using LLAMA_MINIMAL_SHARDED implementation");
            return true;
        }

        // Check for llama post SDPA case
        if (input_tensor_shape[0] == 1 && input_tensor_shape[1] == 8 && input_tensor_shape[2] == 32 &&
            input_tensor_shape[3] == 128 && input_tensor_memory_config.buffer_type() == BufferType::L1 &&
            output_mem_config.buffer_type() == BufferType::L1 &&
            input_tensor_memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
            output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
            input_tensor_memory_config.shard_spec()->shape[0] == 32 &&
            input_tensor_memory_config.shard_spec()->shape[1] == 128 &&
            output_mem_config.shard_spec()->shape[0] == 32 && output_mem_config.shard_spec()->shape[1] == 128 &&
            input_shard_num_cores == 8 && output_shard_num_cores == 32) {
            log_trace(tt::LogOp, "Matching conditions for Llama post SDPA, using LLAMA_MINIMAL_SHARDED implementation");
            return true;
        }

        // Check for llama rms norm case
        if (input_tensor_shape[0] == 1 && input_tensor_shape[1] == 1 && input_tensor_shape[2] == 32 &&
            input_tensor_shape[3] == 32 && input_tensor_memory_config.buffer_type() == BufferType::L1 &&
            output_mem_config.buffer_type() == BufferType::L1 &&
            input_tensor_memory_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            output_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            input_tensor_memory_config.shard_spec()->shape[0] == 32 &&
            input_tensor_memory_config.shard_spec()->shape[1] == 32 && output_mem_config.shard_spec()->shape[0] == 32 &&
            output_mem_config.shard_spec()->shape[1] == 128 && input_shard_num_cores == 1 &&
            output_shard_num_cores == 1) {
            log_trace(
                tt::LogOp, "Matching conditions for Llama rms norm case, using LLAMA_MINIMAL_SHARDED implementation");
            return true;
        }
    }

    return false;
}

bool using_composite_all_gather(
    const ttnn::Tensor& input_tensor, const int32_t dim, const std::optional<ttnn::MemoryConfig>& memory_config) {
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    auto input_shape = input_tensor.logical_shape();

    int32_t rank = input_tensor.logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    auto input_memory_config = input_tensor.memory_config();
    auto output_memory_config = memory_config.value_or(input_memory_config);

    // Use composite for row-major tensors
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        return true;
    }

    // Use composite if tiled and padded on the gather dim
    bool is_tiled_and_padded_on_gather_dim =
        input_tensor.layout() == Layout::TILE && ((gather_dim == 2 && input_shape[2] % tile_height != 0) ||
                                                  (gather_dim == 3 && input_shape[3] % tile_width != 0));
    if (is_tiled_and_padded_on_gather_dim) {
        // printf("composite all gather due to tiled and padded on gather dim\n");
        return true;
    }

    // Use composite if gathering on dim 0 or dim 1, and input_shape[0] != 1 or input_shape[1] != 1
    if ((gather_dim == 0 || gather_dim == 1) && (input_shape[0] != 1 || input_shape[1] != 1)) {
        // printf("composite all gather due to gathering on dim 0 or 1 and input_shape[0] != 1 or input_shape[1] !=
        // 1\n");
        return true;
    }

    return false;
}

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

ttnn::Tensor all_gather_composite(
    ttnn::Tensor input_tensor,
    const int32_t dim,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis) {
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    auto input_shape = input_tensor.logical_shape();

    int32_t rank = input_tensor.logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    bool is_tiled_and_not_tile_aligned = input_tensor.layout() == Layout::TILE &&
                                         (input_shape[2] % tile_height != 0 || input_shape[3] % tile_width != 0);

    // If we need to convert to row-major, then if the input dtype is bfloat8_b we need to typecast before untilizing
    // and after re-tilizing
    DataType input_dtype = input_tensor.dtype();
    bool convert_to_bfloat16_for_composite = is_tiled_and_not_tile_aligned && input_dtype == DataType::BFLOAT8_B;
    auto input_memory_config = input_tensor.memory_config();
    auto output_memory_config = memory_config.value_or(input_memory_config);

    // Convert to row major
    if (is_tiled_and_not_tile_aligned) {
        // If input is tiled bfloat8_b, convert to bfloat16 to do the all_broadcast_async + concat
        if (convert_to_bfloat16_for_composite) {
            input_tensor = ttnn::typecast(input_tensor, DataType::BFLOAT16);
        }
        input_tensor = ttnn::to_layout(input_tensor, Layout::ROW_MAJOR);
    }

    std::vector<ttnn::Tensor> broadcasted_tensors = ttnn::operations::experimental::ccl::all_broadcast_async(
        input_tensor, num_links, input_memory_config, ttnn::ccl::Topology::Linear, cluster_axis, subdevice_id);

    ttnn::Tensor all_gather_output_tensor = ttnn::concat(broadcasted_tensors, gather_dim);
    // Convert back to tiled
    if (is_tiled_and_not_tile_aligned) {
        all_gather_output_tensor = ttnn::to_layout(all_gather_output_tensor, Layout::TILE);
        // If we had to convert the input dtype in order to execute the row-major composite op, convert back to the
        // input dtype
        if (convert_to_bfloat16_for_composite) {
            all_gather_output_tensor = ttnn::typecast(all_gather_output_tensor, input_dtype);
        }
    }

    if (input_memory_config.memory_layout() != output_memory_config.memory_layout()) {
        all_gather_output_tensor = ttnn::to_memory_config(all_gather_output_tensor, output_memory_config);
    }

    // printf("executing composite all gather\n");
    return all_gather_output_tensor;
}

Tensor strided_reduce(
    const ttnn::Tensor& gathered_tensor,
    int reduce_dim,
    int num_devices,
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
    auto reduced_tensor = ttnn::sum(transposed_tensor, local_rows_dim, false, memory_config);

    // The shape of reduced_tensor is now [N, local_rows, H, W], which is the desired final shape.
    if (is_rm) {
        return ttnn::to_layout(reduced_tensor, Layout::ROW_MAJOR);
    }
    return reduced_tensor;
}

ttnn::Tensor ExecuteAllReduceMinimalAsync::invoke(
    const ttnn::Tensor& input_tensor,
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
    bool composite_all_gather = using_composite_all_gather(input_tensor, composite_dim, out_memory_config);
    bool composite_reduce_scatter = using_composite_reduce_scatter(input_tensor, composite_dim, std::nullopt);

    if (composite_all_gather || composite_reduce_scatter || (dim != composite_dim)) {
        // All reduce = all gather + local reduce
        auto gather_tensor = ttnn::operations::experimental::ccl::all_gather_async(
            input_tensor,
            composite_dim,
            ag_global_semaphores,
            num_preferred_links.value_or(1),
            out_memory_config,
            topology,
            worker_subdevice_id_opt,
            false,
            false,
            barrier_semaphores[0]);
        return ttnn::sum(
            gather_tensor,
            static_cast<int>(composite_dim),
            true,  // keepdim
            out_memory_config,
            std::nullopt);
    }
    // Reduce scatter + all gather
    bool use_llama_sharded = using_all_gather_async_llama_sharded(input_tensor, out_memory_config);
    ttnn::Tensor scattered_tensor = ttnn::operations::experimental::ccl::reduce_scatter_minimal_async(
        input_tensor,
        std::nullopt,
        dim,
        rs_global_semaphores,   //
        barrier_semaphores[0],  //
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

ttnn::Tensor ExecuteAllReduceMinimalAsync::invoke(
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
    printf("selected dim: %u\n", dim);
    auto composite_dim = (dim == input_tensor.padded_shape().size()) ? dim - 1 : dim;
    printf("composite_dim: %d\n", composite_dim);
    bool composite_all_gather = using_composite_all_gather(input_tensor, composite_dim, out_memory_config);
    bool composite_reduce_scatter = using_composite_reduce_scatter(input_tensor, composite_dim, cluster_axis);
    // printf("composite_all_gather: %d, composite_reduce_scatter: %d\n", composite_all_gather,
    // composite_reduce_scatter);
    if (composite_all_gather || composite_reduce_scatter || (dim != composite_dim)) {
        // All reduce = all gather + local reduce
        printf("using all gather + local reduce\n");
        auto gather_tensor = all_gather_composite(
            input_tensor,
            composite_dim,
            num_preferred_links.value_or(1),
            out_memory_config,
            worker_subdevice_id_opt,
            cluster_axis);
        printf(
            "after composite all gather with size: %u %u %u %u\n",
            gather_tensor.logical_shape()[0],
            gather_tensor.logical_shape()[1],
            gather_tensor.logical_shape()[2],
            gather_tensor.logical_shape()[3]);
        /*
        auto gather_tensor = ttnn::operations::experimental::ccl::all_gather_async(
            input_tensor,
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
            false,
            barrier_semaphores[0]);
        */
        // printf("gathered tensor shape: %u %u %u %u\n", gather_tensor.logical_shape()[0],
        // gather_tensor.logical_shape()[1], gather_tensor.logical_shape()[2], gather_tensor.logical_shape()[3]);

        auto sum_tensor =
            strided_reduce(gather_tensor, static_cast<int>(composite_dim), devices.size(), out_memory_config);
        printf(
            "after sum tensor with size: %u %u %u %u\n",
            sum_tensor.logical_shape()[0],
            sum_tensor.logical_shape()[1],
            sum_tensor.logical_shape()[2],
            sum_tensor.logical_shape()[3]);
        // auto sum_tensor = ttnn::sum(
        //             gather_tensor,
        //             static_cast<int>(dim),
        //             true, // keepdim
        //             out_memory_config, // memory_config
        //             std::nullopt);
        // printf("summed tensor shape: %u %u %u %u\n", sum_tensor.logical_shape()[0], sum_tensor.logical_shape()[1],
        // sum_tensor.logical_shape()[2], sum_tensor.logical_shape()[3]);
        return sum_tensor;
    }
    // Reduce scatter + all gather
    printf("using reduce scatter + all gather\n");
    bool use_llama_sharded = using_all_gather_async_llama_sharded(input_tensor, out_memory_config);
    ttnn::Tensor scattered_tensor = ttnn::operations::experimental::ccl::reduce_scatter_minimal_async(
        input_tensor,
        std::nullopt,
        dim,
        rs_global_semaphores,   //
        barrier_semaphores[0],  //
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

}  // namespace ttnn::operations::experimental::ccl
