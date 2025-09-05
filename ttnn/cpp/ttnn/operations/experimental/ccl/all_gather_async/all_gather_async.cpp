// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_async.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_broadcast_async/device/all_broadcast_async_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

bool use_all_gather_async_llama_sharded(const Tensor& input_tensor, const MemoryConfig& output_mem_config) {
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

bool use_composite_all_gather(
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
        return true;
    }

    // Use composite if gathering on dim 0 or dim 1, and input_shape[0] != 1 or input_shape[1] != 1
    if ((gather_dim == 0 || gather_dim == 1) && (input_shape[0] != 1 || input_shape[1] != 1)) {
        return true;
    }

    return false;
}

ttnn::Tensor composite_all_gather(
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

    return all_gather_output_tensor;
}

ttnn::Tensor ExecuteAllGatherAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    bool use_optimal_ccl_for_llama,
    const std::optional<GlobalSemaphore>& barrier_semaphore) {
    bool composite_all_gather_case = use_composite_all_gather(input_tensor, dim, memory_config);
    bool all_gather_async_llama_sharded_case =
        use_all_gather_async_llama_sharded(input_tensor, memory_config.value_or(input_tensor.memory_config()));
    if (composite_all_gather_case && !all_gather_async_llama_sharded_case) {
        return composite_all_gather(
            input_tensor,
            dim,
            num_links,
            memory_config,
            subdevice_id,
            /*cluster_axis*/ std::nullopt);
    } else {
        return ttnn::operations::experimental::ccl::all_gather_async(
            input_tensor,
            dim,
            multi_device_global_semaphore,
            barrier_semaphore.has_value(),
            num_links,
            memory_config,
            topology,
            subdevice_id,
            all_gather_async_llama_sharded_case,
            use_optimal_ccl_for_llama);
    }
}

ttnn::Tensor ExecuteAllGatherAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis,
    bool use_optimal_ccl_for_llama,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel) {
    bool composite_all_gather_case = use_composite_all_gather(input_tensor, dim, memory_config);
    bool all_gather_async_llama_sharded_case =
        use_all_gather_async_llama_sharded(input_tensor, memory_config.value_or(input_tensor.memory_config()));
    if (composite_all_gather_case && !all_gather_async_llama_sharded_case) {
        return composite_all_gather(input_tensor, dim, num_links, memory_config, subdevice_id, cluster_axis);
    } else {
        return ttnn::operations::experimental::ccl::all_gather_async(
            input_tensor,
            persistent_output_buffer,
            dim,
            multi_device_global_semaphore,
            barrier_semaphore.has_value(),
            num_links,
            memory_config,
            topology,
            subdevice_id,
            cluster_axis,
            all_gather_async_llama_sharded_case,
            use_optimal_ccl_for_llama,
            chunks_per_sync,
            num_workers_per_link,
            num_buffers_per_channel);
    }
}

ttnn::Tensor ExecuteAllGatherAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    bool use_optimal_ccl_for_llama,
    const std::optional<GlobalSemaphore>& barrier_semaphore) {
    bool composite_all_gather_case = use_composite_all_gather(input_tensor, dim, memory_config);
    bool all_gather_async_llama_sharded_case =
        use_all_gather_async_llama_sharded(input_tensor, memory_config.value_or(input_tensor.memory_config()));
    if (composite_all_gather_case && !all_gather_async_llama_sharded_case) {
        return composite_all_gather(
            input_tensor, dim, num_preferred_links.value_or(1), memory_config, subdevice_id, cluster_axis);
    } else {
        return ttnn::operations::experimental::ccl::all_gather_async(
            input_tensor,
            dim,
            cluster_axis,
            mesh_device,
            topology,
            multi_device_global_semaphore,
            barrier_semaphore.has_value(),
            persistent_output_tensor,
            memory_config,
            num_preferred_links,
            subdevice_id,
            all_gather_async_llama_sharded_case,
            use_optimal_ccl_for_llama);
    }
}

}  // namespace ttnn::operations::experimental::ccl
