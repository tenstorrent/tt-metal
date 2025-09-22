// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/operations/experimental/ccl/common.hpp"

namespace composite_common {

bool use_composite_reduce_scatter(
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
    if (input_tensor.layout() == ttnn::Layout::ROW_MAJOR) {
        return true;
    }

    // Use composite if scattering on a dim that isn't 3
    if (scatter_dim != 3) {
        return true;
    }

    // Use composite if tiled and scattering on padded dim 3
    auto output_shape = input_shape;
    output_shape[scatter_dim] /= num_devices;
    return scatter_dim == 3 && output_shape[scatter_dim] % tile_width != 0;
}

bool use_all_gather_async_llama_sharded(const ttnn::Tensor& input_tensor, const ttnn::MemoryConfig& output_mem_config) {
    auto input_tensor_shape = input_tensor.padded_shape();
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
            input_tensor_shape[3] == 960 && input_tensor_memory_config.buffer_type() == ttnn::BufferType::L1 &&
            output_mem_config.buffer_type() == ttnn::BufferType::L1 &&
            input_tensor_memory_config.memory_layout() == ttnn::TensorMemoryLayout::WIDTH_SHARDED &&
            output_mem_config.memory_layout() == ttnn::TensorMemoryLayout::WIDTH_SHARDED &&
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
            input_tensor_shape[3] == 128 && input_tensor_memory_config.buffer_type() == ttnn::BufferType::L1 &&
            output_mem_config.buffer_type() == ttnn::BufferType::L1 &&
            input_tensor_memory_config.memory_layout() == ttnn::TensorMemoryLayout::HEIGHT_SHARDED &&
            output_mem_config.memory_layout() == ttnn::TensorMemoryLayout::HEIGHT_SHARDED &&
            input_tensor_memory_config.shard_spec()->shape[0] == 32 &&
            input_tensor_memory_config.shard_spec()->shape[1] == 128 &&
            output_mem_config.shard_spec()->shape[0] == 32 && output_mem_config.shard_spec()->shape[1] == 128 &&
            input_shard_num_cores == 8 && output_shard_num_cores == 32) {
            log_trace(tt::LogOp, "Matching conditions for Llama post SDPA, using LLAMA_MINIMAL_SHARDED implementation");
            return true;
        }

        // Check for llama rms norm case
        if (input_tensor_shape[0] == 1 && input_tensor_shape[1] == 1 && input_tensor_shape[2] == 32 &&
            input_tensor_shape[3] == 32 && input_tensor_memory_config.buffer_type() == ttnn::BufferType::L1 &&
            output_mem_config.buffer_type() == ttnn::BufferType::L1 &&
            input_tensor_memory_config.memory_layout() == ttnn::TensorMemoryLayout::WIDTH_SHARDED &&
            output_mem_config.memory_layout() == ttnn::TensorMemoryLayout::WIDTH_SHARDED &&
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
    if (input_tensor.layout() == ttnn::Layout::ROW_MAJOR) {
        return true;
    }

    // Use composite if tiled and padded on the gather dim
    bool is_tiled_and_padded_on_gather_dim =
        input_tensor.layout() == ttnn::Layout::TILE && ((gather_dim == 2 && input_shape[2] % tile_height != 0) ||
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

    bool is_tiled_and_not_tile_aligned = input_tensor.layout() == ttnn::Layout::TILE &&
                                         (input_shape[2] % tile_height != 0 || input_shape[3] % tile_width != 0);

    // If we need to convert to row-major, then if the input dtype is bfloat8_b we need to typecast before untilizing
    // and after re-tilizing
    ttnn::DataType input_dtype = input_tensor.dtype();
    bool convert_to_bfloat16_for_composite = is_tiled_and_not_tile_aligned && input_dtype == ttnn::DataType::BFLOAT8_B;
    auto input_memory_config = input_tensor.memory_config();
    auto output_memory_config = memory_config.value_or(input_memory_config);

    // Convert to row major
    if (is_tiled_and_not_tile_aligned) {
        // If input is tiled bfloat8_b, convert to bfloat16 to do the all_broadcast_async + concat
        if (convert_to_bfloat16_for_composite) {
            input_tensor = ttnn::typecast(input_tensor, ttnn::DataType::BFLOAT16);
        }
        input_tensor = ttnn::to_layout(input_tensor, ttnn::Layout::ROW_MAJOR);
    }

    std::vector<ttnn::Tensor> broadcasted_tensors = ttnn::operations::experimental::ccl::all_broadcast_async(
        input_tensor, num_links, input_memory_config, ttnn::ccl::Topology::Linear, cluster_axis, subdevice_id);

    ttnn::Tensor all_gather_output_tensor = ttnn::concat(broadcasted_tensors, gather_dim);
    // Convert back to tiled
    if (is_tiled_and_not_tile_aligned) {
        all_gather_output_tensor = ttnn::to_layout(all_gather_output_tensor, ttnn::Layout::TILE);
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

// same as above but for vector of mesh
std::vector<ttnn::Tensor> composite_all_gather(
    const std::vector<ttnn::Tensor>& input_tensors,
    const int32_t dim,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis) {
    std::vector<ttnn::Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    for (size_t i = 0; i < input_tensors.size(); i++) {
        output_tensors.push_back(
            composite_all_gather(input_tensors[i], dim, num_links, memory_config, subdevice_id, cluster_axis));
    }
    return output_tensors;
}

}  // namespace composite_common
