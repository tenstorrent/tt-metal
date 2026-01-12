// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_device_operation.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/split/split.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"

namespace composite_common {

// Map a dimension of an ND tensor to 4D. If dim > than rank difference, subtract rank difference.
std::tuple<uint32_t, int32_t> normalize_dim_4d(const uint32_t dim, const uint32_t rank) {
    constexpr int32_t RANK_4D = 4, RANK_2D = 2;

    // special case for rank 2
    if (rank == RANK_2D) {
        return std::make_tuple(RANK_2D + dim, RANK_2D);
    }

    const auto rank_diff = static_cast<int32_t>(rank) - RANK_4D;
    const auto normalized_dim = (dim < std::abs(rank_diff)) ? dim : dim - rank_diff;

    return std::make_tuple(normalized_dim, rank_diff);
}

bool use_composite_reduce_scatter(
    const ttnn::Tensor& input_tensor, const int32_t dim, std::optional<uint32_t> cluster_axis) {
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    int32_t rank = input_tensor.logical_shape().rank();
    int32_t scatter_dim = (dim < 0) ? rank + dim : dim;

    const auto normalized_scatter_dim = std::get<0>(normalize_dim_4d(scatter_dim, rank));

    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);

    // Must scatter evenly
    auto input_shape = input_tensor.logical_shape();
    if (input_shape[scatter_dim] % num_devices != 0) {
        return false;
    }

    // Use composite for row major tensors
    if (input_tensor.layout() == ttnn::Layout::ROW_MAJOR) {
        return true;
    }

    // Use composite if tiled and scattering on padded dim 2 or 3
    auto output_shape = input_shape;
    output_shape[scatter_dim] /= num_devices;
    return (normalized_scatter_dim == 3 && output_shape[scatter_dim] % tile_width != 0) ||
           (normalized_scatter_dim == 2 && output_shape[scatter_dim] % tile_height != 0);
}

ttnn::Tensor composite_reduce_scatter(
    ttnn::Tensor input_tensor,
    const int32_t dim,
    const uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel) {
    bool is_row_major = input_tensor.layout() == ttnn::Layout::ROW_MAJOR;

    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);

    int32_t rank = input_tensor.logical_shape().rank();
    int32_t scatter_dim = (dim < 0) ? rank + dim : dim;

    auto output_shape = input_tensor.logical_shape();
    output_shape[scatter_dim] /= num_devices;

    TT_FATAL(
        !(input_tensor.memory_config().is_sharded() && !memory_config.has_value()),
        "If input memory config is sharded, then output memory config must be provided. Defaulting the output memory "
        "config to the input sharded memory config will break the op as the input and output shapes are different.");
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());

    /*
     * - If sharded to interleaved, convert to the final interleaved memory config, and use that final
     *   interleaved memory config for all ops within the composite.
     * - If interleaved to sharded, use the input memory config for all ops within the composite, and
     *   then we'll convert to the output sharded memory config at the end.
     * - If sharded to sharded, use DRAM interleaved as the intermediate memory config for all ops
     *   within the composite, and then we'll convert to the output sharded memory config at the end.
     * - If interleaved DRAM to interleaved L1 or interleaved L1 to interleaved DRAM, do the conversion
     *   during the execution of native RS.
     * - If interleaved DRAM to interleaved DRAM or interleaved L1 to interleaved L1, use the single
     *   memory config for all ops within the composite.
     */
    ttnn::MemoryConfig native_rs_output_memory_config;
    if (input_tensor.memory_config().is_sharded()) {
        native_rs_output_memory_config =
            output_memory_config.is_sharded() ? ttnn::DRAM_MEMORY_CONFIG : output_memory_config;
        input_tensor = ttnn::to_memory_config(input_tensor, native_rs_output_memory_config);
    } else if (!output_memory_config.is_sharded() && input_tensor.memory_config() != output_memory_config) {
        native_rs_output_memory_config = output_memory_config;
    } else {
        native_rs_output_memory_config = input_tensor.memory_config();
    }

    // split the input tensor so we can insert internal padding
    std::vector<ttnn::Tensor> split_tensors =
        ttnn::split(input_tensor, output_shape[scatter_dim], scatter_dim, input_tensor.memory_config());
    if (is_row_major) {
        for (uint32_t i = 0; i < num_devices; ++i) {
            split_tensors[i] =
                ttnn::tilize_with_zero_padding(split_tensors[i], split_tensors[i].memory_config(), std::nullopt, true);
        }
    }

    // insert the internal padding (only pad on the dim we're scattering on)
    auto logical_shape = split_tensors[0].logical_shape();
    auto padded_shape = split_tensors[0].padded_shape();
    ttnn::SmallVector<std::array<uint32_t, 2>> padding = {
        {0, padded_shape[-2] - logical_shape[-2]}, {0, padded_shape[-1] - logical_shape[-1]}};
    for (uint32_t i = 0; i < num_devices; ++i) {
        split_tensors[i] = ttnn::pad(split_tensors[i], padding, 0, true, split_tensors[i].memory_config());
    }

    // concat back into a single input tensor, now with internal padding
    ttnn::Tensor padded_native_rs_input_tensor = ttnn::concat(split_tensors, scatter_dim);
    tt::tt_fabric::Topology topology_ =
        ::ttnn::ccl::get_usable_topology(padded_native_rs_input_tensor, topology, cluster_axis);
    topology_ = ::ttnn::ccl::convert_2d_to_1d_topology(topology_);
    // execute native RS
    ttnn::Tensor padded_native_rs_output_tensor = ttnn::prim::reduce_scatter(
                                                      padded_native_rs_input_tensor,
                                                      scatter_dim,
                                                      cluster_axis,
                                                      subdevice_id,
                                                      native_rs_output_memory_config,
                                                      std::nullopt,  // optional intermediate memory config
                                                      std::nullopt,  // optional output tensor
                                                      num_links,
                                                      topology_,
                                                      chunks_per_sync,
                                                      num_workers_per_link,
                                                      num_buffers_per_channel)
                                                      .at(1);  // first is the intermediate tensor
    // remove the padding we previously inserted
    ttnn::Tensor rs_output_tensor;
    if (is_row_major) {
        tt::tt_metal::Shape ends(ttsl::SmallVector<uint32_t>(output_shape.rank(), 0));
        for (uint32_t i = 0; i < output_shape.rank(); ++i) {
            ends[i] = output_shape[i] - 1;
        }

        rs_output_tensor =
            ttnn::untilize_with_unpadding(padded_native_rs_output_tensor, ends, native_rs_output_memory_config);
    } else {
        const ttnn::SmallVector<int32_t> steps(output_shape.rank(), 1);
        ttnn::SmallVector<int32_t> begins(output_shape.rank(), 0), ends(output_shape.cbegin(), output_shape.cend());
        const tt::stl::Span<const int32_t> sbegins(begins), ssteps(steps), sends(ends);
        rs_output_tensor =
            ttnn::slice(padded_native_rs_output_tensor, sbegins, sends, ssteps, native_rs_output_memory_config);
    }

    // if the output is sharded, do the conversion
    if (output_memory_config.is_sharded()) {
        rs_output_tensor = ttnn::to_memory_config(rs_output_tensor, output_memory_config);
    }
    return rs_output_tensor;
}

bool use_all_gather_async_llama_sharded(const ttnn::Tensor& input_tensor, const ttnn::MemoryConfig& output_mem_config) {
    auto input_tensor_shape = input_tensor.padded_shape();
    auto input_tensor_memory_config = input_tensor.memory_config();
    bool input_is_sharded = input_tensor_memory_config.shard_spec().has_value();
    bool output_is_sharded = output_mem_config.shard_spec().has_value();
    bool input_is_tile = input_tensor.layout() == ttnn::Layout::TILE;

    log_trace(tt::LogOp, "[select_version] input_tensor_shape: {}", input_tensor_shape);
    log_trace(tt::LogOp, "[select_version] input_tensor_memory_config: {}", input_tensor_memory_config);
    log_trace(tt::LogOp, "[select_version] output_mem_config: {}", output_mem_config);

    log_trace(tt::LogOp, "[select_version] input_is_sharded: {}", input_is_sharded);
    log_trace(tt::LogOp, "[select_version] output_is_sharded: {}", output_is_sharded);

    // Check for minimal sharded case
    if (input_is_sharded && output_is_sharded && input_is_tile) {
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
    auto is_true_2d_mesh = [](const ttnn::Tensor& t) {
        const auto mesh_shape = t.device()->shape();
        return mesh_shape.dims() >= 2 && mesh_shape[0] > 1 && mesh_shape[1] > 1;
    };

    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    auto input_shape = input_tensor.logical_shape();

    int32_t rank = input_tensor.logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    auto input_memory_config = input_tensor.memory_config();
    auto output_memory_config = memory_config.value_or(input_memory_config);

    if (tt::tt_fabric::GetFabricConfig() == tt::tt_fabric::FabricConfig::FABRIC_2D && is_true_2d_mesh(input_tensor)) {
        return true;
    }
    // Use composite for row-major tensors
    if (input_tensor.layout() == ttnn::Layout::ROW_MAJOR) {
        return true;
    }

    // Use composite if tiled and padded on the gather dim
    bool is_tiled_and_padded_on_gather_dim = input_tensor.layout() == ttnn::Layout::TILE &&
                                             ((gather_dim == rank - 2 && input_shape[-2] % tile_height != 0) ||
                                              (gather_dim == rank - 1 && input_shape[-1] % tile_width != 0));
    return is_tiled_and_padded_on_gather_dim;
}

bool use_composite_all_to_all(
    const ttnn::Tensor& input_tensor,
    int32_t in_dim,
    int32_t out_dim,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    auto input_shape = input_tensor.logical_shape();
    uint32_t num_devices = ttnn::ccl::get_topological_dimension(input_tensor, std::nullopt);

    int32_t rank = input_tensor.logical_shape().rank();
    in_dim = (in_dim < 0) ? rank + in_dim : in_dim;
    out_dim = (out_dim < 0) ? rank + out_dim : out_dim;

    auto last_dim = rank - 1;
    auto second_last_dim = rank - 2;

    auto output_out_dim = input_shape[out_dim] / num_devices;
    bool is_tiled_and_tile_aligned = input_tensor.layout() == ttnn::Layout::TILE &&
                                     (in_dim != second_last_dim || input_shape[in_dim] % (tile_height / 2) == 0) &&
                                     (in_dim != last_dim || input_shape[in_dim] % tile_width == 0) &&
                                     (out_dim != second_last_dim || output_out_dim % (tile_height / 2) == 0) &&
                                     (out_dim != last_dim || output_out_dim % tile_width == 0);

    // the current native implementation works for very specific cases
    bool use_native =
        (input_tensor.layout() == ttnn::Layout::TILE &&
         input_tensor.buffer()->buffer_type() == ttnn::BufferType::DRAM &&
         input_tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED &&
         (!memory_config.has_value() ||
          memory_config.value().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED) &&
         is_tiled_and_tile_aligned);

    return !use_native;
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

    // If we need to convert to row-major, then if the input dtype is bfloat8_b we need to typecast before untilizing
    // and after re-tilizing
    ttnn::DataType input_dtype = input_tensor.dtype();
    bool is_tiled_and_not_tile_aligned = input_tensor.layout() == ttnn::Layout::TILE &&
                                         (input_shape[-2] % tile_height != 0 || input_shape[-1] % tile_width != 0);
    bool convert_to_bfloat16_for_composite = is_tiled_and_not_tile_aligned && input_dtype == ttnn::DataType::BFLOAT8_B;

    auto input_memory_config = input_tensor.memory_config();
    TT_FATAL(
        !(input_memory_config.is_sharded() && !memory_config.has_value()),
        "If input memory config is sharded, then output memory config must be provided. Defaulting the output memory "
        "config to the input sharded memory config will break the op as the input and output shapes are different.");
    auto output_memory_config = memory_config.value_or(input_memory_config);

    if (input_memory_config.is_sharded()) {
        /*
         * If sharded to interleaved, convert to the final interleaved memory config.
         * If sharded to sharded, use DRAM interleaved as the intermediate memory
         * config for executing the composite.
         */
        auto intermediate_memory_config =
            output_memory_config.is_sharded() ? ttnn::DRAM_MEMORY_CONFIG : output_memory_config;
        input_tensor = ttnn::to_memory_config(input_tensor, intermediate_memory_config);
    }

    if (convert_to_bfloat16_for_composite) {
        input_tensor = ttnn::typecast(input_tensor, ttnn::DataType::BFLOAT16);
    }

    std::vector<ttnn::Tensor> broadcasted_tensors = ttnn::prim::all_broadcast(
        input_tensor, cluster_axis, subdevice_id, input_tensor.memory_config(), num_links, ttnn::ccl::Topology::Linear);

    // Do the gather itself
    ttnn::Tensor all_gather_output_tensor = ttnn::concat(broadcasted_tensors, gather_dim);

    if (convert_to_bfloat16_for_composite) {
        all_gather_output_tensor = ttnn::typecast(all_gather_output_tensor, input_dtype);
    }

    if (output_memory_config.is_sharded()) {
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
    for (const auto& input_tensor : input_tensors) {
        output_tensors.push_back(
            composite_all_gather(input_tensor, dim, num_links, memory_config, subdevice_id, cluster_axis));
    }
    return output_tensors;
}

ttnn::Tensor composite_all_to_all(
    ttnn::Tensor input_tensor,
    int32_t in_dim,
    int32_t out_dim,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    auto input_shape = input_tensor.logical_shape();

    int32_t rank = input_tensor.logical_shape().rank();
    in_dim = (in_dim < 0) ? rank + in_dim : in_dim;
    out_dim = (out_dim < 0) ? rank + out_dim : out_dim;

    bool is_tiled_and_not_tile_aligned = input_tensor.layout() == ttnn::Layout::TILE &&
                                         (input_shape[2] % tile_height != 0 || input_shape[3] % tile_width != 0);

    // If we need to convert to row-major, then if the input dtype is bfloat8_b we need to typecast before untilizing
    // and after re-tilizing
    ttnn::DataType input_dtype = input_tensor.dtype();
    bool convert_to_bfloat16_for_composite = is_tiled_and_not_tile_aligned && input_dtype == ttnn::DataType::BFLOAT8_B;

    bool is_sharded = input_tensor.is_sharded();
    auto input_memory_config = input_tensor.memory_config();
    auto interim_memory_config = is_sharded ? ttnn::DRAM_MEMORY_CONFIG : input_memory_config;
    auto output_memory_config = memory_config.value_or(input_memory_config);

    ttnn::Tensor temp_tensor;

    // Convert to row major
    if (is_tiled_and_not_tile_aligned) {
        // If input is tiled bfloat8_b, convert to bfloat16 to do the all_broadcast + concat
        if (convert_to_bfloat16_for_composite) {
            temp_tensor = ttnn::typecast(input_tensor, ttnn::DataType::BFLOAT16);
            input_tensor.deallocate();
            input_tensor = temp_tensor;
        }
        temp_tensor = ttnn::to_layout(input_tensor, ttnn::Layout::ROW_MAJOR);
        input_tensor.deallocate();
        input_tensor = temp_tensor;
    }

    // Sharded input is challenging to work with, because we perform slice and concat separately
    // and the user can't give us intermediate shard specs. So the most foolproof solution is
    // to simply undo sharding by converting to DRAM interleaved storage.
    if (is_sharded) {
        temp_tensor = ttnn::to_memory_config(input_tensor, interim_memory_config);
        input_tensor.deallocate();
        input_tensor = temp_tensor;
    }

    // Step 1: make every device have a copy of every tensor
    std::vector<ttnn::Tensor> broadcasted_tensors = ttnn::prim::all_broadcast(
        input_tensor,
        /* cluster_axis */ std::nullopt,
        subdevice_id,
        interim_memory_config,
        num_links,
        ttnn::ccl::Topology::Linear);
    input_tensor.deallocate();

    // Step 2: Slice out the index range each device cares about, along out_dim
    for (auto& broadcasted_tensor : broadcasted_tensors) {
        temp_tensor =
            ttnn::mesh_partition(broadcasted_tensor, out_dim, /* cluster_axis */ std::nullopt, interim_memory_config);
        broadcasted_tensor.deallocate();
        broadcasted_tensor = temp_tensor;
    }

    // Step 3: Concatenate along in_dim
    ttnn::Tensor output_tensor = ttnn::concat(broadcasted_tensors, in_dim, interim_memory_config);
    for (auto& tensor : broadcasted_tensors) {
        tensor.deallocate();
    }

    // Convert back to tiled
    if (is_tiled_and_not_tile_aligned) {
        temp_tensor = ttnn::to_layout(output_tensor, ttnn::Layout::TILE);
        output_tensor.deallocate();
        output_tensor = temp_tensor;
        // If we had to convert the input dtype in order to execute the row-major composite op, convert back to the
        // input dtype
        if (convert_to_bfloat16_for_composite) {
            temp_tensor = ttnn::typecast(output_tensor, input_dtype);
            output_tensor.deallocate();
            output_tensor = temp_tensor;
        }
    }

    if (output_memory_config.memory_layout() != interim_memory_config.memory_layout()) {
        temp_tensor = ttnn::to_memory_config(output_tensor, output_memory_config);
        output_tensor.deallocate();
        output_tensor = temp_tensor;
    }

    return output_tensor;
}

}  // namespace composite_common
