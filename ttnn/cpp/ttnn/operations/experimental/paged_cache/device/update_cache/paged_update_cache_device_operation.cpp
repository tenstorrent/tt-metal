// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_update_cache_device_operation.hpp"
#include "paged_update_cache_device_operation_types.hpp"
#include "paged_update_cache_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

PagedUpdateCacheDeviceOperation::program_factory_t PagedUpdateCacheDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    // Use mesh workload factory when mesh_coords is provided to enable coordinate filtering
    if (operation_attributes.mesh_coords.has_value()) {
        return PagedUpdateCacheMeshWorkloadFactory{};
    }
    return PagedUpdateCacheProgramFactory{};
}

void PagedUpdateCacheDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& cache_tensor = tensor_args.cache_tensor;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& update_idxs_tensor = tensor_args.update_idxs_tensor;
    const auto& page_table = tensor_args.page_table;

    // Device and storage validation
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE && cache_tensor.storage_type() == StorageType::DEVICE,
        "Operands to update_cache need to be on device!");
    TT_FATAL(input_tensor.device() == cache_tensor.device(), "Operands to update_cache need to be on the same device!");
    TT_FATAL(
        input_tensor.buffer() != nullptr && cache_tensor.buffer() != nullptr,
        "Operands to update_cache need to be allocated in buffers on device!");

    // Layout and data type validation
    TT_FATAL(input_tensor.layout() == Layout::TILE, "Input tensor in non-fused update_cache must be tilized");
    TT_FATAL(cache_tensor.layout() == Layout::TILE, "Cache tensor in update_cache must be tilized");
    TT_FATAL(
        cache_tensor.dtype() == DataType::FLOAT32 || cache_tensor.dtype() == DataType::BFLOAT16 ||
            cache_tensor.dtype() == DataType::BFLOAT8_B || cache_tensor.dtype() == DataType::BFLOAT4_B,
        "Data type of cache tensor must be FLOAT32, BFLOAT16, BFLOAT8_B, or BFLOAT4_B and is {}",
        cache_tensor.dtype());

    // Shape validation
    TT_FATAL(input_tensor.padded_shape()[0] == 1, "Dim 0 of input tensor must be 1");
    TT_FATAL(
        cache_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only interleaved cache is supported");

    // Overrides are paged-mode only: in non-paged mode cache.shape[2] is the
    // sequence length, and a "geometry mismatch" error there would be misleading.
    if (operation_attributes.block_size_override.has_value()) {
        TT_FATAL(
            page_table.has_value(),
            "block_size_override is only supported in paged mode (when page_table is provided)");
    }
    if (operation_attributes.num_kv_heads_override.has_value()) {
        TT_FATAL(
            page_table.has_value(),
            "num_kv_heads_override is only supported in paged mode (when page_table is provided)");
    }
    if (operation_attributes.cache_position_modulo.has_value()) {
        TT_FATAL(
            page_table.has_value(),
            "cache_position_modulo is only supported in paged mode (when page_table is provided)");
    }

    // Per-block element-count consistency. The program factory wires the kernel's
    // per-block stride from (num_heads, effective_block_size, input_head_dim) read off
    // the call view. For each new physical block in the cache the kernel jumps by that
    // stride, so it must equal the cache's actual per-block element count. Allowing
    // input_num_heads != cache_num_heads (via num_kv_heads_override) is what enables
    // HMA cross-group tensor sharing for models with asymmetric num_kv_heads per layer
    // type (e.g. Gemma4-26B-A4B sliding kv=8 / full kv=2). Trivially holds for legacy
    // callers with no override and matching shapes. Non-paged callers keep the stricter
    // legacy last-dim equality below (sharding/stride assumptions upstream).
    const uint32_t input_head_dim = input_tensor.padded_shape()[-1];
    if (page_table.has_value()) {
        const uint32_t cache_num_heads = cache_tensor.padded_shape()[1];
        const uint32_t cache_block_size = cache_tensor.padded_shape()[2];
        const uint32_t cache_head_dim = cache_tensor.padded_shape()[-1];
        const uint32_t effective_block_size = operation_attributes.block_size_override.value_or(cache_block_size);
        const uint32_t input_num_heads = operation_attributes.num_kv_heads_override.value_or(cache_num_heads);
        const uint64_t cache_elems_per_block =
            static_cast<uint64_t>(cache_num_heads) * cache_block_size * cache_head_dim;
        const uint64_t view_elems_per_block =
            static_cast<uint64_t>(input_num_heads) * effective_block_size * input_head_dim;
        TT_FATAL(
            view_elems_per_block == cache_elems_per_block,
            "paged_update_cache geometry mismatch: cache has {} elems/block "
            "(kv_heads={}, block_size={}, head_dim={}) but call view is {} "
            "(kv_heads={}, block_size={}, head_dim={}).",
            cache_elems_per_block,
            cache_num_heads,
            cache_block_size,
            cache_head_dim,
            view_elems_per_block,
            input_num_heads,
            effective_block_size,
            input_head_dim);
        TT_FATAL(
            effective_block_size % tt::constants::TILE_HEIGHT == 0,
            "effective block_size ({}) must be a multiple of TILE_HEIGHT ({})",
            effective_block_size,
            tt::constants::TILE_HEIGHT);
        if (operation_attributes.cache_position_modulo.has_value()) {
            const uint32_t cache_position_modulo = operation_attributes.cache_position_modulo.value();
            TT_FATAL(
                cache_position_modulo > 0 && cache_position_modulo % effective_block_size == 0,
                "cache_position_modulo ({}) must be a positive multiple of effective block_size ({}); "
                "otherwise a wrapped position would split across blocks and the kernel can't address it.",
                cache_position_modulo,
                effective_block_size);
        }
    } else {
        TT_FATAL(
            input_head_dim == cache_tensor.padded_shape()[-1],
            "Last dim of input tensor ({}) must match last dim of cache tensor ({}) in non-paged mode",
            input_head_dim,
            cache_tensor.padded_shape()[-1]);
    }
    // Wt is derived from input.padded_shape[-1], so the last dim must be tile-aligned
    // regardless of cache shape.
    TT_FATAL(
        input_head_dim % tt::constants::TILE_WIDTH == 0,
        "input_tensor last dim ({}) must be a multiple of TILE_WIDTH ({})",
        input_head_dim,
        tt::constants::TILE_WIDTH);

    // Paged cache validation
    const bool paged_cache = page_table.has_value();
    if (!paged_cache) {
        if (operation_attributes.share_cache) {
            TT_FATAL(
                cache_tensor.padded_shape()[0] == 1, "Share cache feature expects cache tensor to have batch of 1");
        } else {
            TT_FATAL(
                input_tensor.padded_shape()[1] == cache_tensor.padded_shape()[0],
                "Expect batch in input tensor match the batch in cache tensor");
        }
    } else {
        TT_FATAL(!operation_attributes.share_cache, "share_cache not supported with paged cache");
        TT_FATAL(update_idxs_tensor.has_value(), "Paged cache requires update_idxs tensor");

        auto page_table_val = page_table.value();

        if (page_table_val.is_sharded()) {
            TT_FATAL(page_table_val.dtype() == DataType::UINT16, "Expect page table to have datatype UINT16");
        } else {
            TT_FATAL(page_table_val.dtype() == DataType::INT32, "Expect page table to have datatype INT32");
        }

        TT_FATAL(page_table_val.layout() == Layout::ROW_MAJOR, "Expect page table to have memory layout in ROW MAJOR");

        if (page_table_val.is_sharded()) {
            uint32_t num_cores = page_table_val.memory_config().shard_spec()->grid.num_cores();
            uint32_t page_table_shard_height = page_table_val.padded_shape()[0] / num_cores;
            TT_FATAL(
                page_table_shard_height == input_tensor.padded_shape()[1],
                "Batch size in input tensor {} should match page table shard height {}",
                input_tensor.padded_shape()[1],
                page_table_shard_height);
        } else {
            TT_FATAL(
                page_table_val.padded_shape()[0] == input_tensor.padded_shape()[1],
                "Batch size between page_table and input_tensor must match");
        }
        if (operation_attributes.cache_position_modulo.has_value()) {
            // Bounded-pool path (vLLM SlidingWindowSpec): page_table is right-padded
            // out to max_model_len/block_size, but the kernel wraps positions into
            // [0, modulo) before the page_table lookup, so the tail is never read.
            // Only the bounded prefix needs to fit in the physical pool.
            const uint32_t effective_block_size =
                operation_attributes.block_size_override.value_or(cache_tensor.padded_shape()[2]);
            const uint32_t modulo = operation_attributes.cache_position_modulo.value();
            const uint32_t bounded_blocks_per_seq = modulo / effective_block_size;
            TT_FATAL(
                bounded_blocks_per_seq <= cache_tensor.padded_shape()[0],
                "cache_position_modulo/block_size ({}) must be <= max_num_blocks ({})",
                bounded_blocks_per_seq,
                cache_tensor.padded_shape()[0]);
        } else {
            TT_FATAL(
                page_table_val.padded_shape()[1] <= cache_tensor.padded_shape()[0],
                "max_num_blocks_per_seq must be less than max_num_blocks: max_num_blocks_per_seq={}, "
                "max_num_blocks={}",
                page_table_val.padded_shape()[1],
                cache_tensor.padded_shape()[0]);
        }
    }

    // Update indices validation
    TT_FATAL(
        (update_idxs_tensor.has_value()) != (!operation_attributes.update_idxs.empty()),
        "Only an update tensor or an update vector can be provided. Not both or neither.");

    uint32_t num_indices = 0;
    uint32_t num_cores_cur_pos = 0;
    if (update_idxs_tensor.has_value()) {
        const auto& update_idxs_tensor_val = update_idxs_tensor.value();
        TT_FATAL(update_idxs_tensor_val.dtype() == DataType::INT32, "Expected update_idxs to have datatype INT32");
        TT_FATAL(
            update_idxs_tensor_val.layout() == Layout::ROW_MAJOR,
            "Expected update_idxs to have memory layout in ROW MAJOR");

        if (update_idxs_tensor_val.is_sharded()) {
            TT_FATAL(
                update_idxs_tensor_val.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
                "Expect update_idxs to be HEIGHT SHARDED if sharded");
            TT_FATAL(
                update_idxs_tensor_val.buffer()->buffer_type() == tt::tt_metal::BufferType::L1,
                "Expect update_idxs to have buffer type L1 if sharded");
            num_cores_cur_pos = update_idxs_tensor_val.padded_shape()[0];
            num_indices = update_idxs_tensor_val.logical_shape()[1];
        } else {
            TT_FATAL(
                update_idxs_tensor_val.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
                "Expect update_idxs to be DRAM INTERLEAVED");
            TT_FATAL(
                update_idxs_tensor_val.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
                "Expect update_idxs to have buffer type DRAM");
            num_cores_cur_pos = 0;
            num_indices = update_idxs_tensor_val.padded_shape()[0];
        }
    } else {
        num_indices = operation_attributes.update_idxs.size();
    }
    if (update_idxs_tensor.has_value() && update_idxs_tensor.value().is_sharded()) {
        uint32_t in_num_cores_cur_pos = update_idxs_tensor.value().shard_spec().value().grid.num_cores();
        TT_FATAL(
            input_tensor.logical_shape()[1] == num_indices,
            "Number of update_idxs ({}) should match batch size ({}) if sharded",
            num_indices,
            input_tensor.logical_shape()[1]);
        TT_FATAL(
            in_num_cores_cur_pos == num_cores_cur_pos,
            "Number of cores sharded on L1 ({}) should match dimension of update_idxs at 0 ({})",
            in_num_cores_cur_pos,
            num_cores_cur_pos);
    } else {
        TT_FATAL(
            input_tensor.padded_shape()[1] == num_indices,
            "Number of update_idxs ({}) should match batch size ({})",
            num_indices,
            input_tensor.padded_shape()[1]);
    }

    // Sharding validation
    TT_FATAL(input_tensor.is_sharded(), "Expect input_tensor to be sharded");
    if (input_tensor.is_sharded()) {
        TT_FATAL(
            input_tensor.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
            "Expect input_tensor to NOT have memory layout WIDTH SHARDED");
        TT_FATAL(
            input_tensor.shard_spec().value().shape[1] == input_tensor.padded_shape()[-1],
            "Expect input_tensor to have shard width ({}) equal to the last dimension of the input tensor padded "
            "shape ({})",
            input_tensor.shard_spec().value().shape[1],
            input_tensor.padded_shape()[-1]);
        TT_FATAL(
            (input_tensor.physical_volume() / input_tensor.padded_shape()[-1]) %
                    input_tensor.shard_spec().value().shape[0] ==
                0,
            "Input tensor's height must be divisible by the number of shards along the height dimension. Got "
            "height = {}, number of shards = {}.",
            (input_tensor.physical_volume() / input_tensor.padded_shape()[-1]),
            input_tensor.shard_spec().value().shape[0]);
        TT_FATAL(
            input_tensor.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
            "Only ROW_MAJOR sharding is supported");
        // The program factory dispatches one user per core (cores[i] is assigned update_idxs[i]
        // and reads a shard whose height is exactly one user's worth of kv-heads). A grid with
        // num_cores != num_users splits or duplicates a user's data across cores and silently
        // corrupts the cache (issue #44923). Canonical layout is a {num_users, 1} grid, but any
        // grid with num_cores == num_users works.
        const uint32_t num_users = input_tensor.padded_shape()[1];
        const uint32_t input_num_shards = input_tensor.shard_spec().value().grid.num_cores();
        TT_FATAL(
            input_num_shards == num_users,
            "Input tensor shard grid must have num_cores ({}) equal to the number of users / "
            "batch ({}). The kernel dispatches one user per core; use a {{num_users, 1}} grid "
            "(or any CoreRangeSet whose num_cores == num_users).",
            input_num_shards,
            num_users);
    }

    // Data type validation
    TT_FATAL(
        input_tensor.dtype() == DataType::FLOAT32 || input_tensor.dtype() == DataType::BFLOAT16,
        "Data type of input tensor for update cache must be FLOAT32 or BFLOAT16");

    TT_FATAL(operation_attributes.batch_offset == 0, "batch_offset must be 0");
}

PagedUpdateCacheDeviceOperation::spec_return_value_t PagedUpdateCacheDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // Do nothing because it's an in-place operation
    return tensor_args.cache_tensor.tensor_spec();
}

PagedUpdateCacheDeviceOperation::tensor_return_value_t PagedUpdateCacheDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // In-place operation, return the cache tensor
    return tensor_args.cache_tensor;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::PagedUpdateCacheDeviceOperation::tensor_return_value_t paged_update_cache(
    const Tensor& cache_tensor,
    const Tensor& input_tensor,
    const std::vector<uint32_t>& update_idxs,
    const std::optional<const Tensor>& update_idxs_tensor,
    std::optional<bool> share_cache,
    const std::optional<const Tensor>& page_table,
    uint32_t batch_offset,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const std::set<ttnn::MeshCoordinate>>& mesh_coords,
    std::optional<uint32_t> block_size_override,
    std::optional<uint32_t> num_kv_heads_override,
    std::optional<uint32_t> cache_position_modulo) {
    using OperationType = ttnn::experimental::prim::PagedUpdateCacheDeviceOperation;

    auto kernel_config_val = init_device_compute_kernel_config(input_tensor.device()->arch(), compute_kernel_config);
    const bool share_cache_arg = share_cache.has_value() ? share_cache.value() : false;

    auto operation_attributes = OperationType::operation_attributes_t{
        .update_idxs = update_idxs,
        .batch_offset = batch_offset,
        .compute_kernel_config = kernel_config_val,
        .share_cache = share_cache_arg,
        .mesh_coords = mesh_coords,
        .block_size_override = block_size_override,
        .num_kv_heads_override = num_kv_heads_override,
        .cache_position_modulo = cache_position_modulo};

    auto tensor_args = OperationType::tensor_args_t{
        .cache_tensor = cache_tensor,
        .input_tensor = input_tensor,
        .update_idxs_tensor =
            update_idxs_tensor.has_value() ? std::optional<Tensor>(update_idxs_tensor.value()) : std::nullopt,
        .page_table = page_table.has_value() ? std::optional<Tensor>(page_table.value()) : std::nullopt};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
