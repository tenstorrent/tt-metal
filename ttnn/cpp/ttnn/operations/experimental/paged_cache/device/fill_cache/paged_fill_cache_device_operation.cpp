// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_fill_cache_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

PagedFillCacheDeviceOperation::program_factory_t PagedFillCacheDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& /*tensor_args*/) {
    // Use mesh workload factory when mesh_coords is provided to enable coordinate filtering
    if (args.mesh_coords.has_value()) {
        return PagedFillCacheMeshWorkloadFactory{};
    }
    return PagedFillCacheProgramFactory{};
}

void PagedFillCacheDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& cache_tensor = tensor_args.cache_tensor;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& page_table_tensor = tensor_args.page_table;

    // Data type validation
    TT_FATAL(
        input_tensor.dtype() == DataType::FLOAT32 || input_tensor.dtype() == DataType::BFLOAT16 ||
            cache_tensor.dtype() == DataType::BFLOAT8_B || cache_tensor.dtype() == DataType::BFLOAT4_B,
        "Data type of input tensor for fill cache must be FLOAT32, BFLOAT16, or BFLOAT8_b");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Expect input_tensor to have memory layout INTERLEAVED");
    TT_FATAL(
        page_table_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Expect page_table_tensor to have memory layout INTERLEAVED");
    TT_FATAL(page_table_tensor.dtype() == DataType::INT32, "Expect page_table_tensor to have datatype INT32");

    auto cache_shape = cache_tensor.padded_shape();
    auto input_shape = input_tensor.padded_shape();
    auto page_table_shape = page_table_tensor.padded_shape();

    // batch_idx indexes the page table, not the cache's block dimension.
    TT_FATAL(
        args.batch_idx_fallback < page_table_shape[0],
        "Batch idx must be within the page_table batch size");

    // Per-block element-count consistency. The program factory reads num_heads,
    // block_size, and head_dim from the *input* tensor and computes the kernel's
    // per-block stride as input_num_heads * effective_block_size * input_head_dim.
    // For each new physical block in the cache buffer the kernel jumps by that
    // stride, so it must equal the cache's actual per-block element count.
    // Allowing input_num_heads != cache_num_heads (as long as the per-block
    // element count is preserved) enables HMA cross-group tensor sharing for
    // models with asymmetric num_kv_heads per layer type (e.g. Gemma4 26B-A4B
    // with sliding kv=8 and full kv=2). Trivially holds for legacy callers
    // with no override and matching shapes.
    const uint32_t cache_num_heads = cache_shape[1];
    const uint32_t input_num_heads = input_shape[1];
    const uint32_t cache_block_size = cache_shape[2];
    const uint32_t cache_head_dim = cache_shape[3];
    const uint32_t input_head_dim = input_shape[3];
    const uint32_t effective_block_size = args.block_size_override.value_or(cache_block_size);
    const uint64_t cache_elems_per_block = static_cast<uint64_t>(cache_num_heads) * cache_block_size * cache_head_dim;
    const uint64_t view_elems_per_block =
        static_cast<uint64_t>(input_num_heads) * effective_block_size * input_head_dim;
    TT_FATAL(
        view_elems_per_block == cache_elems_per_block,
        "paged_fill_cache geometry mismatch: cache has {} elems/block "
        "(kv_heads={}, block_size={}, head_dim={}) but input view is {} "
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
        input_head_dim % tt::constants::TILE_WIDTH == 0,
        "input_tensor last dim ({}) must be a multiple of TILE_WIDTH ({})",
        input_head_dim,
        tt::constants::TILE_WIDTH);
    TT_FATAL(
        effective_block_size % tt::constants::TILE_HEIGHT == 0,
        "effective block_size ({}) must be a multiple of TILE_HEIGHT ({})",
        effective_block_size,
        tt::constants::TILE_HEIGHT);

    if (args.cache_position_modulo.has_value()) {
        const uint32_t modulo = args.cache_position_modulo.value();
        TT_FATAL(modulo > 0, "cache_position_modulo must be > 0 when provided");
        TT_FATAL(
            modulo % effective_block_size == 0,
            "cache_position_modulo ({}) must be a positive multiple of effective block_size ({}); "
            "otherwise a wrapped position would split across blocks and the kernel can't address it.",
            modulo,
            effective_block_size);
        TT_FATAL(
            modulo <= effective_block_size * page_table_shape[1],
            "cache_position_modulo ({}) must fit in max_num_blocks_per_seq ({}) * block_size ({})",
            modulo,
            page_table_shape[1],
            effective_block_size);
    } else {
        // Legacy path: input must fit in the page_table address space directly.
        TT_FATAL(
            input_shape[2] <= effective_block_size * page_table_shape[1],
            "Input seq_len ({}) must fit in max_num_blocks_per_seq ({}) * block_size ({})",
            input_shape[2],
            page_table_shape[1],
            effective_block_size);
    }

    if (tensor_args.batch_idx_tensor_opt.has_value()) {
        const auto& tensor = tensor_args.batch_idx_tensor_opt.value();
        const auto input_batch = input_shape[0];
        TT_FATAL(
            tensor.physical_volume() == input_batch,
            "Batch idx tensor must have input_tensor batch dim ({}) elements, got {}",
            input_batch,
            tensor.physical_volume());
        TT_FATAL(
            tensor.dtype() == DataType::UINT32 || tensor.dtype() == DataType::INT32,
            "Batch idx tensor must be an integer type");
        // The writer kernel reads the tensor as a single contiguous noc page
        // via TensorAccessor::get_noc_addr(0). That only resolves correctly for
        // a ROW_MAJOR, INTERLEAVED, DRAM-resident buffer; sharded or L1-resident
        // tensors would have batch_idx values scattered across NoC locations
        // the single read won't cover.
        TT_FATAL(tensor.layout() == Layout::ROW_MAJOR, "Batch idx tensor must be in ROW_MAJOR layout");
        TT_FATAL(
            tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Batch idx tensor must have INTERLEAVED memory layout");
        TT_FATAL(
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Batch idx tensor must be DRAM-resident");
    }

    // valid_seq_len tensor: a single int (block-aligned token count). Read by the
    // writer via TensorAccessor::get_noc_addr(0), so it must be a ROW_MAJOR,
    // INTERLEAVED, DRAM-resident int tensor (same constraints as batch_idx_tensor).
    if (tensor_args.valid_seq_len_tensor_opt.has_value()) {
        const auto& tensor = tensor_args.valid_seq_len_tensor_opt.value();
        TT_FATAL(
            tensor.dtype() == DataType::UINT32 || tensor.dtype() == DataType::INT32,
            "valid_seq_len tensor must be an integer type");
        TT_FATAL(
            tensor.logical_volume() == 1,
            "valid_seq_len tensor must contain exactly 1 element, got logical_volume={}",
            tensor.logical_volume());
        TT_FATAL(tensor.layout() == Layout::ROW_MAJOR, "valid_seq_len tensor must be in ROW_MAJOR layout");
        TT_FATAL(
            tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "valid_seq_len tensor must have INTERLEAVED memory layout");
        TT_FATAL(
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "valid_seq_len tensor must be DRAM-resident");
        // Only meaningful for a bounded (circular) fill; a no-op otherwise.
        TT_FATAL(
            args.cache_position_modulo.has_value(),
            "valid_seq_len tensor is only supported when cache_position_modulo is set (bounded fill)");
    }

    // When mesh_coords is provided, validate it is a subset of the tensor's mesh
    // coordinates. The descriptor framework dispatches a (noop) program for every
    // tensor_coord regardless of mesh_coords membership, so a stray coord in mesh_coords
    // that doesn't exist in tensor_coords would silently never run — matches the legacy
    // create_mesh_workload's TT_FATAL check.
    if (args.mesh_coords.has_value()) {
        const auto tensor_coords_vec =
            ttnn::device_operation::mesh_device_operation_utils::extract_tensor_coordinates(tensor_args);
        const std::set<ttnn::MeshCoordinate> tensor_coords_set(tensor_coords_vec.begin(), tensor_coords_vec.end());
        for (const auto& mesh_coord : args.mesh_coords.value()) {
            TT_FATAL(
                tensor_coords_set.contains(mesh_coord),
                "Mesh coordinate ({}, {}) is in mesh_coords but not found in tensor_coords. "
                "mesh_coords size: {}, tensor_coords size: {}",
                mesh_coord[0],
                mesh_coord[1],
                args.mesh_coords->size(),
                tensor_coords_set.size());
        }
    }
}

TensorSpec PagedFillCacheDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // In-place operation, return cache tensor's spec
    return tensor_args.cache_tensor.tensor_spec();
}

Tensor PagedFillCacheDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // In-place operation, return the cache tensor
    return tensor_args.cache_tensor;
}

ttsl::hash::hash_t PagedFillCacheDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto program_factory = select_program_factory(args, tensor_args);

    // Exclude batch_idx_fallback and noop (runtime-only).
    // Include mesh_coords (affects program factory selection).
    // Include block_size_override and cache_position_modulo (enter compile-time args).
    return operation::hash_operation<PagedFillCacheDeviceOperation>(
        args.mesh_coords, args.block_size_override, args.cache_position_modulo, tensor_args, program_factory.index());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor paged_fill_cache(
    const Tensor& cache_tensor,
    const Tensor& input_tensor,
    const Tensor& page_table,
    const std::optional<Tensor>& batch_idx_tensor,
    uint32_t batch_idx_fallback,
    const std::optional<std::set<ttnn::MeshCoordinate>>& mesh_coords,
    std::optional<uint32_t> block_size_override,
    std::optional<uint32_t> cache_position_modulo,
    const std::optional<Tensor>& valid_seq_len_tensor) {
    using OperationType = ttnn::experimental::prim::PagedFillCacheDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .batch_idx_fallback = batch_idx_fallback,
        .mesh_coords = mesh_coords,
        .noop = false,
        .block_size_override = block_size_override,
        .cache_position_modulo = cache_position_modulo,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .cache_tensor = cache_tensor,
        .input_tensor = input_tensor,
        .page_table = page_table,
        .batch_idx_tensor_opt = batch_idx_tensor,
        .valid_seq_len_tensor_opt = valid_seq_len_tensor,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
