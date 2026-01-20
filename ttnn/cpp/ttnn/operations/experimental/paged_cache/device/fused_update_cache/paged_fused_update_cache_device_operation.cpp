// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_fused_update_cache_device_operation.hpp"
#include "paged_fused_update_cache_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

PagedFusedUpdateCacheDeviceOperation::program_factory_t PagedFusedUpdateCacheDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor1 = tensor_args.input_tensor1;
    const auto& input_tensor2 = tensor_args.input_tensor2;
    const bool use_mesh_workload_factory = operation_attributes.mesh_coords.has_value();

    if (input_tensor1.layout() == Layout::TILE && input_tensor2.layout() == Layout::TILE) {
        if (use_mesh_workload_factory) {
            return PagedTiledFusedUpdateCacheMeshWorkloadFactory{};
        }
        return PagedTiledFusedUpdateCacheProgramFactory{};
    }
    if (input_tensor1.layout() == Layout::ROW_MAJOR && input_tensor2.layout() == Layout::ROW_MAJOR) {
        if (use_mesh_workload_factory) {
            return PagedRowMajorFusedUpdateCacheMeshWorkloadFactory{};
        }
        return PagedRowMajorFusedUpdateCacheProgramFactory{};
    }
    TT_FATAL(false, "input_tensor1 and input_tensor2 must be either both tiled or both row-major");
}

void PagedFusedUpdateCacheDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void PagedFusedUpdateCacheDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& cache_tensor1 = tensor_args.cache_tensor1;
    const auto& input_tensor1 = tensor_args.input_tensor1;
    const auto& cache_tensor2 = tensor_args.cache_tensor2;
    const auto& input_tensor2 = tensor_args.input_tensor2;
    const auto& update_idxs_tensor = tensor_args.update_idxs_tensor;
    const auto& page_table = tensor_args.page_table;

    // Common validation for both tensor pairs
    for (int i = 0; i < 2; ++i) {
        const auto& cache_tensor = (i == 0) ? cache_tensor1 : cache_tensor2;
        const auto& input_tensor = (i == 0) ? input_tensor1 : input_tensor2;

        // Device and storage validation
        TT_FATAL(
            input_tensor.storage_type() == StorageType::DEVICE && cache_tensor.storage_type() == StorageType::DEVICE,
            "Operands to update_cache need to be on device!");
        TT_FATAL(
            input_tensor.device() == cache_tensor.device(), "Operands to update_cache need to be on the same device!");
        TT_FATAL(
            input_tensor.buffer() != nullptr && cache_tensor.buffer() != nullptr,
            "Operands to update_cache need to be allocated in buffers on device!");

        // Layout and data type validation
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
        TT_FATAL(
            input_tensor.padded_shape()[-1] == cache_tensor.padded_shape()[-1],
            "Last dim of input tensor must match last dim of cache tensor");

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

            if (page_table.value().is_sharded()) {
                TT_FATAL(page_table.value().dtype() == DataType::UINT16, "Expect page table to have datatype UINT16");
            } else {
                TT_FATAL(page_table.value().dtype() == DataType::INT32, "Expect page table to have datatype INT32");
            }

            TT_FATAL(
                page_table.value().layout() == Layout::ROW_MAJOR,
                "Expect page table to have memory layout in ROW MAJOR");

            if (page_table.value().is_sharded()) {
                uint32_t num_cores = page_table.value().memory_config().shard_spec()->grid.num_cores();
                uint32_t page_table_shard_height = page_table.value().padded_shape()[0] / num_cores;
                TT_FATAL(
                    page_table_shard_height == input_tensor.padded_shape()[1],
                    "Batch size in input tensor {} should match page table shard height {}",
                    input_tensor.padded_shape()[1],
                    page_table_shard_height);
            } else {
                TT_FATAL(
                    page_table.value().padded_shape()[0] == input_tensor.padded_shape()[1],
                    "Batch size between page_table and input_tensor must match");
            }
            TT_FATAL(
                page_table.value().padded_shape()[1] <= cache_tensor.padded_shape()[0],
                "max_num_blocks_per_seq must be less than max_num_blocks: max_num_blocks_per_seq={}, "
                "max_num_blocks={}",
                page_table.value().padded_shape()[1],
                cache_tensor.padded_shape()[0]);
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
        }

        // Data type validation
        TT_FATAL(
            input_tensor.dtype() == DataType::FLOAT32 || input_tensor.dtype() == DataType::BFLOAT16,
            "Data type of input tensor for update cache must be FLOAT32 or BFLOAT16");

        TT_FATAL(operation_attributes.batch_offset == 0, "batch_offset must be 0");
    }

    // Fused update specific validation
    // Validate either both should be tiled or row-major
    bool is_tiled = input_tensor1.layout() == Layout::TILE && input_tensor2.layout() == Layout::TILE;
    bool is_row_major = input_tensor1.layout() == Layout::ROW_MAJOR && input_tensor2.layout() == Layout::ROW_MAJOR;

    TT_FATAL(is_tiled || is_row_major, "input_tensor1 and input_tensor2 must be either both tiled or both row-major");
    if (is_row_major) {
        TT_FATAL(
            input_tensor1.padded_shape()[-1] == 128 && input_tensor2.padded_shape()[-2] == 8,
            "when input_tensor1 and input_tensor2 are row major, only Llama70b tensor shapes are supported");
    }

    CoreRangeSet input1_cores = input_tensor1.shard_spec().value().grid;
    CoreRangeSet input2_cores = input_tensor2.shard_spec().value().grid;

    bool is_overlap = input1_cores.intersects(input2_cores);
    TT_FATAL(!is_overlap, "input_tensor1 ({}) and input_tensor2 ({}) must not overlap", input1_cores, input2_cores);
    TT_FATAL(
        input1_cores.num_cores() == input2_cores.num_cores(),
        "input_tensor1 ({}) and input_tensor2 ({}) must have same number of cores",
        input1_cores,
        input2_cores);
}

PagedFusedUpdateCacheDeviceOperation::spec_return_value_t PagedFusedUpdateCacheDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // Do nothing because it's an in-place operation
    return {tensor_args.cache_tensor1.tensor_spec(), tensor_args.cache_tensor2.tensor_spec()};
}

PagedFusedUpdateCacheDeviceOperation::tensor_return_value_t PagedFusedUpdateCacheDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // In-place operation, return the cache tensors
    return std::make_tuple(tensor_args.cache_tensor1, tensor_args.cache_tensor2);
}

tt::stl::hash::hash_t PagedFusedUpdateCacheDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    // Exclude runtime-only parameters from hash:
    // - update_idxs: values are runtime-only (used only in runtime args), size is validated to match input tensor shape
    // (already in tensor_args)
    // - batch_offset: validated to be 0, doesn't affect program structure
    // Include parameters that affect program structure:
    // - compute_kernel_config: affects compile-time args (fp32_dest_acc_en)
    // - share_cache: affects program structure (semaphore setup)
    // - mesh_coords: affects program factory selection
    return operation::hash_operation<PagedFusedUpdateCacheDeviceOperation>(
        operation_attributes.compute_kernel_config,
        operation_attributes.share_cache,
        operation_attributes.mesh_coords,
        tensor_args,
        program_factory.index());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::PagedFusedUpdateCacheResult paged_fused_update_cache(
    const Tensor& cache_tensor1,
    const Tensor& input_tensor1,
    const Tensor& cache_tensor2,
    const Tensor& input_tensor2,
    const std::vector<uint32_t>& update_idxs,
    const std::optional<const Tensor>& update_idxs_tensor,
    const std::optional<bool> share_cache,
    const std::optional<const Tensor>& page_table,
    const uint32_t batch_offset,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const std::set<ttnn::MeshCoordinate>>& mesh_coords) {
    using OperationType = ttnn::experimental::prim::PagedFusedUpdateCacheDeviceOperation;

    auto kernel_config_val = init_device_compute_kernel_config(input_tensor1.device()->arch(), compute_kernel_config);
    const bool share_cache_arg = share_cache.has_value() ? share_cache.value() : false;

    auto operation_attributes = OperationType::operation_attributes_t{
        .update_idxs = update_idxs,
        .batch_offset = batch_offset,
        .compute_kernel_config = kernel_config_val,
        .share_cache = share_cache_arg,
        .mesh_coords = mesh_coords};

    auto tensor_args = OperationType::tensor_args_t{
        .cache_tensor1 = cache_tensor1,
        .input_tensor1 = input_tensor1,
        .cache_tensor2 = cache_tensor2,
        .input_tensor2 = input_tensor2,
        .update_idxs_tensor =
            update_idxs_tensor.has_value() ? std::optional<Tensor>(update_idxs_tensor.value()) : std::nullopt,
        .page_table = page_table.has_value() ? std::optional<Tensor>(page_table.value()) : std::nullopt};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
