// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_fused_update_cache_device_operation.hpp"

#include "paged_tiled_fused_update_cache_program_factory.hpp"
#include "paged_row_major_fused_update_cache_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::paged_cache {

void PagedFusedUpdateCacheDeviceOperation::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 4, "Expect 4 input tensors for fused_update_cache");

    // Common validation for both tensor pairs
    for (int i = 0; i < 4; i += 2) {
        const auto& cache_tensor = input_tensors.at(i);
        const auto& input_tensor = input_tensors.at(i + 1);

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
        const bool paged_cache = optional_input_tensors.at(1).has_value();
        if (!paged_cache) {
            if (this->share_cache) {
                TT_FATAL(
                    cache_tensor.padded_shape()[0] == 1, "Share cache feature expects cache tensor to have batch of 1");
            } else {
                TT_FATAL(
                    input_tensor.padded_shape()[1] == cache_tensor.padded_shape()[0],
                    "Expect batch in input tensor match the batch in cache tensor");
            }
        } else {
            TT_FATAL(!this->share_cache, "share_cache not supported with paged cache");
            TT_FATAL(optional_input_tensors.at(0).has_value(), "Paged cache requires update_idxs tensor");

            auto page_table = optional_input_tensors.at(1).value();

            if (page_table.is_sharded()) {
                TT_FATAL(page_table.dtype() == DataType::UINT16, "Expect page table to have datatype UINT16");
            } else {
                TT_FATAL(page_table.dtype() == DataType::INT32, "Expect page table to have datatype INT32");
            }

            TT_FATAL(page_table.layout() == Layout::ROW_MAJOR, "Expect page table to have memory layout in ROW MAJOR");

            if (page_table.is_sharded()) {
                uint32_t num_cores = page_table.memory_config().shard_spec()->grid.num_cores();
                uint32_t page_table_shard_height = page_table.padded_shape()[0] / num_cores;
                TT_FATAL(
                    page_table_shard_height == input_tensor.padded_shape()[1],
                    "Batch size in input tensor {} should match page table shard height {}",
                    input_tensor.padded_shape()[1],
                    page_table_shard_height);
            } else {
                TT_FATAL(
                    page_table.padded_shape()[0] == input_tensor.padded_shape()[1],
                    "Batch size between page_table and input_tensor must match");
            }
            TT_FATAL(
                page_table.padded_shape()[1] <= cache_tensor.padded_shape()[0],
                "max_num_blocks_per_seq must be less than max_num_blocks: max_num_blocks_per_seq={}, "
                "max_num_blocks={}",
                page_table.padded_shape()[1],
                cache_tensor.padded_shape()[0]);
        }

        // Update indices validation
        TT_FATAL(
            (optional_input_tensors.at(0).has_value()) != (!this->update_idxs.empty()),
            "Only an update tensor or an update vector can be provided. Not both or neither.");

        uint32_t num_indices = 0;
        uint32_t num_cores_cur_pos = 0;
        if (optional_input_tensors.at(0).has_value()) {
            const auto& update_idxs_tensor = optional_input_tensors.at(0).value();
            TT_FATAL(update_idxs_tensor.dtype() == DataType::INT32, "Expected update_idxs to have datatype INT32");
            TT_FATAL(
                update_idxs_tensor.layout() == Layout::ROW_MAJOR,
                "Expected update_idxs to have memory layout in ROW MAJOR");

            if (optional_input_tensors.at(0)->is_sharded()) {
                TT_FATAL(
                    update_idxs_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
                    "Expect update_idxs to be HEIGHT SHARDED if sharded");
                TT_FATAL(
                    update_idxs_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::L1,
                    "Expect update_idxs to have buffer type L1 if sharded");
                num_cores_cur_pos = update_idxs_tensor.padded_shape()[0];
                num_indices = update_idxs_tensor.logical_shape()[1];
            } else {
                TT_FATAL(
                    update_idxs_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
                    "Expect update_idxs to be DRAM INTERLEAVED");
                TT_FATAL(
                    update_idxs_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
                    "Expect update_idxs to have buffer type DRAM");
                num_cores_cur_pos = 0;
                num_indices = update_idxs_tensor.padded_shape()[0];
            }
        } else {
            num_indices = this->update_idxs.size();
        }
        if (optional_input_tensors.at(0).has_value() && optional_input_tensors.at(0)->is_sharded()) {
            uint32_t in_num_cores_cur_pos = optional_input_tensors.at(0)->shard_spec().value().grid.num_cores();
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

        TT_FATAL(this->batch_offset == 0, "batch_offset must be 0");
    }

    // Fused update specific validation
    const auto& input_tensor1 = input_tensors.at(1);
    const auto& input_tensor2 = input_tensors.at(3);
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

std::vector<ttnn::TensorSpec> PagedFusedUpdateCacheDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    // Do nothing because it's an in-place operation
    return {};
}

operation::MeshWorkloadWithCallbacks PagedFusedUpdateCacheDeviceOperation::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    operation::MeshWorkloadWithCallbacks workload_with_callbacks;
    for (const auto& range : tensor_coords.ranges()) {
        for (const auto& coord : range) {
            // If mesh_coords is provided, check if the coordinate is in the set
            if (this->mesh_coords.has_value()) {
                bool enable_on_coord =
                    std::find(this->mesh_coords->begin(), this->mesh_coords->end(), coord) != this->mesh_coords->end();
                if (!enable_on_coord) {
                    continue;  // Skip this coordinate if it's not in the mesh_coords set
                }
            }

            // Create the program for the coordinate
            const ttnn::MeshCoordinateRange program_range(coord, coord);
            auto program_with_callbacks = PagedFusedUpdateCacheDeviceOperation::create_program_at(
                {0, 0}, input_tensors, optional_input_tensors, output_tensors);
            workload_with_callbacks.workload.add_program(program_range, std::move(program_with_callbacks.program));
            if (program_with_callbacks.override_runtime_arguments_callback.has_value()) {
                workload_with_callbacks.per_program_callbacks.emplace(
                    program_range, std::move(*program_with_callbacks.override_runtime_arguments_callback));
            }
        }
    }
    return workload_with_callbacks;
}

operation::ProgramWithCallbacks PagedFusedUpdateCacheDeviceOperation::create_program_at(
    const ttnn::MeshCoordinate& _,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    switch (this->get_parallelization_strategy(input_tensors)) {
        case PagedUpdateCacheOpParallelizationStrategy::MULTI_CORE:
        default:
            const auto& cache_tensor1 = input_tensors.at(0);
            const auto& input_tensor1 = input_tensors.at(1);
            const auto& cache_tensor2 = input_tensors.at(2);
            const auto& input_tensor2 = input_tensors.at(3);
            const auto& update_idxs_tensor =
                optional_input_tensors.at(0);  // TODO: Is this tensor passed around by value?
            const auto& page_table = optional_input_tensors.at(1);
            if (input_tensor1.layout() == Layout::TILE && input_tensor2.layout() == Layout::TILE) {
                return detail::paged_tiled_fused_update_cache_multi_core(
                    cache_tensor1,
                    input_tensor1,
                    cache_tensor2,
                    input_tensor2,
                    update_idxs_tensor,
                    page_table,
                    this->update_idxs,
                    this->batch_offset,
                    this->compute_kernel_config,
                    this->share_cache);
            } else if (input_tensor1.layout() == Layout::ROW_MAJOR && input_tensor2.layout() == Layout::ROW_MAJOR) {
                return detail::paged_row_major_fused_update_cache_multi_core(
                    cache_tensor1,
                    input_tensor1,
                    cache_tensor2,
                    input_tensor2,
                    update_idxs_tensor,
                    page_table,
                    this->update_idxs,
                    this->batch_offset,
                    this->compute_kernel_config,
                    this->share_cache);
            } else {
                TT_FATAL(false, "Error: input tensor1 and input tensor2 must be either both tiled or both row-major");
            }
    }
}

PagedUpdateCacheOpParallelizationStrategy PagedFusedUpdateCacheDeviceOperation::get_parallelization_strategy(
    const std::vector<Tensor>& input_tensors) const {
    return PagedUpdateCacheOpParallelizationStrategy::MULTI_CORE;
}

operation::Hash PagedFusedUpdateCacheDeviceOperation::compute_program_hash(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    return operation::hash_operation<PagedFusedUpdateCacheDeviceOperation>(
        input_tensors, optional_input_tensors, this->mesh_coords);
}

}  // namespace ttnn::operations::experimental::paged_cache
