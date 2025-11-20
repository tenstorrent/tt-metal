// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_update_cache_device_operation.hpp"

#include "paged_update_cache_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::paged_cache {

void PagedUpdateCacheDeviceOperation::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 2, "Expect 2 input tensors for update_cache");

    const auto& cache_tensor = input_tensors.at(0);
    const auto& input_tensor = input_tensors.at(1);

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

std::vector<ttnn::TensorSpec> PagedUpdateCacheDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    // Do nothing because it's an in-place operation
    return {};
}

operation::MeshWorkloadWithCallbacks PagedUpdateCacheDeviceOperation::create_mesh_workload(
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
            auto program_with_callbacks = PagedUpdateCacheDeviceOperation::create_program_at(
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

operation::ProgramWithCallbacks PagedUpdateCacheDeviceOperation::create_program_at(
    const ttnn::MeshCoordinate& _,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    switch (this->get_parallelization_strategy(input_tensors)) {
        case PagedUpdateCacheOpParallelizationStrategy::MULTI_CORE:
        default:
            const auto& cache_tensor = input_tensors.at(0);
            const auto& input_tensor = input_tensors.at(1);
            const auto& update_idxs_tensor =
                optional_input_tensors.at(0);  // TODO: Is this tensor passed around by value?
            const auto& page_table = optional_input_tensors.at(1);
            return detail::paged_update_cache_multi_core(
                cache_tensor,
                input_tensor,
                update_idxs_tensor,
                page_table,
                this->update_idxs,
                this->batch_offset,
                this->compute_kernel_config,
                this->share_cache);
    }
}

PagedUpdateCacheOpParallelizationStrategy PagedUpdateCacheDeviceOperation::get_parallelization_strategy(
    const std::vector<Tensor>& input_tensors) const {
    return PagedUpdateCacheOpParallelizationStrategy::MULTI_CORE;
}

operation::Hash PagedUpdateCacheDeviceOperation::compute_program_hash(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    return operation::hash_operation<PagedUpdateCacheDeviceOperation>(
        input_tensors, optional_input_tensors, this->mesh_coords);
}

}  // namespace ttnn::operations::experimental::paged_cache
