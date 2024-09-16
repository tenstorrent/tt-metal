// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_cache_operation.hpp"

#include "paged_update_cache_program_factory.hpp"
#include "paged_fill_cache_program_factory.hpp"

namespace ttnn::operations::experimental::paged_cache {


void PagedUpdateCacheDeviceOperation::validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    const auto& cache_tensor = input_tensors.at(0);
    const auto& input_tensor = input_tensors.at(1);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE and cache_tensor.storage_type() == StorageType::DEVICE, "Operands to update_cache need to be on device!");
    TT_FATAL(input_tensor.device() == cache_tensor.device(), "Operands to update_cache need to be on the same device!");
    TT_FATAL(input_tensor.buffer() != nullptr and cache_tensor.buffer() != nullptr, "Operands to update_cache need to be allocated in buffers on device!");
    TT_FATAL((input_tensor.get_layout() == Layout::TILE && cache_tensor.get_layout() == Layout::TILE), "Inputs to update_cache must be tilized");
    TT_FATAL(input_tensor.get_dtype() == DataType::FLOAT32 || input_tensor.get_dtype() == DataType::BFLOAT16, "Data type of input tensor must be FLOAT32 or BFLOAT16");
    TT_FATAL(cache_tensor.get_dtype() == DataType::FLOAT32 || cache_tensor.get_dtype() == DataType::BFLOAT16 || cache_tensor.get_dtype() == DataType::BFLOAT8_B, "Data type of cache tensor must be FLOAT32, BFLOAT16 or BFLOAT8_B");

    // input_tensor: [1, b, padded_heads, head_dim]
    // cache_tensor: [b, n_heads, kv_len, head_dim]
    TT_FATAL(input_tensor.get_legacy_shape()[0] == 1, "Dim 0 of input tensor must be 1");
    TT_FATAL(cache_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Only interleaved cache is supported");
    TT_FATAL(input_tensor.get_legacy_shape()[-1] == cache_tensor.get_legacy_shape()[-1], "Last dim of input tensor must match last dim of cache tensor");

    if (this->op_type == PagedUpdateCacheOpType::UPDATE) {

        const bool paged_cache = optional_input_tensors.at(1).has_value();
        uint32_t batch_size;
        if (!paged_cache) {
            // TT_FATAL(cache_tensor.get_legacy_shape()[1] == 1, "Only supports 1 head now.");
            //TT_FATAL(input_tensor.get_shape()[2] == cache_tensor.get_shape()[1], "Error");
            if (this->share_cache){
                TT_FATAL(cache_tensor.get_legacy_shape()[0] == 1, "Error");
            }
            else {
                TT_FATAL(input_tensor.get_legacy_shape()[1] == cache_tensor.get_legacy_shape()[0], "Error");
            }
        } else {
            TT_FATAL(optional_input_tensors.at(0).has_value(), "Paged cache requires update_idxs tensor");
            // TODO: How to validate page_table and paged_cache?
            auto page_table = optional_input_tensors.at(1).value();
            TT_FATAL(page_table.get_dtype() == DataType::INT32, "Error");
            TT_FATAL(page_table.get_layout() == Layout::ROW_MAJOR, "Error");
            TT_FATAL(page_table.get_legacy_shape()[0] == input_tensor.get_legacy_shape()[1], "Batch size between page_table and input_tensor must match");
            TT_FATAL(page_table.get_legacy_shape()[1] <= cache_tensor.get_legacy_shape()[0], "max_num_blocks_per_seq must be less than max_num_blocks");
        }


        TT_FATAL(optional_input_tensors.at(0).has_value() != this->update_idxs.size() > 0, "Only an update tensor or an update vector can be provided. Not both or neither.");

        uint32_t num_indices;
        if (optional_input_tensors.at(0).has_value()) {
            const auto& update_idxs_tensor = optional_input_tensors.at(0).value();
            TT_FATAL(update_idxs_tensor.get_dtype() == DataType::INT32, "Error");
            TT_FATAL(update_idxs_tensor.get_layout() == Layout::ROW_MAJOR, "Error");
            // update_idxs_tensor: [num_indices]
            num_indices = update_idxs_tensor.get_legacy_shape()[0];

            // must be iterleaved
            TT_FATAL(update_idxs_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
            // must be in dram
            TT_FATAL(update_idxs_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM, "Error");
        } else {
            num_indices = this->update_idxs.size();
        }

        TT_FATAL(input_tensor.get_legacy_shape()[1] == num_indices, "Number of update_idxs should match batch size");

        TT_FATAL(input_tensor.is_sharded(), "Error");
        if (input_tensor.is_sharded()) {
            TT_FATAL(input_tensor.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED, "Error");
            TT_FATAL(input_tensor.shard_spec().value().shape[1] == input_tensor.get_legacy_shape()[-1], "Error");
            // Require even work division for now
            // TT_FATAL(input_tensor.shard_spec().value().grid.num_cores() == cache_tensor.get_legacy_shape()[0], "Input must be sharded on batch num cores");
            TT_FATAL((input_tensor.volume() / input_tensor.get_legacy_shape()[-1]) % input_tensor.shard_spec().value().shape[0] == 0, "Error");
            TT_FATAL(input_tensor.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR, "Only ROW_MAJOR sharding is supported");
        }
        // TT_FATAL(cache_tensor.get_legacy_shape()[0] <= input_tensor.get_legacy_shape()[-1], "Error");
        // batch offset is only valid if num_user less than 32 and batch_offset + num_user <= 32
        // if (cache_tensor.get_legacy_shape()[0] < 32) TT_FATAL(this->batch_offset + cache_tensor.get_legacy_shape()[0] <= 32, "Error");
        // else TT_FATAL(this->batch_offset == 0, "Error");
        TT_FATAL(this->batch_offset == 0, "Error");
    } else {

        TT_FATAL(this->op_type == PagedUpdateCacheOpType::FILL, "Error");
        const auto& page_table_tensor = input_tensors.at(2);

        TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
        TT_FATAL(page_table_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");

        TT_FATAL(page_table_tensor.get_dtype() == DataType::INT32, "Error");

        auto cache_shape = cache_tensor.get_legacy_shape();
        auto input_shape = input_tensor.get_legacy_shape();
        auto page_table_shape = page_table_tensor.get_legacy_shape();

        TT_FATAL(cache_shape[1] == 1, "Error");
        TT_FATAL(this->batch_idx <= cache_shape[0], "Error");
        TT_FATAL(input_shape[2] <= cache_shape[2] * page_table_shape[1], "Input seq_len must fit in max_num_blocks_per_seq");
    }
}

const std::vector<tt::tt_metal::LegacyShape> PagedUpdateCacheDeviceOperation::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    // Do nothing because it's an in-place operation
    return {};
}

std::vector<Tensor> PagedUpdateCacheDeviceOperation::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    // Do nothing because it's an in-place operation
    return {};
}

operation::ProgramWithCallbacks PagedUpdateCacheDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& cache_tensor = input_tensors.at(0);
    const auto& input_tensor = input_tensors.at(1);


    switch(this->get_parallelization_strategy(input_tensors)) {
        case PagedUpdateCacheOpParallelizationStrategy::MULTI_CORE:
        default:
            if (this->op_type == PagedUpdateCacheOpType::UPDATE) {
                const auto update_idxs_tensor = optional_input_tensors.at(0); // TODO: Is this tensor passed around by value?
                const auto page_table = optional_input_tensors.at(1);
                return detail::paged_update_cache_multi_core(cache_tensor, input_tensor, update_idxs_tensor, page_table, this->update_idxs, this->batch_offset, this->compute_kernel_config, this->share_cache);
            } else {
                const auto& page_table = input_tensors.at(2);
                return detail::paged_fill_cache_multi_core(cache_tensor, input_tensor, page_table, this->batch_idx);
            }
    };
}


PagedUpdateCacheOpParallelizationStrategy PagedUpdateCacheDeviceOperation::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    return PagedUpdateCacheOpParallelizationStrategy::MULTI_CORE;
}

const operation::Hash PagedUpdateCacheDeviceOperation::compute_program_hash(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    return operation::hash_operation<PagedUpdateCacheDeviceOperation>(this->op_type, input_tensors, optional_input_tensors);
}

}  // namespace ttnn::operations::experimental::paged_cache
