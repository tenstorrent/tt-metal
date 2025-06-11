// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_cache_operation.hpp"

#include "paged_update_cache_program_factory.hpp"
#include "paged_fused_update_cache_program_factory.hpp"
#include "paged_fill_cache_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::paged_cache {

void PagedUpdateCacheDeviceOperation::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    const auto validateInputTensorCount = [](PagedUpdateCacheOpType op_type, uint32_t num_input_tensors) {
        switch (op_type) {
            case PagedUpdateCacheOpType::UPDATE:
                TT_FATAL(num_input_tensors == 2, "Expect 2 input tensors for update_cache");
                break;
            case PagedUpdateCacheOpType::FUSED_UPDATE:
                TT_FATAL(num_input_tensors == 4, "Expect 4 input tensors for fused_update_cache");
                break;
            case PagedUpdateCacheOpType::FILL:
                TT_FATAL(num_input_tensors == 3, "Expect 3 input tensors for fill_cache");
                break;
            default: TT_FATAL(false, "Invalid op type");
        }
    };

    const auto validateTensorBasics = [this](const Tensor& cache_tensor, const Tensor& input_tensor) {
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
        if (this->op_type == PagedUpdateCacheOpType::UPDATE) {
            TT_FATAL(input_tensor.layout() == Layout::TILE, "Input tensor in non-fused update_cache must be tilized");
        }
        TT_FATAL(cache_tensor.layout() == Layout::TILE, "Cache tensor in update_cache must be tilized");
        TT_FATAL(
            cache_tensor.dtype() == DataType::FLOAT32 || cache_tensor.dtype() == DataType::BFLOAT16 ||
                cache_tensor.dtype() == DataType::BFLOAT8_B || cache_tensor.dtype() == DataType::BFLOAT4_B,
            "Data type of cache tensor must be FLOAT32, BFLOAT16, BFLOAT8_B, or BFLOAT4_B and is {}",
            cache_tensor.dtype());
    };

    const auto validateTensorShapes = [](const Tensor& cache_tensor, const Tensor& input_tensor) {
        TT_FATAL(input_tensor.padded_shape()[0] == 1, "Dim 0 of input tensor must be 1");
        TT_FATAL(
            cache_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Only interleaved cache is supported");
        TT_FATAL(
            input_tensor.padded_shape()[-1] == cache_tensor.padded_shape()[-1],
            "Last dim of input tensor must match last dim of cache tensor");
    };

    const auto validatePagedCache = [](bool paged_cache,
                                       bool share_cache,
                                       const Tensor& cache_tensor,
                                       const Tensor& input_tensor,
                                       const std::vector<std::optional<const Tensor>>& optional_input_tensors) {
        if (!paged_cache) {
            if (share_cache) {
                TT_FATAL(
                    cache_tensor.padded_shape()[0] == 1, "Share cache feature expects cache tensor to have batch of 1");
            } else {
                TT_FATAL(
                    input_tensor.padded_shape()[1] == cache_tensor.padded_shape()[0],
                    "Expect batch in input tensor match the batch in cache tensor");
            }
        } else {
            TT_FATAL(!share_cache, "share_cache not supported with paged cache");
            TT_FATAL(optional_input_tensors.at(0).has_value(), "Paged cache requires update_idxs tensor");

            auto page_table = optional_input_tensors.at(1).value();
            TT_FATAL(page_table.dtype() == DataType::INT32, "Error");
            TT_FATAL(page_table.layout() == Layout::ROW_MAJOR, "Error");
            TT_FATAL(
                page_table.padded_shape()[0] == input_tensor.padded_shape()[1],
                "Batch size between page_table and input_tensor must match");
            TT_FATAL(
                page_table.padded_shape()[1] <= cache_tensor.padded_shape()[0],
                "max_num_blocks_per_seq must be less than max_num_blocks");
        }
    };

    const auto validateUpdateIndices = [](const Tensor& input_tensor,
                                          const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                          const std::vector<uint32_t>& update_idxs) {
        TT_FATAL(
            optional_input_tensors.at(0).has_value() != update_idxs.size() > 0,
            "Only an update tensor or an update vector can be provided. Not both or neither.");

        uint32_t num_indices;
        if (optional_input_tensors.at(0).has_value()) {
            const auto& update_idxs_tensor = optional_input_tensors.at(0).value();
            TT_FATAL(update_idxs_tensor.dtype() == DataType::INT32, "Error");
            TT_FATAL(update_idxs_tensor.layout() == Layout::ROW_MAJOR, "Error");
            num_indices = update_idxs_tensor.padded_shape()[0];

            TT_FATAL(update_idxs_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
            TT_FATAL(update_idxs_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM, "Error");
        } else {
            num_indices = update_idxs.size();
        }

        TT_FATAL(input_tensor.padded_shape()[1] == num_indices, "Number of update_idxs should match batch size");
    };

    const auto validateSharding = [](const Tensor& input_tensor) {
        TT_FATAL(input_tensor.is_sharded(), "Error");
        if (input_tensor.is_sharded()) {
            TT_FATAL(input_tensor.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED, "Error");
            TT_FATAL(input_tensor.shard_spec().value().shape[1] == input_tensor.padded_shape()[-1], "Error");
            TT_FATAL(
                (input_tensor.physical_volume() / input_tensor.padded_shape()[-1]) %
                        input_tensor.shard_spec().value().shape[0] ==
                    0,
                "Error");
            TT_FATAL(
                input_tensor.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
                "Only ROW_MAJOR sharding is supported");
        }
    };

    const auto validateUpdateOperation = [validatePagedCache, validateUpdateIndices, validateSharding](
                                             const Tensor& cache_tensor,
                                             const Tensor& input_tensor,
                                             const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                             bool share_cache,
                                             const std::vector<uint32_t>& update_idxs,
                                             uint32_t batch_offset) {
        TT_FATAL(
            input_tensor.dtype() == DataType::FLOAT32 || input_tensor.dtype() == DataType::BFLOAT16,
            "Data type of input tensor for update cache must be FLOAT32 or BFLOAT16");

        const bool paged_cache = optional_input_tensors.at(1).has_value();
        validatePagedCache(paged_cache, share_cache, cache_tensor, input_tensor, optional_input_tensors);
        validateUpdateIndices(input_tensor, optional_input_tensors, update_idxs);
        validateSharding(input_tensor);

        TT_FATAL(batch_offset == 0, "Error");
    };

    const auto validateFillOperation = [](const Tensor& cache_tensor,
                                          const Tensor& input_tensor,
                                          const Tensor& page_table_tensor,
                                          uint32_t batch_idx) {
        TT_FATAL(
            input_tensor.dtype() == DataType::FLOAT32 || input_tensor.dtype() == DataType::BFLOAT16 ||
                cache_tensor.dtype() == DataType::BFLOAT8_B || cache_tensor.dtype() == DataType::BFLOAT4_B,
            "Data type of input tensor for fill cache must be FLOAT32, BFLOAT16, or BFLOAT8_b");

        TT_FATAL(input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
        TT_FATAL(page_table_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
        TT_FATAL(page_table_tensor.dtype() == DataType::INT32, "Error");

        auto cache_shape = cache_tensor.padded_shape();
        auto input_shape = input_tensor.padded_shape();
        auto page_table_shape = page_table_tensor.padded_shape();

        TT_FATAL(batch_idx <= cache_shape[0], "Error");
        TT_FATAL(
            input_shape[2] <= cache_shape[2] * page_table_shape[1], "Input seq_len must fit in max_num_blocks_per_seq");
    };

    const auto validateFusedUpdateTensors = [](const Tensor& input_tensor1, const Tensor& input_tensor2) {
        // Validate either both should be tiled or row-major
        bool is_tiled = input_tensor1.layout() == Layout::TILE && input_tensor2.layout() == Layout::TILE;
        bool is_row_major = input_tensor1.layout() == Layout::ROW_MAJOR && input_tensor2.layout() == Layout::ROW_MAJOR;

        TT_FATAL(
            is_tiled || is_row_major, "input_tensor1 and input_tensor2 must be either both tiled or both row-major");
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
    };

    // Validate number of input tensors based on operation type
    uint32_t num_input_tensors = input_tensors.size();
    validateInputTensorCount(this->op_type, num_input_tensors);

    if (this->op_type == PagedUpdateCacheOpType::UPDATE || this->op_type == PagedUpdateCacheOpType::FUSED_UPDATE) {
        // Common validation for all tensor pairs
        for (int i = 0; i < num_input_tensors; i += 2) {
            const auto& cache_tensor = input_tensors.at(i);
            const auto& input_tensor = input_tensors.at(i + 1);

            validateTensorBasics(cache_tensor, input_tensor);
            validateTensorShapes(cache_tensor, input_tensor);
            validateUpdateOperation(
                cache_tensor,
                input_tensor,
                optional_input_tensors,
                this->share_cache,
                this->update_idxs,
                this->batch_offset);
        }
        if (this->op_type == PagedUpdateCacheOpType::FUSED_UPDATE) {
            validateFusedUpdateTensors(input_tensors.at(1), input_tensors.at(3));
        }
    } else if (this->op_type == PagedUpdateCacheOpType::FILL) {
        // Validate based on batch_idx_fallback for the host-side check
        validateFillOperation(input_tensors.at(0), input_tensors.at(1), input_tensors.at(2), this->batch_idx_fallback);
        if (this->batch_idx_tensor_opt.has_value()) {
            const auto& tensor = this->batch_idx_tensor_opt.value();
            TT_FATAL(tensor.physical_volume() == 1, "Batch idx tensor must have a single element");
            TT_FATAL(
                tensor.dtype() == DataType::UINT32 || tensor.dtype() == DataType::INT32,
                "Batch idx tensor must be an integer type");
            // Add any other necessary validation for the tensor itself
        }
    }
}

std::vector<ttnn::TensorSpec> PagedUpdateCacheDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    // Do nothing because it's an in-place operation
    return {};
}

operation::ProgramWithCallbacks PagedUpdateCacheDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    switch (this->get_parallelization_strategy(input_tensors)) {
        case PagedUpdateCacheOpParallelizationStrategy::MULTI_CORE:
        default:
            if (this->op_type == PagedUpdateCacheOpType::UPDATE) {
                const auto& cache_tensor = input_tensors.at(0);
                const auto& input_tensor = input_tensors.at(1);
                const auto update_idxs_tensor =
                    optional_input_tensors.at(0);  // TODO: Is this tensor passed around by value?
                const auto page_table = optional_input_tensors.at(1);
                return detail::paged_update_cache_multi_core(
                    cache_tensor,
                    input_tensor,
                    update_idxs_tensor,
                    page_table,
                    this->update_idxs,
                    this->batch_offset,
                    this->compute_kernel_config,
                    this->share_cache);
            } else if (this->op_type == PagedUpdateCacheOpType::FUSED_UPDATE) {
                const auto& cache_tensor1 = input_tensors.at(0);
                const auto& input_tensor1 = input_tensors.at(1);
                const auto& cache_tensor2 = input_tensors.at(2);
                const auto& input_tensor2 = input_tensors.at(3);
                const auto update_idxs_tensor =
                    optional_input_tensors.at(0);  // TODO: Is this tensor passed around by value?
                const auto page_table = optional_input_tensors.at(1);
                return detail::paged_fused_update_cache_multi_core(
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
                const auto& cache_tensor = input_tensors.at(0);
                const auto& input_tensor = input_tensors.at(1);
                const auto& page_table = input_tensors.at(2);
                return detail::paged_fill_cache_multi_core(
                    cache_tensor, input_tensor, page_table, this->batch_idx_tensor_opt, this->batch_idx_fallback);
            }
    };
}

PagedUpdateCacheOpParallelizationStrategy PagedUpdateCacheDeviceOperation::get_parallelization_strategy(
    const std::vector<Tensor>& input_tensors) const {
    return PagedUpdateCacheOpParallelizationStrategy::MULTI_CORE;
}

operation::Hash PagedUpdateCacheDeviceOperation::compute_program_hash(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    return operation::hash_operation<PagedUpdateCacheDeviceOperation>(
        this->op_type, input_tensors, optional_input_tensors);
}

}  // namespace ttnn::operations::experimental::paged_cache
