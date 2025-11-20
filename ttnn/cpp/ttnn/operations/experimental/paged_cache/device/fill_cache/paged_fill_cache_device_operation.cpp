// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_fill_cache_device_operation.hpp"

#include "paged_fill_cache_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::paged_cache {

void PagedFillCacheDeviceOperation::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 3, "Expect 3 input tensors for fill_cache");

    const auto& cache_tensor = input_tensors.at(0);
    const auto& input_tensor = input_tensors.at(1);
    const auto& page_table_tensor = input_tensors.at(2);

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

    TT_FATAL(batch_idx_fallback <= cache_shape[0], "Batch idx must fit in cache batch size");
    TT_FATAL(
        input_shape[2] <= cache_shape[2] * page_table_shape[1], "Input seq_len must fit in max_num_blocks_per_seq");

    if (this->batch_idx_tensor_opt.has_value()) {
        const auto& tensor = this->batch_idx_tensor_opt.value();
        TT_FATAL(tensor.physical_volume() == 1, "Batch idx tensor must have a single element");
        TT_FATAL(
            tensor.dtype() == DataType::UINT32 || tensor.dtype() == DataType::INT32,
            "Batch idx tensor must be an integer type");
    }
}

std::vector<ttnn::TensorSpec> PagedFillCacheDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    // Do nothing because it's an in-place operation
    return {};
}

operation::MeshWorkloadWithCallbacks PagedFillCacheDeviceOperation::create_mesh_workload(
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
            auto program_with_callbacks = PagedFillCacheDeviceOperation::create_program_at(
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

operation::ProgramWithCallbacks PagedFillCacheDeviceOperation::create_program_at(
    const ttnn::MeshCoordinate& _,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    switch (this->get_parallelization_strategy(input_tensors)) {
        case PagedUpdateCacheOpParallelizationStrategy::MULTI_CORE:
        default:
            const auto& cache_tensor = input_tensors.at(0);
            const auto& input_tensor = input_tensors.at(1);
            const auto& page_table = input_tensors.at(2);
            return detail::paged_fill_cache_multi_core(
                cache_tensor, input_tensor, page_table, this->batch_idx_tensor_opt, this->batch_idx_fallback);
    }
}

PagedUpdateCacheOpParallelizationStrategy PagedFillCacheDeviceOperation::get_parallelization_strategy(
    const std::vector<Tensor>& input_tensors) const {
    return PagedUpdateCacheOpParallelizationStrategy::MULTI_CORE;
}

operation::Hash PagedFillCacheDeviceOperation::compute_program_hash(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    return operation::hash_operation<PagedFillCacheDeviceOperation>(
        input_tensors, optional_input_tensors, this->mesh_coords);
}

}  // namespace ttnn::operations::experimental::paged_cache
