// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_fill_cache_device_operation.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::paged_cache::fill {

PagedFillCacheDeviceOperation::program_factory_t PagedFillCacheDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& /*tensor_args*/) {
    // Use mesh workload factory when mesh_coords is provided to enable coordinate filtering
    if (args.mesh_coords.has_value()) {
        return program::PagedFillCacheMeshWorkloadFactory{};
    }
    return program::PagedFillCacheProgramFactory{};
}

void PagedFillCacheDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
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

    TT_FATAL(args.batch_idx_fallback <= cache_shape[0], "Batch idx must fit in cache batch size");
    TT_FATAL(
        input_shape[2] <= cache_shape[2] * page_table_shape[1], "Input seq_len must fit in max_num_blocks_per_seq");

    if (tensor_args.batch_idx_tensor_opt.has_value()) {
        const auto& tensor = tensor_args.batch_idx_tensor_opt.value();
        TT_FATAL(tensor.physical_volume() == 1, "Batch idx tensor must have a single element");
        TT_FATAL(
            tensor.dtype() == DataType::UINT32 || tensor.dtype() == DataType::INT32,
            "Batch idx tensor must be an integer type");
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

tt::stl::hash::hash_t PagedFillCacheDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto program_factory = select_program_factory(args, tensor_args);

    // Exclude batch_idx_fallback and noop from hash since they're runtime-only parameters (used only in runtime args)
    // Include mesh_coords since it affects program factory selection
    return operation::hash_operation<PagedFillCacheDeviceOperation>(
        args.mesh_coords, tensor_args, program_factory.index());
}

}  // namespace ttnn::operations::experimental::paged_cache::fill

namespace ttnn::prim {

ttnn::operations::experimental::paged_cache::fill::PagedFillCacheDeviceOperation::tensor_return_value_t
paged_fill_cache(
    const Tensor& cache_tensor,
    const Tensor& input_tensor,
    const Tensor& page_table,
    const std::optional<Tensor>& batch_idx_tensor,
    uint32_t batch_idx_fallback,
    const std::optional<std::set<ttnn::MeshCoordinate>>& mesh_coords) {
    using OperationType = ttnn::operations::experimental::paged_cache::fill::PagedFillCacheDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .batch_idx_fallback = batch_idx_fallback,
        .mesh_coords = mesh_coords,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .cache_tensor = cache_tensor,
        .input_tensor = input_tensor,
        .page_table = page_table,
        .batch_idx_tensor_opt = batch_idx_tensor,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
