// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexed_entry_copy_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim {

namespace {

constexpr std::string_view kOperation = "indexed_entry_copy";

void check_dram_operand(const Tensor& tensor, std::string_view name) {
    kda_factory_detail::check_allocated_device_tensor(tensor, kOperation, name);
    const bool interleaved = tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
    const bool nd_sharded_tiles =
        tensor.memory_config().nd_shard_spec().has_value() && tensor.layout() == tt::tt_metal::Layout::TILE;
    TT_FATAL(
        tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM && (interleaved || nd_sharded_tiles),
        "{}: {} must be interleaved DRAM, or ND-sharded DRAM for TILE",
        kOperation,
        name);
}

}  // namespace

void IndexedEntryCopyDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& params, const tensor_args_t& inputs) {
    const auto& source = inputs.source;
    const auto& destination = inputs.destination;

    check_dram_operand(source, "source");
    check_dram_operand(destination, "destination");
    kda_factory_detail::check_same_device(source, destination, kOperation, "destination");
    kda_factory_detail::check_device_scalar(source, inputs.index, kOperation, "index");
    kda_factory_detail::check_matching_dtype(source, destination, kOperation, "source and destination");

    // An entry is a dim-0 slice; for TILE, rank >= 3 keeps dim 0 out of the tiled dims.
    const auto& source_shape = source.logical_shape();
    const auto& destination_shape = destination.logical_shape();
    TT_FATAL(
        source.layout() == destination.layout() && source.tensor_spec().tile() == destination.tensor_spec().tile() &&
            source.buffer()->page_size() == destination.buffer()->page_size(),
        "{}: source and destination must share layout, tile and page size",
        kOperation);
    TT_FATAL(
        source_shape.rank() == destination_shape.rank() &&
            (source.layout() != tt::tt_metal::Layout::TILE || source_shape.rank() >= 3),
        "{}: source and destination must have equal rank, at least 3 for TILE",
        kOperation);
    // The program strides both sides by one entry of the single-entry side's padded shape, so the padding must agree
    // past dim 0 and leave dim 0 unpadded.
    const auto& source_padded = source.padded_shape();
    const auto& destination_padded = destination.padded_shape();
    TT_FATAL(
        source_padded[0] == source_shape[0] && destination_padded[0] == destination_shape[0],
        "{}: source {} / destination {} must not pad dim 0",
        kOperation,
        source_padded,
        destination_padded);
    for (int dim = 1; dim < static_cast<int>(source_shape.rank()); ++dim) {
        TT_FATAL(
            source_shape[dim] == destination_shape[dim],
            "{}: source {} and destination {} differ past dim 0",
            kOperation,
            source_shape,
            destination_shape);
        TT_FATAL(
            source_padded[dim] == destination_padded[dim],
            "{}: source padded {} and destination padded {} differ past dim 0",
            kOperation,
            source_padded,
            destination_padded);
    }
    TT_FATAL(
        source_shape[0] == 1 || destination_shape[0] == 1,
        "{}: one of source {} / destination {} must hold a single entry (dim 0 == 1)",
        kOperation,
        source_shape,
        destination_shape);
    TT_FATAL(params.index_scale >= 1, "{}: index_scale must be at least 1", kOperation);
}

IndexedEntryCopyDeviceOperation::spec_return_value_t IndexedEntryCopyDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& inputs) {
    return inputs.destination.tensor_spec();
}

IndexedEntryCopyDeviceOperation::tensor_return_value_t IndexedEntryCopyDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& inputs) {
    return inputs.destination;
}

Tensor indexed_entry_copy(
    const Tensor& source, const Tensor& destination, const Tensor& index, uint32_t index_scale, uint32_t index_offset) {
    using OperationType = ttnn::experimental::prim::IndexedEntryCopyDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.index_scale = index_scale, .index_offset = index_offset},
        OperationType::tensor_args_t{.source = source, .destination = destination, .index = index});
}

}  // namespace ttnn::experimental::prim
