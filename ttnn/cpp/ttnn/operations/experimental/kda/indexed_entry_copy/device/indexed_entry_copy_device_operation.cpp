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
        tensor.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM && (interleaved || nd_sharded_tiles),
        "{}: {} must be interleaved DRAM, or ND-sharded DRAM for TILE",
        kOperation,
        name);
}

// One (source, destination) pair: the entry layout the program assumes.
void validate_pair(const IndexedEntryCopyInputs& inputs, const Tensor& source, const Tensor& destination) {
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
            source.mesh_buffer().page_size() == destination.mesh_buffer().page_size(),
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
}

}  // namespace

void IndexedEntryCopyDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& params, const tensor_args_t& inputs) {
    validate_pair(inputs, inputs.source, inputs.destination);
    TT_FATAL(
        inputs.paired_source.has_value() == inputs.paired_destination.has_value(),
        "{}: pass both paired_source and paired_destination, or neither",
        kOperation);
    if (inputs.paired_source.has_value()) {
        validate_pair(inputs, *inputs.paired_source, *inputs.paired_destination);
        kda_factory_detail::check_same_device(inputs.source, *inputs.paired_source, kOperation, "paired_source");
        // Both entries index the same dim-0 side, so one index serves the pair.
        TT_FATAL(
            (inputs.source.logical_shape()[0] > 1) == (inputs.paired_source->logical_shape()[0] > 1),
            "{}: the pair must index the same side (source or destination) as the first copy",
            kOperation);
    }
    TT_FATAL(params.index_scale >= 1, "{}: index_scale must be at least 1", kOperation);
    if (inputs.zero_fill_at_zero.has_value()) {
        kda_factory_detail::check_device_scalar(
            inputs.source, *inputs.zero_fill_at_zero, kOperation, "zero_fill_at_zero");
    }
}

void IndexedEntryCopyDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t& inputs) {
    // A hit has the cached program's shapes, layouts and memory configs (they are in the key): only what the key does
    // not pin can fail, the tensors being allocated on the device.
    for (const auto* tensor :
         {&inputs.source,
          &inputs.destination,
          &inputs.index,
          inputs.zero_fill_at_zero ? &*inputs.zero_fill_at_zero : nullptr,
          inputs.paired_source ? &*inputs.paired_source : nullptr,
          inputs.paired_destination ? &*inputs.paired_destination : nullptr}) {
        TT_FATAL(
            tensor == nullptr || (tensor->storage_type() == StorageType::DEVICE && tensor->is_allocated()),
            "{}: every tensor must be allocated on the device",
            kOperation);
    }
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
    const Tensor& source,
    const Tensor& destination,
    const Tensor& index,
    uint32_t index_scale,
    uint32_t index_offset,
    const std::optional<Tensor>& zero_fill_at_zero,
    const std::optional<Tensor>& paired_source,
    const std::optional<Tensor>& paired_destination) {
    using OperationType = ttnn::experimental::prim::IndexedEntryCopyDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.index_scale = index_scale, .index_offset = index_offset},
        OperationType::tensor_args_t{
            .source = source,
            .destination = destination,
            .index = index,
            .zero_fill_at_zero = zero_fill_at_zero,
            .paired_source = paired_source,
            .paired_destination = paired_destination});
}

}  // namespace ttnn::experimental::prim
