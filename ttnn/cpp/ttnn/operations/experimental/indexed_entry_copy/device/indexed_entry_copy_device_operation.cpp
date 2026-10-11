// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexed_entry_copy_device_operation.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

namespace {

void validate_dram_operand(const Tensor& tensor, const char* name) {
    TT_FATAL(tensor.storage_type() == ttnn::StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must be allocated", name);
    TT_FATAL(tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM, "{} must be in DRAM", name);
    const auto& memory_config = tensor.memory_config();
    TT_FATAL(
        memory_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED ||
            memory_config.nd_shard_spec().has_value(),
        "{} must be interleaved or ND-sharded DRAM",
        name);
}

}  // namespace

uint32_t indexed_entry_copy_num_pages(const Tensor& tensor) {
    const auto& shape = tensor.padded_shape();
    if (tensor.layout() == tt::tt_metal::Layout::TILE) {
        const auto& tile = tensor.tensor_spec().tile();
        return shape.volume() / (tile.get_height() * tile.get_width());
    }
    return shape.volume() / shape[-1];
}

void IndexedEntryCopyDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& params, const tensor_args_t& inputs) {
    const auto& source = inputs.source;
    const auto& destination = inputs.destination;
    const auto& index = inputs.index;

    validate_dram_operand(source, "source");
    validate_dram_operand(destination, "destination");
    TT_FATAL(source.dtype() == destination.dtype(), "source and destination dtypes differ");
    TT_FATAL(source.layout() == destination.layout(), "source and destination layouts differ");
    TT_FATAL(
        source.buffer()->page_size() == destination.buffer()->page_size(),
        "source page size {} != destination page size {}",
        source.buffer()->page_size(),
        destination.buffer()->page_size());

    const auto& source_shape = source.padded_shape();
    const auto& destination_shape = destination.padded_shape();
    TT_FATAL(source_shape.rank() == destination_shape.rank(), "source and destination ranks differ");
    for (int dim = 1; dim < static_cast<int>(source_shape.rank()); ++dim) {
        TT_FATAL(
            source_shape[dim] == destination_shape[dim],
            "source {} and destination {} differ past dim 0",
            source_shape,
            destination_shape);
    }
    TT_FATAL(
        source_shape[0] == 1 || destination_shape[0] == 1,
        "one of source {} / destination {} must hold a single entry (dim 0 == 1)",
        source_shape,
        destination_shape);

    TT_FATAL(index.storage_type() == ttnn::StorageType::DEVICE, "index must be on device");
    TT_FATAL(
        index.dtype() == tt::tt_metal::DataType::UINT32 || index.dtype() == tt::tt_metal::DataType::INT32,
        "index must be UINT32 or INT32");
    TT_FATAL(index.layout() == tt::tt_metal::Layout::ROW_MAJOR, "index must be ROW_MAJOR");
    TT_FATAL(index.physical_volume() == 1, "index must hold exactly one element");
    TT_FATAL(
        index.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
            index.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "index must be interleaved DRAM");
    TT_FATAL(params.index_scale >= 1, "index_scale must be at least 1");
}

IndexedEntryCopyDeviceOperation::spec_return_value_t IndexedEntryCopyDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& inputs) {
    return inputs.destination.tensor_spec();
}

IndexedEntryCopyDeviceOperation::tensor_return_value_t IndexedEntryCopyDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& inputs) {
    return inputs.destination;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor indexed_entry_copy(
    const Tensor& source, const Tensor& destination, const Tensor& index, uint32_t index_scale, uint32_t index_offset) {
    using OperationType = ttnn::experimental::prim::IndexedEntryCopyDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.index_scale = index_scale, .index_offset = index_offset},
        OperationType::tensor_args_t{.source = source, .destination = destination, .index = index});
}

}  // namespace ttnn::prim
