// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "move_codegen_device_operation.hpp"
#include "move_codegen_supported.hpp"
#include "ttnn/device_operation.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::prim {

namespace {

using namespace tt::tt_metal;

constexpr uint32_t kReadBatch = 4;
constexpr uint32_t kWriteBatch = 4;
constexpr uint32_t kMinCbDepth = 8;
// Reserve 128KB of L1 for code/stacks, matching ops/identity/spec.py's _resolve_batches /
// _resolve_rm exactly (byte-identical host math, minus the TTDM_OVERRIDE_* sweep-tuning env hooks
// — those exist only to let the codegen harness's autotuner override batch depth for benchmarking
// and have no place in a production prim).
constexpr uint32_t kL1ReservedBytes = 128 * 1024;

uint32_t align_up(uint32_t size, uint32_t alignment) { return ((size + alignment - 1) / alignment) * alignment; }

// ops/identity/spec.py::_resolve_batches + _identity_tile_plan.
MoveCodegenOperationAttributes compute_tile_attributes(
    const Tensor& input_tensor, const tt::tt_metal::MemoryConfig& output_mem_config) {
    uint32_t read_batch = kReadBatch;
    uint32_t write_batch = kWriteBatch;

    const tt::DataFormat data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t raw_tile_bytes = tt::tile_size(data_format);
    const uint32_t page_bytes = align_up(raw_tile_bytes, hal::get_dram_alignment());

    const uint32_t available_l1 = hal::get_l1_size() - kL1ReservedBytes;
    const uint32_t max_tiles = available_l1 / raw_tile_bytes;

    uint32_t cb_depth = std::max(2 * std::max(read_batch, write_batch), kMinCbDepth);
    if (cb_depth > max_tiles) {
        cb_depth = std::max(kMinCbDepth, max_tiles);
        const uint32_t max_batch = cb_depth / 2;
        read_batch = std::min(read_batch, max_batch);
        write_batch = std::min(write_batch, max_batch);
    }
    cb_depth = std::max(2 * std::max(read_batch, write_batch), kMinCbDepth);

    const uint32_t total_pages = static_cast<uint32_t>(
        input_tensor.physical_volume() / (tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH));

    return MoveCodegenOperationAttributes{
        .total_pages = total_pages,
        .page_bytes = page_bytes,
        .read_batch = read_batch,
        .write_batch = write_batch,
        .cb_depth = cb_depth,
        .output_mem_config = output_mem_config,
    };
}

// ops/identity/spec.py::_resolve_rm + _identity_rm_plan.
MoveCodegenOperationAttributes compute_rm_attributes(
    const Tensor& input_tensor, const tt::tt_metal::MemoryConfig& output_mem_config) {
    uint32_t read_batch = kReadBatch;
    uint32_t write_batch = kWriteBatch;

    const uint32_t width = input_tensor.padded_shape()[-1];
    const uint32_t stick_bytes = width * input_tensor.element_size();

    const uint32_t available_l1 = hal::get_l1_size() - kL1ReservedBytes;
    uint32_t cb_depth = std::max(2 * std::max(read_batch, write_batch), kMinCbDepth);
    if (cb_depth * stick_bytes > available_l1) {
        cb_depth = std::max(kMinCbDepth, available_l1 / stick_bytes);
        const uint32_t max_batch = cb_depth / 2;
        read_batch = std::min(read_batch, max_batch);
        write_batch = std::min(write_batch, max_batch);
    }
    cb_depth = std::max(2 * std::max(read_batch, write_batch), kMinCbDepth);

    const uint32_t alignment = input_tensor.memory_config().buffer_type() == BufferType::L1
                                    ? hal::get_l1_alignment()
                                    : hal::get_dram_alignment();
    const uint32_t page_bytes = align_up(stick_bytes, alignment);

    const uint32_t total_pages = static_cast<uint32_t>(input_tensor.physical_volume() / width);

    return MoveCodegenOperationAttributes{
        .total_pages = total_pages,
        .page_bytes = page_bytes,
        .read_batch = read_batch,
        .write_batch = write_batch,
        .cb_depth = cb_depth,
        .output_mem_config = output_mem_config,
    };
}

MoveCodegenOperationAttributes compute_operation_attributes(
    const Tensor& input_tensor, const tt::tt_metal::MemoryConfig& output_mem_config) {
    if (input_tensor.layout() == Layout::TILE) {
        return compute_tile_attributes(input_tensor, output_mem_config);
    }
    TT_FATAL(
        input_tensor.layout() == Layout::ROW_MAJOR,
        "ttnn.move (codegen): unsupported layout (supported_by_codegen() should have rejected this)");
    return compute_rm_attributes(input_tensor, output_mem_config);
}

}  // namespace

MoveCodegenDeviceOperation::program_factory_t MoveCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return MoveCodegenProgramFactory{};
}

void MoveCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    TT_FATAL(
        ttnn::operations::data_movement::move_codegen::supported_by_codegen(
            tensor_args.input_tensor, operation_attributes.output_mem_config),
        "ttnn.move: implementation=\"codegen\" is not supported for this call");
}

void MoveCodegenDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const Tensor& output_tensor = tensor_args.output_tensor;

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "ttnn.move: input tensor must be on device on cache hit. Got storage type: {}",
        input_tensor.storage_type());

    TT_FATAL(input_tensor.buffer() != nullptr, "ttnn.move: input tensor buffer must be allocated on cache hit");

    TT_FATAL(
        output_tensor.storage_type() == StorageType::DEVICE,
        "ttnn.move: output tensor must be on device on cache hit. Got storage type: {}",
        output_tensor.storage_type());

    TT_FATAL(output_tensor.buffer() != nullptr, "ttnn.move: output tensor buffer must be allocated on cache hit");
}

MoveCodegenDeviceOperation::spec_return_value_t MoveCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    return tensor_args.output_tensor.tensor_spec();
}

MoveCodegenDeviceOperation::tensor_return_value_t MoveCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    return tensor_args.output_tensor;
}

}  // namespace ttnn::prim

namespace ttnn::prim {
ttnn::prim::MoveCodegenDeviceOperation::tensor_return_value_t move_codegen(
    const Tensor& input_tensor, const Tensor& output_tensor, const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = ttnn::prim::MoveCodegenDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        compute_operation_attributes(input_tensor, output_mem_config),
        OperationType::tensor_args_t{.input_tensor = input_tensor, .output_tensor = output_tensor});
}
}  // namespace ttnn::prim
