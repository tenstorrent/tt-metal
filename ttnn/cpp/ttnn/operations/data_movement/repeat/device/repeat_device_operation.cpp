// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/operations/data_movement/repeat/device/repeat_program_factory_last_dim.hpp"
#include "ttnn/operations/data_movement/repeat/device/repeat_program_factory_higher_dim.hpp"
#include "ttnn/operations/data_movement/repeat/device/repeat_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

RepeatDeviceOperation::program_factory_t RepeatDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    if (operation_attributes.m_tile_page_size_bytes > 0) {
        return RepeatProgramFactoryHigherDim{};
    }
    if (operation_attributes.m_is_last_dim) {
        return RepeatProgramFactoryLastDim{};
    }
    return RepeatProgramFactoryHigherDim{};
}

void RepeatDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& input_tensor_a = tensor_args.input;
    TT_FATAL(
        input_tensor_a.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operands to repeat need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands need to be allocated in buffers on device!");

    if (operation_attributes.m_tile_page_size_bytes > 0) {
        TT_FATAL(input_tensor_a.layout() == tt::tt_metal::Layout::TILE, "Tile-native repeat requires TILE layout");
    } else {
        TT_FATAL(
            input_tensor_a.layout() == tt::tt_metal::Layout::ROW_MAJOR, "ROW_MAJOR repeat requires ROW_MAJOR layout");
        TT_FATAL(
            input_tensor_a.dtype() == tt::tt_metal::DataType::UINT16 or
                input_tensor_a.dtype() == tt::tt_metal::DataType::BFLOAT16 or
                input_tensor_a.dtype() == tt::tt_metal::DataType::UINT32 or
                input_tensor_a.dtype() == tt::tt_metal::DataType::INT32 or
                input_tensor_a.dtype() == tt::tt_metal::DataType::FLOAT32,
            "Can only work with UINT16, BFLOAT16, UINT32, INT32, FLOAT32 data types");
    }

    TT_FATAL(
        operation_attributes.m_output_mem_config.memory_layout() == input_tensor_a.memory_config().memory_layout(),
        "Output tensor must have the same memory layout as input tensor");
}

RepeatDeviceOperation::spec_return_value_t RepeatDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& input_tensors) {
    const auto& input_tensor_a = input_tensors.input;
    auto output_shape = input_tensor_a.logical_shape();
    if (operation_attributes.m_repeat_dim >= 0) {
        output_shape[operation_attributes.m_repeat_dim] *= operation_attributes.m_num_repeats;
    } else {
        output_shape[operation_attributes.m_is_last_dim ? -1 : 1] *= operation_attributes.m_num_repeats;
    }

    auto mem_config = operation_attributes.m_output_mem_config;
    if (input_tensor_a.memory_config().is_sharded()) {
        auto shard_spec = input_tensor_a.shard_spec().value();
        shard_spec.shape[0] = output_shape[0];
        mem_config = mem_config.with_shard_spec(shard_spec);
    }
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            input_tensor_a.dtype(), tt::tt_metal::PageConfig(input_tensor_a.layout()), mem_config));
}

RepeatDeviceOperation::tensor_return_value_t RepeatDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& input_tensors) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, input_tensors), input_tensors.input.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> RepeatDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input_tensor = tensor_args.input;
    int ideal_dev_clock_cycles = operations::data_movement::common_tm_bw_model(input_tensor, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, output_tensor, ideal_dev_clock_cycles);
    return result;
}

RepeatDeviceOperation::tensor_return_value_t repeat(
    const Tensor& input,
    uint32_t m_num_repeats,
    bool m_is_last_dim,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = RepeatDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .m_num_repeats = m_num_repeats, .m_is_last_dim = m_is_last_dim, .m_output_mem_config = output_mem_config},
        OperationType::tensor_args_t{.input = input});
}

RepeatDeviceOperation::tensor_return_value_t repeat_tile(
    const Tensor& input,
    uint32_t num_repeats,
    int32_t repeat_dim,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    uint32_t tile_higher_pages,
    uint32_t tile_rep_dim_pages,
    uint32_t tile_lower_pages,
    uint32_t tile_page_size_bytes) {
    using OperationType = RepeatDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .m_num_repeats = num_repeats,
            .m_is_last_dim = false,
            .m_output_mem_config = output_mem_config,
            .m_tile_higher_pages = tile_higher_pages,
            .m_tile_rep_dim_pages = tile_rep_dim_pages,
            .m_tile_lower_pages = tile_lower_pages,
            .m_tile_page_size_bytes = tile_page_size_bytes,
            .m_repeat_dim = repeat_dim},
        OperationType::tensor_args_t{.input = input});
}
}  // namespace ttnn::prim
