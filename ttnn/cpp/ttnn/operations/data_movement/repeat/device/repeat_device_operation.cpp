// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/operations/data_movement/repeat/device/repeat_program_factory_last_dim.hpp"
#include "ttnn/operations/data_movement/repeat/device/repeat_program_factory_higher_dim.hpp"
#include "ttnn/operations/data_movement/repeat/device/repeat_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement::repeat {

RepeatDeviceOperation::program_factory_t RepeatDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    bool is_last_dim = operation_attributes.m_is_last_dim;
    if (is_last_dim) {
        return program::RepeatProgramFactoryLastDim{};
    }
    return program::RepeatProgramFactoryHigherDim{};
}

void RepeatDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Validate the input tensor
    const Tensor& input_tensor_a = tensor_args.input;
    TT_FATAL(
        input_tensor_a.storage_type() == tt::tt_metal::StorageType::DEVICE,
        "Operands to reshape need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.layout() == tt::tt_metal::Layout::ROW_MAJOR, "This function is for RM->RM");
    TT_FATAL(
        input_tensor_a.dtype() == tt::tt_metal::DataType::UINT16 or
            input_tensor_a.dtype() == tt::tt_metal::DataType::BFLOAT16 or
            input_tensor_a.dtype() == tt::tt_metal::DataType::UINT32 or
            input_tensor_a.dtype() == tt::tt_metal::DataType::INT32 or
            input_tensor_a.dtype() == tt::tt_metal::DataType::FLOAT32,
        "Can only work with UINT16, BFLOAT16, UINT32, INT32, FLOAT32 data types");
    // is this relevant?
    TT_FATAL(
        operation_attributes.m_output_mem_config.memory_layout() == input_tensor_a.memory_config().memory_layout(),
        "Output tensor must have the same memory layout as input tensor");
}

void RepeatDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

RepeatDeviceOperation::spec_return_value_t RepeatDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& input_tensors) {
    const auto& input_tensor_a = input_tensors.input;
    auto output_shape = input_tensor_a.logical_shape();
    output_shape[operation_attributes.m_is_last_dim ? -1 : 1] *= operation_attributes.m_num_repeats;

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
}  // namespace ttnn::operations::data_movement::repeat

namespace ttnn::prim {
ttnn::operations::data_movement::repeat::RepeatDeviceOperation::tensor_return_value_t repeat(
    const Tensor& input,
    uint32_t m_num_repeats,
    bool m_is_last_dim,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = ttnn::operations::data_movement::repeat::RepeatDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .m_num_repeats = m_num_repeats, .m_is_last_dim = m_is_last_dim, .m_output_mem_config = output_mem_config},
        OperationType::tensor_args_t{.input = input});
}
}  // namespace ttnn::prim
