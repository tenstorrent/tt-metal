// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <enchantum/enchantum.hpp>

namespace ttnn::operations::data_movement::scatter {

ScatterDeviceOperation::program_factory_t ScatterDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if ((args.opt_reduction != ScatterReductionType::INVALID) &&
        tensor_args.input_tensor.dtype() == DataType::BFLOAT16) {
        return ScatterReduceBfloat16ProgramFactory{};
    }
    return ScatterProgramFactory{};
}

void ScatterDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void ScatterDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor{tensor_args.input_tensor};
    const auto& index_tensor{tensor_args.index_tensor};
    const auto& src_tensor{tensor_args.src_tensor};
    const auto& input_dtype{input_tensor.dtype()};
    const auto& index_dtype{index_tensor.dtype()};
    const auto& src_dtype{src_tensor.dtype()};

    TT_FATAL(
        input_dtype == src_dtype,
        "input_dtype differs from src_dtype (input_dtype: {}, src_dtype: {}).",
        enchantum::to_string(input_dtype),
        enchantum::to_string(src_dtype));

    TT_FATAL(
        index_dtype == DataType::INT32 || index_dtype == DataType::UINT8 || index_dtype == DataType::UINT16 ||
            index_dtype == DataType::UINT32,
        "index_dtype is not integer, it is {}.",
        enchantum::to_string(index_dtype));

    TT_FATAL(!input_tensor.is_sharded(), "Sharded tensors are not supported - input_tensor is sharded.");
    TT_FATAL(!index_tensor.is_sharded(), "Sharded tensors are not supported - index_tensor is sharded.");
    TT_FATAL(!src_tensor.is_sharded(), "Sharded tensors are not supported - src_tensor is sharded.");

    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor's buffer is null.");
    TT_FATAL(index_tensor.buffer() != nullptr, "Index tensor's buffer is null.");
    TT_FATAL(src_tensor.buffer() != nullptr, "Src tensor's buffer is null.");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be allocated on a device.");
    TT_FATAL(index_tensor.storage_type() == StorageType::DEVICE, "Index tensor must be allocated on a device.");
    TT_FATAL(src_tensor.storage_type() == StorageType::DEVICE, "Src tensor must be allocated on a device.");
}

ScatterDeviceOperation::spec_return_value_t ScatterDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;
    return TensorSpec{
        tensor_args.input_tensor.logical_shape(),
        TensorLayout{tensor_args.input_tensor.dtype(), PageConfig{Layout::ROW_MAJOR}, args.output_memory_config}};
}

ScatterDeviceOperation::tensor_return_value_t ScatterDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<ScatterDeviceOperation::tensor_return_value_t>
ScatterDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*op_attr*/, const tensor_args_t& inputs, const Tensor& output) {
    const auto& input_tensor = inputs.input_tensor;
    int ideal_dev_clock_cycles = data_movement::common_tm_bw_model(input_tensor, output);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttnn::operations::data_movement::scatter

namespace ttnn::prim {
ttnn::Tensor scatter(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const MemoryConfig& output_memory_config,
    const operations::data_movement::scatter::ScatterReductionType& reduction,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    using OperationType = ttnn::operations::data_movement::scatter::ScatterDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{dim, output_memory_config, reduction, sub_core_grid},
        OperationType::tensor_args_t{input_tensor, index_tensor, source_tensor});
}
}  // namespace ttnn::prim
