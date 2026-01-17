// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_program_factory.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::indexed_fill {

IndexedFillDeviceOperation::program_factory_t IndexedFillDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return program::IndexedFillProgramFactory{};
}

void IndexedFillDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void IndexedFillDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& batch_ids = tensor_args.batch_id;
    auto input_tensor_a_shape = input_tensor_a.padded_shape();
    auto input_tensor_b_shape = input_tensor_b.padded_shape();
    TT_FATAL(input_tensor_a.layout() == Layout::ROW_MAJOR, "Currently only supporting row major layout");
    TT_FATAL(input_tensor_b.layout() == input_tensor_a.layout(), "Inputs must be same layout");
    TT_FATAL(
        input_tensor_a_shape[1] == input_tensor_b_shape[1] && input_tensor_a_shape[2] == input_tensor_b_shape[2] &&
            input_tensor_a_shape[3] == input_tensor_b_shape[3],
        "Dims except batch dim must be the same on inputs");
    TT_FATAL(
        input_tensor_b_shape[0] == batch_ids.padded_shape()[-1], "Second input and batch ids must be same outer dim");
    TT_FATAL(batch_ids.layout() == Layout::ROW_MAJOR, "Batch IDs must be ROW MAJOR");
    TT_FATAL(args.dim == 0, "Currently only supporting batch dimension");
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to Index Fill need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to Index Fill need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input tensor memory layout must be INTERLEAVED but got {}",
        input_tensor_a.memory_config().memory_layout());
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Index Fill does not currently support sharding");
}

IndexedFillDeviceOperation::spec_return_value_t IndexedFillDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor_a;
    return TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout(input_tensor.dtype(), PageConfig(input_tensor.layout()), args.output_mem_config));
}

IndexedFillDeviceOperation::tensor_return_value_t IndexedFillDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor_a.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<IndexedFillDeviceOperation::tensor_return_value_t>
IndexedFillDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input_tensor = tensor_args.batch_id;
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
    return result;
}
}  // namespace ttnn::operations::data_movement::indexed_fill

namespace ttnn::prim {
ttnn::operations::data_movement::indexed_fill::IndexedFillDeviceOperation::tensor_return_value_t indexed_fill(
    const Tensor& batch_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    int64_t dim) {
    using OperationType = ttnn::operations::data_movement::indexed_fill::IndexedFillDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_mem_config = output_mem_config,
            .dim = dim,
        },
        OperationType::tensor_args_t{
            .batch_id = batch_id,
            .input_tensor_a = input_tensor_a,
            .input_tensor_b = input_tensor_b,
        });
}
}  // namespace ttnn::prim
