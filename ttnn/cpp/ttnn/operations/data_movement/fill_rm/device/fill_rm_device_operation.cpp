// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_rm_device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "fill_rm_program_factory.hpp"

namespace ttnn::operations::data_movement::fill_rm {

FillRMDeviceOperation::program_factory_t FillRMDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return program::FillRMProgramFactory{};
}

void FillRMDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void FillRMDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;

    const Tensor& input_tensor_a = tensor_args.input;
    TT_FATAL(
        (args.N > 0 && args.C > 0 && args.H > 0 && args.W > 0),
        "All dimensions must be positive: N={}, C={}, H={}, W={}",
        args.N,
        args.C,
        args.H,
        args.W);
    TT_FATAL(
        (args.hFill <= args.H && args.wFill <= args.W),
        "Fill dimensions must be <= tensor dimensions: hFill={} <= H={}, wFill={} <= W={}",
        args.hFill,
        args.H,
        args.wFill,
        args.W);
    TT_FATAL(
        input_tensor_a.dtype() == DataType::BFLOAT16,
        "Input tensor dtype must be BFLOAT16 but got {}",
        input_tensor_a.dtype());
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "FillRM does not currently support sharding");
    TT_FATAL(
        args.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "FillRM does not currently support sharding");
}

TensorSpec FillRMDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;

    const ttnn::Shape shape({args.N, args.C, args.H, args.W});
    const Tensor& input_tensor = tensor_args.input;
    return TensorSpec(shape, TensorLayout(input_tensor.dtype(), PageConfig(Layout::ROW_MAJOR), args.output_mem_config));
}

Tensor FillRMDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const Tensor& input_tensor = tensor_args.input;
    return create_device_tensor(compute_output_specs(args, tensor_args), input_tensor.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<FillRMDeviceOperation::tensor_return_value_t>
FillRMDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    const tensor_return_value_t& tensor_return_value) {
    using namespace tt::tt_metal;

    const Tensor& input_tensor = tensor_args.input;
    const Tensor& output_tensor = tensor_return_value;
    const int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output_tensor);
    const operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output_tensor}, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttnn::operations::data_movement::fill_rm

namespace ttnn::prim {
ttnn::Tensor fill_rm(
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    uint32_t hFill,
    uint32_t wFill,
    const Tensor& input,
    float val_hi,
    float val_lo,
    const MemoryConfig& output_memory_config) {
    using OperationType = ttnn::operations::data_movement::fill_rm::FillRMDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .N = N,
            .C = C,
            .H = H,
            .W = W,
            .hFill = hFill,
            .wFill = wFill,
            .val_hi = val_hi,
            .val_lo = val_lo,
            .output_mem_config = output_memory_config,
        },
        OperationType::tensor_args_t{.input = input});
}
}  // namespace ttnn::prim
