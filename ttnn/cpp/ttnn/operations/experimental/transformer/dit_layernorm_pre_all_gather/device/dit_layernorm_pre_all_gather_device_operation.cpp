// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_layernorm_pre_all_gather_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

PreAllGatherDeviceOperation::program_factory_t PreAllGatherDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return PreAllGatherWelfordProgramFactory{};
}

void PreAllGatherDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void PreAllGatherDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& tensor = tensor_args;

    TT_FATAL(!tensor.is_sharded(), "DIT layernorm pre-all-gather does not support sharded inputs.");
    TT_FATAL(tensor.layout() == Layout::TILE, "Only tilized inputs supported.");
    TT_FATAL(
        tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only interleaved inputs supported.");
    TT_FATAL(
        tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::BFLOAT8_B ||
            tensor.dtype() == DataType::FLOAT32,
        "Input data format not supported.");
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Operands must be on device.");
    TT_FATAL(tensor.buffer() != nullptr, "Operands must be allocated on device.");
}

PreAllGatherDeviceOperation::spec_return_value_t PreAllGatherDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args;

    auto output_shape = input_tensor.logical_shape();
    output_shape[3] = 2 * TILE_WIDTH;  // two tile columns: sum(x) and sum(x^2)

    auto output_dtype = args.dtype.value_or(input_tensor.dtype());
    return TensorSpec(output_shape, TensorLayout(output_dtype, PageConfig(Layout::TILE), args.memory_config));
}

PreAllGatherDeviceOperation::tensor_return_value_t PreAllGatherDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor dit_layernorm_pre_all_gather(
    const Tensor& input,
    const std::optional<tt::tt_metal::DataType>& dtype,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const tt::tt_metal::MemoryConfig& memory_config) {
    using OperationType = ttnn::experimental::prim::PreAllGatherDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .dtype = dtype,
            .compute_kernel_config = compute_kernel_config,
            .memory_config = memory_config,
        },
        input);
}

}  // namespace ttnn::prim
