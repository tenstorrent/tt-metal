// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_pad_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "fill_pad_program_factory.hpp"

namespace ttnn::operations::data_movement::fill_pad {

using namespace tt::tt_metal;

FillPadDeviceOperation::program_factory_t FillPadDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return program::FillPadProgramFactory{};
}

void FillPadDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void FillPadDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    TT_FATAL(input_tensor.layout() == TILE_LAYOUT, "FillPad should only be used for tile layout");
    TT_FATAL(detail::data_type_to_size.contains(input_tensor.dtype()), "Unsupported datatype {}", input_tensor.dtype());
}

TensorSpec FillPadDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    return input_tensor.tensor_spec();
}

Tensor FillPadDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    return input_tensor;
}

}  // namespace ttnn::operations::data_movement::fill_pad

namespace ttnn::prim {
ttnn::Tensor fill_pad(const Tensor& input, float fill_value, const MemoryConfig& output_memory_config) {
    using OperationType = ttnn::operations::data_movement::fill_pad::FillPadDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .fill_value = fill_value,
            .output_mem_config = output_memory_config,
        },
        OperationType::tensor_args_t{.input = input});
}
}  // namespace ttnn::prim
