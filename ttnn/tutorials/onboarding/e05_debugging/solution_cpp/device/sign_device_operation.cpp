// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sign_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::constants;

namespace ttnn::operations::onboarding {

SignOperation::program_factory_t SignOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ProgramFactory{};
}

void SignOperation::validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input must be on device");
    TT_FATAL(input.layout() == Layout::TILE, "Input must be in tile layout");
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "Input must be bfloat16");
}

void SignOperation::validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}

SignOperation::spec_return_value_t SignOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return TensorSpec(
        tensor_args.input.logical_shape(),
        tt::tt_metal::TensorLayout(
            tensor_args.input.dtype(), tt::tt_metal::PageConfig(tensor_args.input.layout()), MemoryConfig{}));
}

SignOperation::tensor_return_value_t SignOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::onboarding

namespace ttnn::prim {

ttnn::Tensor onboarding_sign(const ttnn::Tensor& input) {
    using OperationType = ttnn::operations::onboarding::SignOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{}, OperationType::tensor_args_t{input});
}

}  // namespace ttnn::prim
