// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise: Implement the device operation methods

#include "matmul_add_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::onboarding::exercise {

MatmulAddOperation::program_factory_t MatmulAddOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ProgramFactory{};
}

void MatmulAddOperation::validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {
    // TODO: Implement validation
}

void MatmulAddOperation::validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}

MatmulAddOperation::spec_return_value_t MatmulAddOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    (void)tensor_args;
    // TODO: Implement
    TT_THROW("Not implemented");
}

MatmulAddOperation::tensor_return_value_t MatmulAddOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.a.device());
}

}  // namespace ttnn::operations::onboarding::exercise

namespace ttnn::prim {

ttnn::Tensor exercise_matmul_add(const ttnn::Tensor& a, const ttnn::Tensor& b, const ttnn::Tensor& c) {
    using OperationType = ttnn::operations::onboarding::exercise::MatmulAddOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{}, OperationType::tensor_args_t{a, b, c});
}

}  // namespace ttnn::prim
