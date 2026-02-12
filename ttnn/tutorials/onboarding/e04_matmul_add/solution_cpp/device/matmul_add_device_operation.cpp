// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_add_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::onboarding {

MatmulAddOperation::program_factory_t MatmulAddOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ProgramFactory{};
}

void MatmulAddOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // a: (M, K), b: (K, N), c: (M, N)
    auto a_shape = tensor_args.a.logical_shape();
    auto b_shape = tensor_args.b.logical_shape();
    auto c_shape = tensor_args.c.logical_shape();

    TT_FATAL(a_shape[-1] == b_shape[-2], "Inner dimensions must match for matmul");
    TT_FATAL(a_shape[-2] == c_shape[-2] && b_shape[-1] == c_shape[-1], "Bias shape must match output");
    TT_FATAL(tensor_args.a.layout() == Layout::TILE, "Input must be in TILE layout");
}

void MatmulAddOperation::validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}

MatmulAddOperation::spec_return_value_t MatmulAddOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    auto a_shape = tensor_args.a.logical_shape();
    auto b_shape = tensor_args.b.logical_shape();
    // Output: (M, N)
    ttnn::SmallVector<uint32_t> out_shape = {a_shape[-2], b_shape[-1]};

    return TensorSpec(
        Shape(out_shape),
        tt::tt_metal::TensorLayout(
            tensor_args.a.dtype(), tt::tt_metal::PageConfig(tensor_args.a.layout()), MemoryConfig{}));
}

MatmulAddOperation::tensor_return_value_t MatmulAddOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.a.device());
}

}  // namespace ttnn::operations::onboarding

namespace ttnn::prim {

ttnn::Tensor onboarding_matmul_add(const ttnn::Tensor& a, const ttnn::Tensor& b, const ttnn::Tensor& c) {
    using OperationType = ttnn::operations::onboarding::MatmulAddOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{}, OperationType::tensor_args_t{a, b, c});
}

}  // namespace ttnn::prim
