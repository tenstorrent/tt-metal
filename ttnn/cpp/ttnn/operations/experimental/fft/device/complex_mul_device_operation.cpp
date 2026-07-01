// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "complex_mul_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

ComplexMulDeviceOperation::program_factory_t
ComplexMulDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&)
{
    return ComplexMulFactory{};
}

void ComplexMulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& args)
{
    const auto& a_r = args.a_real;
    const auto& a_i = args.a_imag;
    const auto& b_r = args.b_real;
    const auto& b_i = args.b_imag;

    TT_FATAL(a_r.dtype() == tt::tt_metal::DataType::FLOAT32 ||
             a_r.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "complex_mul: only Float32 and BFloat16 inputs are supported (got {}).",
        static_cast<int>(a_r.dtype()));
    TT_FATAL(a_i.dtype() == a_r.dtype() &&
             b_r.dtype() == a_r.dtype() &&
             b_i.dtype() == a_r.dtype(),
        "complex_mul: all four input tensors must share the same dtype.");
    TT_FATAL(a_r.layout() == tt::tt_metal::Layout::ROW_MAJOR &&
             a_i.layout() == tt::tt_metal::Layout::ROW_MAJOR &&
             b_r.layout() == tt::tt_metal::Layout::ROW_MAJOR &&
             b_i.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "complex_mul: only ROW_MAJOR layout is supported.");
    TT_FATAL(a_r.padded_shape() == a_i.padded_shape() &&
             a_r.padded_shape() == b_r.padded_shape() &&
             a_r.padded_shape() == b_i.padded_shape(),
        "complex_mul: all four input tensors must share the same shape.");

    const auto& shape = a_r.padded_shape();
    TT_FATAL(shape.size() >= 1 && shape.size() <= 4,
        "complex_mul: input must have 1-4 dimensions (got {}).", shape.size());
    const uint32_t P = static_cast<uint32_t>(shape[-1]);
    TT_FATAL(P >= 1u && P <= 1024u,
        "complex_mul: last-dim row length must be in [1, 1024] (got {}).", P);
}

ComplexMulDeviceOperation::spec_return_value_t
ComplexMulDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& args)
{
    return { args.a_real.tensor_spec(), args.a_real.tensor_spec() };
}

ComplexMulDeviceOperation::tensor_return_value_t
ComplexMulDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& args)
{
    using namespace tt::tt_metal;
    auto make_like = [&](const Tensor& ref) -> Tensor {
        return create_device_tensor(ref.tensor_spec(), ref.device());
    };
    return { make_like(args.a_real), make_like(args.a_real) };
}

tt::stl::hash::hash_t ComplexMulDeviceOperation::compute_program_hash(
    const operation_attributes_t&, const tensor_args_t& args)
{
    // No kernel-affecting attributes — the program identity is purely
    // a function of the input dtype + shape + memory config.  All four
    // tensors share the same spec (validated above) so hashing on
    // a_real alone is sufficient.
    return tt::tt_metal::operation::hash_operation<ComplexMulDeviceOperation>(
        args.a_real.dtype(),
        args.a_real.memory_config(),
        args.a_real.padded_shape());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::tuple<Tensor, Tensor> complex_mul(
    const Tensor& a_real,
    const Tensor& a_imag,
    const Tensor& b_real,
    const Tensor& b_imag)
{
    using OperationType = ttnn::experimental::prim::ComplexMulDeviceOperation;

    OperationType::operation_attributes_t attrs{};
    OperationType::tensor_args_t args{
        .a_real = a_real,
        .a_imag = a_imag,
        .b_real = b_real,
        .b_imag = b_imag,
    };

    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim
