// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "apply_twiddles_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

namespace {

constexpr bool is_pow2_aw(uint32_t n) {
    return n != 0u && (n & (n - 1u)) == 0u;
}

}  // namespace

ApplyTwiddlesDeviceOperation::program_factory_t
ApplyTwiddlesDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&)
{
    // Only one factory for now; future commits may add a fused-with-pass2
    // variant.
    return ApplyTwiddlesFactory{};
}

void ApplyTwiddlesDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    const auto& in_r = args.input_real;
    const auto& in_i = args.input_imag;

    TT_FATAL(in_r.dtype() == tt::tt_metal::DataType::FLOAT32 ||
             in_r.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "apply_twiddles: only Float32 and BFloat16 inputs are supported (got {}).",
        static_cast<int>(in_r.dtype()));
    TT_FATAL(in_i.dtype() == in_r.dtype(),
        "apply_twiddles: input_real and input_imag must have the same dtype.");
    TT_FATAL(in_r.layout() == tt::tt_metal::Layout::ROW_MAJOR &&
             in_i.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "apply_twiddles: only ROW_MAJOR layout is supported.");
    TT_FATAL(in_r.padded_shape() == in_i.padded_shape(),
        "apply_twiddles: input_real and input_imag must share the same shape.");

    const auto& shape = in_r.padded_shape();
    TT_FATAL(shape.size() >= 1 && shape.size() <= 4,
        "apply_twiddles: input must have 1-4 dimensions (got {}).", shape.size());

    const uint32_t N1 = attrs.N1;
    const uint32_t N2 = attrs.N2;
    TT_FATAL(is_pow2_aw(N1) && N1 >= 2u && N1 <= 1024u,
        "apply_twiddles: N1 must be pow-2 in [2, 1024] (got {}).", N1);
    TT_FATAL(is_pow2_aw(N2) && N2 >= 1u && N2 <= 1024u,
        "apply_twiddles: N2 must be pow-2 in [1, 1024] (got {}).", N2);
    TT_FATAL(static_cast<uint32_t>(shape[-1]) == N1,
        "apply_twiddles: input last dim ({}) must equal N1 ({}).",
        static_cast<uint32_t>(shape[-1]), N1);
    uint32_t M = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d) {
        M *= static_cast<uint32_t>(shape[d]);
    }
    TT_FATAL(M >= 1u && (M % N2) == 0u,
        "apply_twiddles: total row count M={} must be a positive multiple of N2={}.",
        M, N2);
}

ApplyTwiddlesDeviceOperation::spec_return_value_t
ApplyTwiddlesDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& args)
{
    return { args.input_real.tensor_spec(), args.input_real.tensor_spec() };
}

ApplyTwiddlesDeviceOperation::tensor_return_value_t
ApplyTwiddlesDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& args)
{
    using namespace tt::tt_metal;
    auto make_like = [&](const Tensor& ref) -> Tensor {
        return create_device_tensor(ref.tensor_spec(), ref.device());
    };
    return { make_like(args.input_real), make_like(args.input_real) };
}

tt::stl::hash::hash_t ApplyTwiddlesDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    return tt::tt_metal::operation::hash_operation<ApplyTwiddlesDeviceOperation>(
        attrs.N1,
        attrs.N2,
        args.input_real.dtype(),
        args.input_real.memory_config(),
        args.input_real.padded_shape());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::tuple<Tensor, Tensor> apply_twiddles(
    const Tensor& input_real,
    const Tensor& input_imag,
    uint32_t N1,
    uint32_t N2)
{
    using OperationType = ttnn::experimental::prim::ApplyTwiddlesDeviceOperation;

    OperationType::operation_attributes_t attrs{ .N1 = N1, .N2 = N2 };
    OperationType::tensor_args_t          args{
        .input_real = input_real,
        .input_imag = input_imag,
    };

    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim
