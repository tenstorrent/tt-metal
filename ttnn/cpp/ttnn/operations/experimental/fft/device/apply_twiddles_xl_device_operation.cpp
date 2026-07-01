// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "apply_twiddles_xl_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

namespace {

constexpr bool is_pow2_xl_op(uint32_t n) {
    return n != 0u && (n & (n - 1u)) == 0u;
}

// Hard upper cap on big_modulus.  At 2^20 the host-side delta table is
// 2 * 2^20 * 4 = 8 MB, which is plenty for N up to 2^30 with cube-
// balanced 3-pass (big_modulus = N1*N2 = 2^20).  Above this we'd need a
// different scheme (e.g. on-device sin/cos), which is out of scope for
// commits 5a/5b.
constexpr uint32_t kBigModulusCap = 1u << 20;

}  // namespace

ApplyTwiddlesXlDeviceOperation::program_factory_t
ApplyTwiddlesXlDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&)
{
    return ApplyTwiddlesXlFactory{};
}

void ApplyTwiddlesXlDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    const auto& in_r = args.input_real;
    const auto& in_i = args.input_imag;

    TT_FATAL(in_r.dtype() == tt::tt_metal::DataType::FLOAT32 ||
             in_r.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "apply_twiddles_xl: only Float32 and BFloat16 inputs are supported (got {}).",
        static_cast<int>(in_r.dtype()));
    TT_FATAL(in_i.dtype() == in_r.dtype(),
        "apply_twiddles_xl: input_real and input_imag must have the same dtype.");
    TT_FATAL(in_r.layout() == tt::tt_metal::Layout::ROW_MAJOR &&
             in_i.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "apply_twiddles_xl: only ROW_MAJOR layout is supported.");
    TT_FATAL(in_r.padded_shape() == in_i.padded_shape(),
        "apply_twiddles_xl: input_real and input_imag must share the same shape.");

    const auto& shape = in_r.padded_shape();
    TT_FATAL(shape.size() >= 1 && shape.size() <= 4,
        "apply_twiddles_xl: input must have 1-4 dimensions (got {}).", shape.size());

    const uint32_t P           = attrs.P;
    const uint32_t big_modulus = attrs.big_modulus;
    const uint32_t full_N      = attrs.full_N;

    TT_FATAL(is_pow2_xl_op(P) && P >= 2u && P <= 1024u,
        "apply_twiddles_xl: P must be pow-2 in [2, 1024] (got {}).", P);
    TT_FATAL(is_pow2_xl_op(big_modulus) && big_modulus >= 1u &&
             big_modulus <= kBigModulusCap,
        "apply_twiddles_xl: big_modulus must be pow-2 in [1, {}] (got {}).",
        kBigModulusCap, big_modulus);
    TT_FATAL(is_pow2_xl_op(full_N) && full_N >= big_modulus,
        "apply_twiddles_xl: full_N must be pow-2 and >= big_modulus "
        "(got full_N={} big_modulus={}).", full_N, big_modulus);

    TT_FATAL(static_cast<uint32_t>(shape[-1]) == P,
        "apply_twiddles_xl: input last dim ({}) must equal P ({}).",
        static_cast<uint32_t>(shape[-1]), P);
    uint32_t M = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d) {
        M *= static_cast<uint32_t>(shape[d]);
    }
    TT_FATAL(M >= 1u && (M % big_modulus) == 0u,
        "apply_twiddles_xl: total row count M={} must be a positive "
        "multiple of big_modulus={}.", M, big_modulus);
}

ApplyTwiddlesXlDeviceOperation::spec_return_value_t
ApplyTwiddlesXlDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& args)
{
    return { args.input_real.tensor_spec(), args.input_real.tensor_spec() };
}

ApplyTwiddlesXlDeviceOperation::tensor_return_value_t
ApplyTwiddlesXlDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& args)
{
    using namespace tt::tt_metal;
    auto make_like = [&](const Tensor& ref) -> Tensor {
        return create_device_tensor(ref.tensor_spec(), ref.device());
    };
    return { make_like(args.input_real), make_like(args.input_real) };
}

tt::stl::hash::hash_t ApplyTwiddlesXlDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    return tt::tt_metal::operation::hash_operation<ApplyTwiddlesXlDeviceOperation>(
        attrs.P,
        attrs.big_modulus,
        attrs.full_N,
        args.input_real.dtype(),
        args.input_real.memory_config(),
        args.input_real.padded_shape());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::tuple<Tensor, Tensor> apply_twiddles_xl(
    const Tensor& input_real,
    const Tensor& input_imag,
    uint32_t P,
    uint32_t big_modulus,
    uint32_t full_N)
{
    using OperationType = ttnn::experimental::prim::ApplyTwiddlesXlDeviceOperation;

    OperationType::operation_attributes_t attrs{
        .P           = P,
        .big_modulus = big_modulus,
        .full_N      = full_N,
    };
    OperationType::tensor_args_t args{
        .input_real = input_real,
        .input_imag = input_imag,
    };

    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim
