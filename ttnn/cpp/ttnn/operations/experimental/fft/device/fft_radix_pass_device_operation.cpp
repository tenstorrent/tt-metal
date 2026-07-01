// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "fft_radix_pass_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

namespace {

constexpr bool is_pow2_op(uint32_t n) {
    return n != 0u && (n & (n - 1u)) == 0u;
}

}  // namespace

FftRadixPassDeviceOperation::program_factory_t
FftRadixPassDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&)
{
    return FftRadixPassFactory{};
}

void FftRadixPassDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    const auto& in = args.input_real;

    TT_FATAL(in.dtype() == tt::tt_metal::DataType::FLOAT32 ||
             in.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "fft_radix_pass: only Float32 and BFloat16 inputs supported (got {}).",
        static_cast<int>(in.dtype()));
    TT_FATAL(in.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "fft_radix_pass: only ROW_MAJOR layout supported.");

    const auto& shape = in.padded_shape();
    TT_FATAL(shape.size() >= 1 && shape.size() <= 4,
        "fft_radix_pass: input must have 1-4 dimensions (got {}).", shape.size());

    const uint32_t N = static_cast<uint32_t>(shape[-1]);
    TT_FATAL(N == attrs.P,
        "fft_radix_pass: input last dim ({}) must equal params.P ({}).", N, attrs.P);
    TT_FATAL(is_pow2_op(N) && N >= 2u && N <= 1024u,
        "fft_radix_pass: P must be pow-2 in [2, 1024] (got {}).", N);

    uint32_t M = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d) {
        M *= static_cast<uint32_t>(shape[d]);
    }
    TT_FATAL(is_pow2_op(M) && M >= 1u,
        "fft_radix_pass: total batch (product of leading dims) must be "
        "pow-2 and >=1 (got {}).", M);

    if (attrs.twiddle_N2 != 0u) {
        TT_FATAL(is_pow2_op(attrs.twiddle_N2) &&
                 attrs.twiddle_N2 >= 1u &&
                 attrs.twiddle_N2 <= 1024u,
            "fft_radix_pass: twiddle_N2 must be pow-2 in [1, 1024] (got {}).",
            attrs.twiddle_N2);
        TT_FATAL(is_pow2_op(attrs.stride) && attrs.stride >= 1u && attrs.stride <= M,
            "fft_radix_pass: stride must be pow-2 in [1, M={}] (got {}).",
            M, attrs.stride);
        TT_FATAL((M % attrs.stride) == 0u,
            "fft_radix_pass: total batch {} must be a multiple of stride {}.",
            M, attrs.stride);
        TT_FATAL(((M / attrs.stride) % attrs.twiddle_N2) == 0u,
            "fft_radix_pass: (M={} / stride={}) must be a multiple of "
            "twiddle_N2 {}.", M, attrs.stride, attrs.twiddle_N2);
    }

    if (args.input_imag.has_value()) {
        const auto& imag = *args.input_imag;
        TT_FATAL(imag.dtype()        == in.dtype()        &&
                 imag.layout()       == in.layout()       &&
                 imag.padded_shape() == shape,
            "fft_radix_pass: input_imag must match input_real in "
            "dtype/layout/shape.");
    }
}

FftRadixPassDeviceOperation::spec_return_value_t
FftRadixPassDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& args)
{
    return { args.input_real.tensor_spec(), args.input_real.tensor_spec() };
}

FftRadixPassDeviceOperation::tensor_return_value_t
FftRadixPassDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& args)
{
    using namespace tt::tt_metal;
    auto make_like = [&](const Tensor& ref) -> Tensor {
        return create_device_tensor(ref.tensor_spec(), ref.device());
    };
    return { make_like(args.input_real), make_like(args.input_real) };
}

tt::stl::hash::hash_t FftRadixPassDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    // Include `has_imag` so a real vs complex call to the same shape
    // doesn't alias program cache entries (the kernel ABI is identical
    // but the imag-buffer wire path differs — caller buffer page-size
    // vs cached zero-scratch tile size).
    //
    // For output_scale we hash only the BOOLEAN "is scale enabled" — the
    // actual float scale value flows through a runtime arg so all
    // non-unity scales share one kernel binary / program cache entry.
    const bool apply_scale = (attrs.output_scale != 1.0f);
    return tt::tt_metal::operation::hash_operation<FftRadixPassDeviceOperation>(
        attrs.P,
        attrs.twiddle_N2,
        attrs.stride,
        args.input_real.dtype(),
        args.input_real.memory_config(),
        args.input_real.padded_shape(),
        args.input_imag.has_value(),
        apply_scale);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::tuple<Tensor, Tensor> fft_radix_pass(
    const Tensor& input_real,
    const std::optional<Tensor>& input_imag,
    uint32_t P,
    uint32_t twiddle_N2,
    uint32_t stride,
    float output_scale)
{
    using OperationType = ttnn::experimental::prim::FftRadixPassDeviceOperation;

    OperationType::operation_attributes_t attrs{
        .P            = P,
        .twiddle_N2   = twiddle_N2,
        .stride       = stride,
        .output_scale = output_scale,
    };
    OperationType::tensor_args_t args{
        .input_real = input_real,
        .input_imag = input_imag,
    };

    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim
