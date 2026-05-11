// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fft_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

namespace {

// Compile-time check: is N a power of two?
constexpr bool is_pow2(uint32_t n) {
    return n != 0u && (n & (n - 1u)) == 0u;
}

// Pick the right backend for (dtype, N). Mirrors the dispatch table in
// fft.hpp's documentation block.
FFTBackend select_backend(tt::tt_metal::DataType dtype, uint32_t N) {
    if (dtype == tt::tt_metal::DataType::BFLOAT16) return FFTBackend::UniversalBf16;

    // dtype == Float32
    if (!is_pow2(N))             return FFTBackend::Universal;
    if (N <= 1u * 1024u * 1024u) return FFTBackend::Stockham;
    if (N <= 16u * 1024u * 1024u) return FFTBackend::UniversalXL;

    // Above 16M is gated even in fft_universal_xl until packed batch_fft_xl
    // ships; we surface it here as an unsupported-backend error.
    TT_THROW(
        "ttnn::experimental::fft does not yet support N={} for Float32 "
        "(K=4 dispatcher / packed batch_fft_xl kernel not yet shipped).",
        N);
}

}  // namespace

void FFTDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto& input = args.input_real;

    TT_FATAL(
        input.dtype() == tt::tt_metal::DataType::FLOAT32 ||
        input.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "fft: only Float32 and BFloat16 inputs are supported (got {}).",
        static_cast<int>(input.dtype()));

    TT_FATAL(
        input.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "fft: only ROW_MAJOR layout is supported (got {}).",
        static_cast<int>(input.layout()));

    const auto& shape = input.padded_shape();
    TT_FATAL(shape.size() >= 1 && shape.size() <= 4,
             "fft: input must have 1-4 dimensions (got {}).", shape.size());

    const uint32_t N = shape[-1];
    TT_FATAL(N >= 2u, "fft: FFT length N must be >= 2 (got {}).", N);

    if (attrs.inverse) {
        TT_FATAL(args.input_imag.has_value(),
                 "fft (inverse): both real and imag spectrum tensors required.");
        TT_FATAL(args.input_imag->dtype()  == input.dtype()  &&
                 args.input_imag->layout() == input.layout() &&
                 args.input_imag->padded_shape() == shape,
                 "fft (inverse): spectrum_real and spectrum_imag must match "
                 "in dtype/layout/shape.");
    }

    // Throws a clear TT_THROW for any unsupported (dtype, N) combo
    // (e.g. fp32 + pow2 + N > 16M).
    (void)select_backend(input.dtype(), N);
}

FFTDeviceOperation::spec_return_value_t FFTDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& args) {
    // Output spectrum has the same shape/dtype/layout as the real input.
    return { args.input_real.tensor_spec(), args.input_real.tensor_spec() };
}

FFTDeviceOperation::tensor_return_value_t FFTDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& args) {
    using namespace tt::tt_metal;

    auto make_like = [&](const Tensor& ref) -> Tensor {
        return create_device_tensor(
            ref.tensor_spec(),
            ref.device());
    };

    return { make_like(args.input_real), make_like(args.input_real) };
}

tt::stl::hash::hash_t FFTDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto& shape = args.input_real.padded_shape();
    return tt::tt_metal::operation::hash_operation<FFTDeviceOperation>(
        attrs,
        args.input_real.dtype(),
        args.input_real.memory_config(),
        shape);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::tuple<Tensor, Tensor> fft(
    const Tensor& input_real,
    bool inverse,
    const std::optional<Tensor>& input_imag,
    ttnn::experimental::prim::FFTPrecision precision) {
    using OperationType = ttnn::experimental::prim::FFTDeviceOperation;

    OperationType::operation_attributes_t attrs{
        .inverse   = inverse,
        .precision = precision,
    };
    OperationType::tensor_args_t          args{ .input_real = input_real,
                                                .input_imag = input_imag };

    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim
