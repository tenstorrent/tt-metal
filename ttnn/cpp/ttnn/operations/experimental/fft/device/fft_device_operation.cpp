// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fft_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

namespace {

// Compile-time check: is N a power of two?
constexpr bool is_pow2(uint32_t n) { return n != 0u && (n & (n - 1u)) == 0u; }

}  // namespace

FFTDeviceOperation::program_factory_t FFTDeviceOperation::select_program_factory(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto& input = args.input_real;
    const auto& shape = input.padded_shape();
    const uint32_t N = static_cast<uint32_t>(shape[-1]);

    // Compute product of leading dims (batch size).
    uint32_t B = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d) {
        B *= static_cast<uint32_t>(shape[d]);
    }

    // Only forward FFT reaches prim::fft directly.  Inverse and all
    // large-N / non-pow-2 cases are routed by the ttnn::experimental::fft
    // C++ composite in fft.cpp and never reach this dispatcher.
    TT_FATAL(
        !attrs.inverse,
        "prim::fft: inverse must be handled by the composite router in fft.cpp "
        "(small_pow2_ifft / fft_two_pass / fft_three_pass_auto / bluestein_dispatch).");

    const auto dt = input.dtype();
    TT_FATAL(
        dt == tt::tt_metal::DataType::FLOAT32 || dt == tt::tt_metal::DataType::BFLOAT16,
        "prim::fft: only Float32 and BFloat16 inputs are supported (got {}).",
        static_cast<int>(dt));
    TT_FATAL(
        is_pow2(N) && N >= 2u && N <= 1024u,
        "prim::fft: only pow-2 N in [2, 1024] reaches this dispatcher; "
        "N={} should have been routed via fft_two_pass / fft_three_pass_auto / "
        "bluestein_dispatch in fft.cpp.",
        N);
    TT_FATAL(
        is_pow2(B) && B >= 1u,
        "prim::fft: batch dim B={} must be a positive power of two. "
        "The pow-2 restriction reflects the SingleTile/BatchedStockham core-grid layout "
        "(cores partition B evenly across a pow-2 grid). Workaround: pad the leading "
        "dims so their product is a pow-2, run the FFT, then slice back on host — or "
        "route via the composite ttnn.experimental.fft entrypoint which handles the "
        "large-N tiers without this restriction.",
        B);

    // Complex-input case (used by the composite router when it feeds an
    // already-complex tensor into a single Stockham pass).  Both halves
    // must be layout/dtype/shape-compatible.
    if (args.input_imag.has_value()) {
        const auto& imag = *args.input_imag;
        TT_FATAL(
            imag.dtype() == dt && imag.layout() == input.layout() && imag.padded_shape() == shape,
            "prim::fft: input_imag must match input_real in dtype/layout/shape.");
        return BatchedStockhamFactory{};
    }
    if (B == 1u) {
        return SingleTileStockhamFactory{};
    }
    return BatchedStockhamFactory{};
}

void FFTDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto& input = args.input_real;

    TT_FATAL(
        input.dtype() == tt::tt_metal::DataType::FLOAT32 || input.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "fft: only Float32 and BFloat16 inputs are supported (got {}).",
        static_cast<int>(input.dtype()));

    TT_FATAL(
        input.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "fft: only ROW_MAJOR layout is supported (got {}).",
        static_cast<int>(input.layout()));

    const auto& shape = input.padded_shape();
    TT_FATAL(shape.size() >= 1 && shape.size() <= 4, "fft: input must have 1-4 dimensions (got {}).", shape.size());

    const uint32_t N = shape[-1];
    TT_FATAL(N >= 2u, "fft: FFT length N must be >= 2 (got {}).", N);
    TT_FATAL(
        N <= 1024u,
        "prim::fft: N={} out of small-N range (2..1024).  Large-N routing "
        "should have intercepted this in fft.cpp.",
        N);

    if (attrs.inverse) {
        TT_FATAL(args.input_imag.has_value(), "fft (inverse): both real and imag spectrum tensors required.");
        TT_FATAL(
            args.input_imag->dtype() == input.dtype() && args.input_imag->layout() == input.layout() &&
                args.input_imag->padded_shape() == shape,
            "fft (inverse): spectrum_real and spectrum_imag must match "
            "in dtype/layout/shape.");
    }
}

FFTDeviceOperation::spec_return_value_t FFTDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& args) {
    // Output spectrum has the same shape/dtype/layout as the real input.
    return {args.input_real.tensor_spec(), args.input_real.tensor_spec()};
}

FFTDeviceOperation::tensor_return_value_t FFTDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& args) {
    using namespace tt::tt_metal;

    auto make_like = [&](const Tensor& ref) -> Tensor { return create_device_tensor(ref.tensor_spec(), ref.device()); };

    return {make_like(args.input_real), make_like(args.input_real)};
}

tt::stl::hash::hash_t FFTDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto& shape = args.input_real.padded_shape();
    // Include has_value() so a real-only call (SingleTileStockhamFactory,
    // index 0) and a complex call (BatchedStockhamFactory, index 1) for the
    // same shape/dtype never share a cache entry.  Without this bit, a
    // real-only N=32 bf16 test that runs first caches factory_index=0; the
    // Bluestein b_cyc FFT (complex, same shape) then gets a cache HIT and
    // blindly uses SingleTileStockhamFactory::create_descriptor, which
    // hard-codes zscratch (all-zeros) as the imaginary input regardless of
    // tensor_args.input_imag — corrupting plan->B_re for every subsequent
    // Bluestein call.  FftRadixPassDeviceOperation has the same fix; see that
    // file's comment for the full rationale.
    return tt::tt_metal::operation::hash_operation<FFTDeviceOperation>(
        attrs, args.input_real.dtype(), args.input_real.memory_config(), shape, args.input_imag.has_value());
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
        .inverse = inverse,
        .precision = precision,
    };
    OperationType::tensor_args_t args{.input_real = input_real, .input_imag = input_imag};

    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim
