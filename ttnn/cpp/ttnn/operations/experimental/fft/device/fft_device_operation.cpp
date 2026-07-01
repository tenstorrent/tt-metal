// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fft_device_operation.hpp"
#include "ttnn/device_operation.hpp"

#include <cstdlib>  // std::getenv

namespace ttnn::experimental::prim {

namespace {

// Compile-time check: is N a power of two?
constexpr bool is_pow2(uint32_t n) {
    return n != 0u && (n & (n - 1u)) == 0u;
}

// New-path rollout switch.  The ProgramDescriptor factories (SingleTile,
// Batched, RadixPass, …) are now the default.  Set TT_FFT_NATIVE=0 to
// revert to the legacy FFTProgramFactory for debugging only.
bool native_path_enabled() {
    const char* v = std::getenv("TT_FFT_NATIVE");
    if (v == nullptr) return true;               // default ON
    return !(v[0] == '0' && v[1] == '\0');       // OFF only if explicitly "0"
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

FFTDeviceOperation::program_factory_t FFTDeviceOperation::select_program_factory(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto& input = args.input_real;
    const auto& shape = input.padded_shape();
    const uint32_t N  = static_cast<uint32_t>(shape[-1]);

    // Compute product of leading dims (batch size). Pow-2 check happens
    // inside the factory; here we just sniff for "batched" vs "single".
    uint32_t B = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d) {
        B *= static_cast<uint32_t>(shape[d]);
    }

    // New ProgramDescriptor paths: fp32 OR bf16, forward FFT, N<=1024,
    // pow-2.  Gated by TT_FFT_NATIVE=1 during rollout.
    //   B == 1, real-only       → SingleTileStockhamFactory (commits 1, 2)
    //   B  > 1, real-only       → BatchedStockhamFactory    (commit 3a)
    //   B >= 1, complex (re+im) → BatchedStockhamFactory    (commit 3c —
    //                              uses caller's imag buffer instead of
    //                              the cached zero scratch; needed for
    //                              the Pass-2 step of the two-pass
    //                              composite, which feeds an already-
    //                              transformed complex tensor in).
    const auto dt = input.dtype();
    const bool dtype_ok =
        dt == tt::tt_metal::DataType::FLOAT32 ||
        dt == tt::tt_metal::DataType::BFLOAT16;
    if (native_path_enabled() &&
        !attrs.inverse &&
        dtype_ok &&
        is_pow2(N) && N >= 2u && N <= 1024u &&
        is_pow2(B) && B >= 1u) {
        // Verify imag (if present) is layout/dtype/shape-compatible —
        // the BatchedStockham kernels assume both buffers are wire-
        // identical except for content.
        if (args.input_imag.has_value()) {
            const auto& imag = *args.input_imag;
            if (imag.dtype() != dt ||
                imag.layout() != input.layout() ||
                imag.padded_shape() != shape) {
                return FFTProgramFactory{};   // fall back, safer than asserting
            }
            return BatchedStockhamFactory{};
        }
        if (B == 1u) return SingleTileStockhamFactory{};
        return BatchedStockhamFactory{};
    }

    // Default: existing dispatcher (covers everything else).
    return FFTProgramFactory{};
}

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
    // Include has_value() so that a real-only call (SingleTileStockhamFactory,
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
        attrs,
        args.input_real.dtype(),
        args.input_real.memory_config(),
        shape,
        args.input_imag.has_value());
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
