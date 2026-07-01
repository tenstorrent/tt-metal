// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FFT program factory — full-backend dispatcher.

#include "fft_program_factory.hpp"

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::experimental::prim {

FFTProgramFactory::cached_program_t FFTProgramFactory::create(
    const FFTParams&            /*operation_attributes*/,
    const FFTTensorArgs&        tensor_args,
    std::tuple<Tensor, Tensor>& /*tensor_return_value*/) {

    const uint32_t N     = tensor_args.input_real.logical_shape()[-1];
    const auto     dtype = tensor_args.input_real.dtype();

    // FFTProgramFactory is a legacy stub that no longer performs any work.
    // All N ranges are now handled by the ProgramDescriptor-based factories
    // (SingleTileStockhamFactory, BatchedStockhamFactory, FftRadixPassFactory)
    // or by the composite C++ routers (fft_two_pass, fft_three_pass_auto,
    // bluestein_dispatch) in fft.cpp.
    //
    // If execution reaches here, it means select_program_factory routed an
    // input that the router in fft.cpp should have intercepted first.  This
    // is always a bug — report it clearly so it can be fixed.
    TT_THROW(
        "FFTProgramFactory::create reached for N={}, dtype={} — this is a "
        "bug.  All supported (N, dtype) combinations should be handled by "
        "the C++ router in fft.cpp before prim::fft is invoked.  "
        "Supported ranges (with TT_FFT_NATIVE default ON):\n"
        "  pow-2 N ≤ 1024          → SingleTile/BatchedStockhamFactory\n"
        "  pow-2 1024 < N ≤ 2^20   → fft_two_pass composite\n"
        "  pow-2 2^20 < N ≤ 2^30   → fft_three_pass_auto composite\n"
        "  non-pow-2 M ≤ 2^30      → bluestein_dispatch composite\n"
        "If you are calling ttnn::prim::fft directly (bypassing fft.cpp),\n"
        "use ttnn::experimental::fft instead.",
        N, static_cast<int>(dtype));
}

void FFTProgramFactory::override_runtime_arguments(
    cached_program_t&           /*cached_program*/,
    const FFTParams&            /*operation_attributes*/,
    const FFTTensorArgs&        /*tensor_args*/,
    std::tuple<Tensor, Tensor>& /*tensor_return_value*/) {
    // This should never be reached — see create() above.
    TT_THROW(
        "FFTProgramFactory::override_runtime_arguments reached — "
        "this is a bug (see create() for details).");
}

}  // namespace ttnn::experimental::prim

