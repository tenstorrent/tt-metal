// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ttnn::experimental::fft / ifft — 1-D Fast Fourier Transform.

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "device/fft_device_operation_types.hpp"

namespace ttnn::operations::experimental {

// Re-export of the device-op precision selector at the public op layer
// so callers don't have to reach into the prim:: namespace.
using FFTPrecision = ttnn::experimental::prim::FFTPrecision;

// We return a 2-tuple (real, imag) instead of std::pair because tt_stl's
// reflection layer (used by the ttnn dispatch / op-tracker) has a
// specialization for std::tuple but not std::pair. Returning a pair throws
// "Unsupported update of object of type: pair<...>" at runtime.

// Forward FFT — real input.
std::tuple<ttnn::Tensor, ttnn::Tensor> fft(
    const ttnn::Tensor& input_real,
    FFTPrecision precision = FFTPrecision::Precise);

// Forward FFT — complex input (input_real + i * input_imag).
std::tuple<ttnn::Tensor, ttnn::Tensor> fft(
    const ttnn::Tensor& input_real,
    const ttnn::Tensor& input_imag,
    FFTPrecision precision = FFTPrecision::Precise);

// Inverse FFT — always 2-arg (complex spectrum).
std::tuple<ttnn::Tensor, ttnn::Tensor> ifft(
    const ttnn::Tensor& spectrum_real,
    const ttnn::Tensor& spectrum_imag,
    FFTPrecision precision = FFTPrecision::Precise);

// Three-pass Cooley–Tukey composite for very large N (2^20 < N ≤ 2^30).
//
// ⚠ The input MUST already be pre-shaped to (B·N1·N2, N3) where
// N3 ≤ 1024 (see pick_three_factorization for the factorization rule).
// We expose this requirement because the (B, N) → (B·N1·N2, N3) reshape
// requires moving an N-element row through a single CB tile per core,
// which blows L1 for N > ~256K.  The caller is expected to do the
// equivalent `torch.view(B·N1·N2, N3)` on the host before
// `ttnn.from_torch`, so the device buffer is allocated with small
// page_size from the start.
//
// Output is returned in the factored shape (B·N1, N2, N3) — caller can
// `to_torch().reshape(B, full_N)` to recover natural-order (B, full_N).
//
// TODO (commit 7): add a streaming DRAM→DRAM rebank kernel so the
// public `fft()` API can transparently route (B, N) → fft_three_pass.
std::tuple<ttnn::Tensor, ttnn::Tensor> fft_three_pass(
    const ttnn::Tensor& input_real,
    uint32_t full_N,
    FFTPrecision precision = FFTPrecision::Precise);

// Complex-input variant (commit 6a, for Bluestein's intermediate
// length-M FFT).  `input_imag` must have the same pre-shape as
// `input_real`.  Adds one extra transpose_rm dispatch on the imag
// tensor at the head of the pipeline; the rest of the three-pass
// chain already handles complex data natively.
//
// `inverse=true` (commit 6c) requests an inverse FFT (IFFT).  Uses
// the swap-trick: requires BOTH halves of the spectrum (input_imag
// must be supplied), and folds the 1/full_N scale into the LAST
// fft_radix_pass writer via output_scale.  Zero extra dispatch vs
// forward FFT — the only "work" is two C++-level relabel swaps.
std::tuple<ttnn::Tensor, ttnn::Tensor> fft_three_pass(
    const ttnn::Tensor& input_real,
    std::optional<ttnn::Tensor> input_imag,
    uint32_t full_N,
    FFTPrecision precision = FFTPrecision::Precise,
    bool inverse = false);

}  // namespace ttnn::operations::experimental
