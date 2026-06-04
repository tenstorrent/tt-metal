// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ttnn::experimental::fft / ifft — 1-D Fast Fourier Transform.

#pragma once

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

}  // namespace ttnn::operations::experimental
