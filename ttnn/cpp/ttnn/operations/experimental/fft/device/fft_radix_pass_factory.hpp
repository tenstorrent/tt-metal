// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// FftRadixPassFactory — ProgramDescriptor factory for the fused
// [batched length-P FFT + optional post-twiddle cmul] device op.
// Single-dispatch building block for the K-pass composite (commit 5).
//
// Reuses batch_fft_compute.cpp and batch_fft_writer.cpp verbatim;
// the only kernel-side delta vs BatchedStockhamFactory is the reader,
// which is radix_pass_reader.cpp (adds in-place scalar post-twiddle
// cmul on the final STATE buffer before signalling SYNC).

#pragma once

#include <tuple>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "fft_radix_pass_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct FftRadixPassFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const FftRadixPassParams& operation_attributes,
        const FftRadixPassTensorArgs& tensor_args,
        std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
