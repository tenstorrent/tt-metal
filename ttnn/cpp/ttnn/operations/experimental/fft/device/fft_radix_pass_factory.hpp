// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// FftRadixPassFactory — custom Metal 2.0 ProgramSpec factory for the fused
// [batched length-P FFT + optional post-twiddle cmul] device op.
// Single-dispatch building block for the K-pass composite.
//
// Reuses batch_fft_compute.cpp. The dataflow kernels are
// radix_pass_reader.cpp and radix_pass_writer.cpp: the reader optionally
// loads post-twiddle rows, and the writer applies the optional post-twiddle
// and 1/N scale on the final STATE buffer before signalling SYNC.

#pragma once

#include <tuple>

#include <optional>
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "fft_radix_pass_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct FftRadixPassFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const FftRadixPassParams& operation_attributes,
        const FftRadixPassTensorArgs& tensor_args,
        std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value);

    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const FftRadixPassParams& operation_attributes,
        const FftRadixPassTensorArgs& tensor_args,
        std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
