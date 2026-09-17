// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SingleTileStockhamFactory — Metal 2.0 ProgramSpec factory for the
// single-tile Stockham FFT path (pow-2 N, 2 <= N <= 1024, fp32 or bf16).
//
// Selected by FFTDeviceOperation::select_program_factory for B == 1 with
// a real-only input.  Complex-input and multi-batch calls fall through
// to BatchedStockhamFactory (see fft_device_operation.cpp).

#pragma once

#include <tuple>

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "fft_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct SingleTileStockhamFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const FFTParams& operation_attributes,
        const FFTTensorArgs& tensor_args,
        std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
