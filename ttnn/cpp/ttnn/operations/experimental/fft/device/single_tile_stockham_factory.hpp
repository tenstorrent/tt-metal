// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SingleTileStockhamFactory — ProgramDescriptor program factory for the
// single-tile Stockham FFT path (pow-2 N, 2 <= N <= 1024, fp32 or bf16).
//
// Selected by FFTDeviceOperation::select_program_factory for B == 1 with
// a real-only input.  Complex-input and multi-batch calls fall through
// to BatchedStockhamFactory (see fft_device_operation.cpp).

#pragma once

#include <tuple>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "fft_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct SingleTileStockhamFactory {
    // ProgramDescriptor pattern: pure declarative program construction.
    // No CachedProgram, no shared_variables_t, no override_runtime_arguments.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const FFTParams& operation_attributes,
        const FFTTensorArgs& tensor_args,
        std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
