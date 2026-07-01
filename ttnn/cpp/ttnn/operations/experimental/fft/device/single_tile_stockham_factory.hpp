// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SingleTileStockhamFactory — modern ProgramDescriptor program factory for
// the fp32 single-tile Stockham FFT path (pow-2 N, 2 <= N <= 1024).
//
// This is the first step of the host-to-device refactor (see PR review
// thread). It establishes the ProgramDescriptor pattern alongside the
// legacy FFTProgramFactory; the dispatcher is env-var gated for safe
// rollout (TT_FFT_NATIVE=1).

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
