// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BatchedStockhamFactory — modern ProgramDescriptor program factory for the
// BATCHED single-tile Stockham FFT path. Handles input tensors of shape
// (B, N) where N is a pow-2 in [2, 1024], runs B independent length-N FFTs
// in parallel across multiple Tensix cores.
//
// This is the multi-core generalization of SingleTileStockhamFactory and
// the building block for the TwoPass composite assembled in fft.cpp.
//
// Wire-compatible with the existing batch_fft_{reader,writer,compute}.cpp
// kernels (which already support batch_per_core > 1 via their outer loop).

#pragma once

#include <tuple>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "fft_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct BatchedStockhamFactory {
    // ProgramDescriptor pattern: pure declarative program construction.
    // Like SingleTileStockhamFactory, no CachedProgram / no shared_variables /
    // no override_runtime_arguments.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const FFTParams& operation_attributes,
        const FFTTensorArgs& tensor_args,
        std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
