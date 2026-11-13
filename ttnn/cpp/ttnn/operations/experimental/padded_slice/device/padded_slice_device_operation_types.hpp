// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct PaddedSliceParams {
    const ttnn::Shape padded_slice_start;
    const ttnn::Shape padded_slice_end;
    const ttnn::Shape step;
    const tt::tt_metal::MemoryConfig output_mem_config;
    // The per-channel affine + SiLU applied in the untilize pass
    const bool prologue_affine = false;
};

struct PaddedSliceInputs {
    const Tensor& input;
    std::optional<Tensor> preallocated_output;
    std::optional<Tensor> prologue_scale;  // [1, C] TILE tensors of the input dtype
    std::optional<Tensor> prologue_shift;
};

}  // namespace ttnn::experimental::prim
