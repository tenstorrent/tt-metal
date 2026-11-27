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
    // A row-major interleaved input's program serves every start and input size, and H and W ends past the input
    // read zeros (see padded_slice_rm_is_shape_free).
    const bool runtime_shapes = false;
};

struct PaddedSliceInputs {
    const Tensor& input;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::experimental::prim
