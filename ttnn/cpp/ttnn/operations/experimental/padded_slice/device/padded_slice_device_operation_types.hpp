// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct PaddedSliceParams {
    const ttnn::Shape padded_slice_start;
    const ttnn::Shape padded_slice_end;
    const ttnn::Shape step;
    const tt::tt_metal::MemoryConfig output_mem_config;
    const std::optional<float> pad_value = std::nullopt;
};

struct PaddedSliceInputs {
    const Tensor& input;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::experimental::prim
