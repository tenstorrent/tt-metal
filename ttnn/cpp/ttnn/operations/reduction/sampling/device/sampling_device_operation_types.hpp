// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct SamplingParams {
    std::optional<uint32_t> seed;
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;
};

struct SamplingInputs {
    Tensor input_values;
    Tensor input_indices;
    Tensor k;
    Tensor p;
    Tensor temp;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::prim
