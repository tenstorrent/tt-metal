// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct AllGatherForMatmulParams {
    tt::tt_metal::CoreRangeSet output_core_range_set;
};

struct AllGatherForMatmulInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::prim
