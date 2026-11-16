// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct AllGatherForMatmulParams {
    tt::tt_metal::CoreRangeSet output_core_range_set;
    // Row tiles only: label the replica [1, num_groups, cores * rows, K / num_groups] with shard
    // [num_groups * rows, K / num_groups] (the same bytes: group g is a run of K / num_groups / 32 tiles).
    uint32_t num_groups = 1;
};

struct AllGatherForMatmulInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::prim
