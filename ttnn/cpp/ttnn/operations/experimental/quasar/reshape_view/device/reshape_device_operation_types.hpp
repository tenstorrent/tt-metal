// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::prim::qsr {

struct ReshapeViewParams {
    ttnn::Shape logical_output_shape;
    ttnn::Shape padded_output_shape;
    tt::tt_metal::MemoryConfig output_mem_config;
    bool recreate_mapping_tensor;
    std::optional<CoreRangeSet> sub_core_grid;

    // Both factories ignore it: the mapping tensor depends only on the keyed shapes.
    static constexpr auto attributes_excluded_from_key = std::forward_as_tuple("recreate_mapping_tensor");
};

struct ReshapeViewInputs {
    Tensor input;
};

}  // namespace ttnn::prim::qsr
