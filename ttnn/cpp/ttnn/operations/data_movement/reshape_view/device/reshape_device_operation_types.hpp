// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::prim {

struct ReshapeViewParams {
    ttnn::Shape logical_output_shape;
    ttnn::Shape padded_output_shape;
    tt::tt_metal::MemoryConfig output_mem_config;
    bool recreate_mapping_tensor;
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grid;
    // The row factory serves every shape with the same last dim (see reshape_view_tiled_rows_is_shape_free).
    bool runtime_shapes = false;
};

struct ReshapeViewInputs {
    Tensor input;
};

}  // namespace ttnn::prim
