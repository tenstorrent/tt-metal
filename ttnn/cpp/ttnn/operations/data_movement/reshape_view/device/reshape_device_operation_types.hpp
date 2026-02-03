// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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
    std::optional<CoreRangeSet> sub_core_grid;
};

struct ReshapeViewInputs {
    Tensor input;
};

}  // namespace ttnn::prim
