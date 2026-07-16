// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::prim::qsr {

struct ReshapeViewParams {
    ttnn::Shape logical_output_shape;
    ttnn::Shape padded_output_shape;
    tt::tt_metal::MemoryConfig output_mem_config;
    bool recreate_mapping_tensor;
    std::optional<CoreRangeSet> sub_core_grid;

    static constexpr auto attribute_names =
        std::forward_as_tuple("logical_output_shape", "output_mem_config", "sub_core_grid");
    auto attribute_values() const {
        return std::make_tuple(std::cref(logical_output_shape), std::cref(output_mem_config), std::cref(sub_core_grid));
    }
};

struct ReshapeViewInputs {
    Tensor input;
};

}  // namespace ttnn::prim::qsr
