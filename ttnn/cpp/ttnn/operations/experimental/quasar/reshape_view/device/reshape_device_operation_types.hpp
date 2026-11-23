// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::prim::qsr {

// Quasar reserves DM0 (DFB ISR) and DM1 (remapper), leaving six DM harts on a node. The reshape kernels
// use no DataflowBuffer and no compute unit, so all six move data.
inline constexpr uint32_t RESHAPE_MAX_DM_THREADS = 6;

struct ReshapeViewParams {
    ttnn::Shape logical_output_shape;
    ttnn::Shape padded_output_shape;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grid;
    // DM harts per node running the reshape kernel (1..RESHAPE_MAX_DM_THREADS). Part of the program-cache key.
    uint32_t num_threads = RESHAPE_MAX_DM_THREADS;
    // TILE layout: the 32-bit word (the element replicated to fill it) written to the tile padding of the
    // output. Part of the program-cache key.
    uint32_t pad_word = 0;
};

struct ReshapeViewInputs {
    Tensor input;
};

}  // namespace ttnn::prim::qsr
