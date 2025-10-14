// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "views/compute_view.h"
#include "views/tile_views.h"

namespace NAMESPACE {
void MAIN {
    // rt args are:
    // - n_tiles
    // - any scalar params used in expression tree
    const auto n_tiles = get_arg_val<uint32_t>(0);

    using namespace tt;
    using namespace views;

    // ct args are:
    // - num_tiles_per_cycle
    // - cb_indices for input and internal nodes in postfix order
    // - cb_index for output tensor
    constexpr auto num_tiles_per_cycle = get_compile_time_arg_val(0);

    COMPUTE_TILES();
}
}  // namespace NAMESPACE
