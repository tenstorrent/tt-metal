// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Bring-up (DeepSeek-V4.1 FIX.15, agent rule 18): the block's programs inside the first max_x columns and max_y rows of
// a chip's compute grid. On a Blackhole Galaxy no op may run on more than 110 Tensix cores of a chip (the chassis'
// power supply trips when a chip's current rises too fast); the full 12 x 10 grid can do it. Default: no cap (the
// source's grid). Set once a process, before the first plan of any op here (MoEBlockResources,
// flat_routed_expert_plan): every planner reads the grid through block_grid(), and the plans are cached.

#pragma once

#include <cstdint>
#include <utility>

#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::bringup::moe_block_main {

// (0, 0): no cap. A cap that differs from the one the first plan saw is refused (TT_FATAL).
void set_max_grid(uint32_t max_x, uint32_t max_y);
std::pair<uint32_t, uint32_t> max_grid();
// the compute grid `full` cut to the cap (each side min(full, cap) where the cap is set); marks the cap as used
tt::tt_metal::CoreCoord block_grid(const tt::tt_metal::CoreCoord& full);

}  // namespace ttnn::operations::bringup::moe_block_main
