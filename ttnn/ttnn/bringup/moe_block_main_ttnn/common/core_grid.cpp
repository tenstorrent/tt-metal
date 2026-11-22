// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core_grid.hpp"

#include <algorithm>
#include <atomic>

#include <tt_stl/assert.hpp>

namespace ttnn::operations::bringup::moe_block_main {

namespace {
std::atomic<uint32_t> g_max_x{0}, g_max_y{0};
std::atomic<bool> g_used{false};
}  // namespace

void set_max_grid(uint32_t max_x, uint32_t max_y) {
    TT_FATAL(
        !g_used.load() || (g_max_x.load() == max_x && g_max_y.load() == max_y),
        "moe_block_main: the core cap is {}x{} since the first plan; set it before any plan ({}x{} asked)",
        g_max_x.load(),
        g_max_y.load(),
        max_x,
        max_y);
    g_max_x = max_x;
    g_max_y = max_y;
}

std::pair<uint32_t, uint32_t> max_grid() { return {g_max_x.load(), g_max_y.load()}; }

tt::tt_metal::CoreCoord block_grid(const tt::tt_metal::CoreCoord& full) {
    g_used = true;
    const uint32_t mx = g_max_x.load(), my = g_max_y.load();
    return {mx ? std::min<size_t>(full.x, mx) : full.x, my ? std::min<size_t>(full.y, my) : full.y};
}

}  // namespace ttnn::operations::bringup::moe_block_main
