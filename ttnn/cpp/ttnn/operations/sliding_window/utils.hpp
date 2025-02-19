// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>

namespace tt::tt_metal {

namespace utils {

inline void init_neighbor_core_xy_mapping(
    CoreCoord grid_size,
    std::map<CoreCoord, CoreCoord>& left_neighbor_core,
    std::map<CoreCoord, CoreCoord>& right_neighbor_core,
    bool is_twod = false) {
    TT_FATAL(
        (grid_size.x == 12 && grid_size.y == 9) || (grid_size.x == 8 && grid_size.y == 8) ||
        (grid_size.x == 8 && grid_size.y == 7));
    if (is_twod) {
        // 2d decomposition case (block sharded)
        // left-right neighbors are calculated along the x dim
        // first the left neighbors (x = 0 has no left neighbor)
        for (int32_t x = 1; x < grid_size.x; ++x) {
            int32_t left_x = x - 1;
            for (int32_t y = 0; y < grid_size.y; ++y) {
                CoreCoord core = {(uint32_t)x, (uint32_t)y};
                left_neighbor_core[core] = {(uint32_t)left_x, (uint32_t)y};
            }
        }
        // then the neighbors (x = grid_size.x - 1 has no left neighbor)
        for (int32_t x = 0; x < grid_size.x - 1; ++x) {
            int32_t right_x = x + 1;
            for (int32_t y = 0; y < grid_size.y; ++y) {
                CoreCoord core = {(uint32_t)x, (uint32_t)y};
                right_neighbor_core[core] = {(uint32_t)right_x, (uint32_t)y};
            }
        }
    } else {
        // default 1d distribution case (height sharded)
        for (int32_t y = 0; y < grid_size.y; ++y) {
            for (int32_t x = 0; x < grid_size.x; ++x) {
                CoreCoord core = {(uint32_t)x, (uint32_t)y};
                // calculate left neighbor
                int32_t left_x = x - 1, left_y = y;
                if (left_x < 0) {
                    left_x = grid_size.x - 1;
                    left_y -= 1;
                }
                if (left_y < 0) {
                    // there is no left neighbor
                } else {
                    left_neighbor_core[core] = {(uint32_t)left_x, (uint32_t)left_y};
                }
                // calculate right neighbor
                int32_t right_x = x + 1, right_y = y;
                if (right_x == grid_size.x) {
                    right_x = 0;
                    right_y += 1;
                }
                if (right_y == grid_size.y) {
                    // there is no right neighbor
                } else {
                    right_neighbor_core[core] = {(uint32_t)right_x, (uint32_t)right_y};
                }
            }
        }
    }
}

}  // namespace utils

}  // namespace tt::tt_metal
