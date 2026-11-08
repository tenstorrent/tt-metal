// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// mhc_post feed (deepseek_v41_flash_d_p P.36): the feed table's layout, shared by the host
// (device/mhc_post_stats_program_factory.cpp feed_table) and the kernels (mhc_post_feed_common.hpp). Word offsets.
// The table is the same for every chip; bytes are packed 4 a word, little end first.

#pragma once

#include <cstdint>

namespace mhc_feed_layout {

constexpr uint32_t FT_MAX_CHIP_ID = 64;  // fabric chip ids < 64
constexpr uint32_t FT_MAX_CHIPS = 64;    // mesh positions (R C)
constexpr uint32_t FT_MAX_G = 8;         // group size (G G side bits in 2 words)
constexpr uint32_t FT_MAX_PORTS = 8;     // 2 L

constexpr uint32_t FT_G = 0;            // group size
constexpr uint32_t FT_AXIS = 1;         // the group's mesh axis
constexpr uint32_t FT_C = 2;            // mesh columns (position = r C + c)
constexpr uint32_t FT_L = 3;            // links
constexpr uint32_t FT_MESH = 4;         // fabric mesh id
constexpr uint32_t FT_POS_OF_CHIP = 5;  // bytes: fabric chip id -> position (0xFF none)
constexpr uint32_t FT_CHIP_OF_POS = FT_POS_OF_CHIP + FT_MAX_CHIP_ID / 4;  // bytes: position -> fabric chip id
constexpr uint32_t FT_SIDE = FT_CHIP_OF_POS + FT_MAX_CHIPS / 4;  // bits: side of group position a -> b at a G + b
constexpr uint32_t FT_DIR = FT_SIDE + 2;                         // the eth direction of side 0, side 1
constexpr uint32_t FT_PORTS = FT_DIR + 2;                        // port slot -> core (x << 16 | y)
constexpr uint32_t FT_ULO = FT_PORTS + FT_MAX_PORTS;             // target t -> first unit touching column block t
constexpr uint32_t FT_UHI = FT_ULO + FT_MAX_G;                   // target t -> past the last
constexpr uint32_t FT_WORDS = FT_UHI + FT_MAX_G;
constexpr uint32_t FT_CHAN_WORDS = FT_MAX_CHIPS * FT_MAX_PORTS / 4;  // bytes: position 8 + slot -> fabric channel

}  // namespace mhc_feed_layout
