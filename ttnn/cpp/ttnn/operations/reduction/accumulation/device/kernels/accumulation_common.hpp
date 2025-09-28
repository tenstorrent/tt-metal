// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

constexpr uint32_t ONE_TILE{1};
constexpr uint32_t FIRST_TILE{0};
constexpr uint32_t WORKING_REG{0};

constexpr uint32_t cb_in = tt::CBIndex::c_0;
constexpr uint32_t cb_out = tt::CBIndex::c_1;
constexpr uint32_t cb_start = tt::CBIndex::c_2;
constexpr uint32_t cb_acc = tt::CBIndex::c_3;

constexpr uint32_t INT32_TILE_DEST = WORKING_REG;
constexpr uint32_t INT32_TILE_ACC = 1;

enum class AccumulationOp : uint8_t { CUMSUM, CUMPROD };

FORCE_INLINE uint32_t get_tile_id(
    uint32_t low_rank_offset,
    uint32_t high_rank_offset,
    uint32_t j,
    uint32_t tiles_per_row,
    uint32_t input_tile_offset) {
    uint32_t base_tileid = low_rank_offset * (tiles_per_row * input_tile_offset) + high_rank_offset;
    uint32_t tileid = base_tileid + j * input_tile_offset;
    return tileid;
}
