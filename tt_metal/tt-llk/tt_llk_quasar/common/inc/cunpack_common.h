// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"

namespace ckernel::unpack
{
// Unpack tilize config register struct
// Covers ADDR32 18-20 for UNPACKER0, ADDR32 30-32 for UNPACKER1
constexpr std::uint32_t NUM_WORDS_UNPACK_TILIZE_CFG = 3;

struct unpack_tilize_cfg_t
{
    // word 0 — ADDR32 18 or 30
    std::uint32_t src_z_stride      : 16; // TILIZE_SRC_Z_STRIDE
    std::uint32_t dst_z_stride      : 8;  // TILIZE_DST_Z_STRIDE
    std::uint32_t stride_val_source : 1;  // STRIDE_VAL_SOURCE
    std::uint32_t stride_no_write   : 1;  // STRIDE_NO_WRITE
    std::uint32_t reserved0         : 6;

    // word 1 — ADDR32 19 or 31
    std::uint32_t stride_mask_val : 32; // STRIDE_MASK_VAL

    // word 2 — ADDR32 20 or 32
    std::uint32_t stride_offset_0 : 16; // STRIDE_OFFSET_0
    std::uint32_t stride_offset_1 : 16; // STRIDE_OFFSET_1
};

static_assert(sizeof(unpack_tilize_cfg_t) == NUM_WORDS_UNPACK_TILIZE_CFG * sizeof(std::uint32_t));

union unpack_tilize_cfg_u
{
    std::uint32_t val[NUM_WORDS_UNPACK_TILIZE_CFG];
    unpack_tilize_cfg_t f;
};

// Number of rows for Unpack functions
constexpr static std::uint32_t UNPACR_STRIDE_MAX_ROWS = 8;
constexpr static std::uint32_t TRISC_ID               = 0;
} // namespace ckernel::unpack
