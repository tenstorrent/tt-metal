// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"

namespace ckernel::pack
{
// Pack untilize stride config register struct
// Covers ADDR32 54-55 (THCON_PACKER0_REG1)
constexpr std::uint32_t NUM_WORDS_PACK_UNTILIZE_STRIDE_CFG = 2;

struct pack_untilize_stride_cfg_t
{
    // word 0 — ADDR32 54
    std::uint32_t edge_mask_mode : 2;  // EDGE_MASK_MODE
    std::uint32_t src_z_stride   : 8;  // PACK_UNTILIZE_SRC_Z_STRIDE
    std::uint32_t dst_z_stride   : 16; // PACK_UNTILIZE_DST_Z_STRIDE
    std::uint32_t reserved0      : 6;

    // word 1 — ADDR32 55
    std::uint32_t stride_offset_0 : 16; // PACK_STRIDE_OFFSET_0
    std::uint32_t stride_offset_1 : 16; // PACK_STRIDE_OFFSET_1
};

static_assert(sizeof(pack_untilize_stride_cfg_t) == NUM_WORDS_PACK_UNTILIZE_STRIDE_CFG * sizeof(std::uint32_t));

union pack_untilize_stride_cfg_u
{
    std::uint32_t val[NUM_WORDS_PACK_UNTILIZE_STRIDE_CFG];
    pack_untilize_stride_cfg_t f;
};

constexpr static std::uint32_t TRISC_ID = 2;
static std::uint32_t clear_dest_bank_id = 0;

inline void _update_clear_dest_bank_id_()
{
    clear_dest_bank_id = 1 - clear_dest_bank_id;
}
} // namespace ckernel::pack
