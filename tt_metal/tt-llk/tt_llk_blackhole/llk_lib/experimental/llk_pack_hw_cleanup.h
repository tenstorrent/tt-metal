// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "experimental/llk_hw_cleanup.h"
#include "llk_defs.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

using namespace ckernel;

template <bool is_fp32_dest_acc_en>
inline void _llk_pack_hw_cleanup_configure_current_bank_()
{
    constexpr std::uint32_t canonical_format = to_underlying(DataFormat::Float16_b);
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
        canonical_format, canonical_format, HW_CLEANUP_CANONICAL_TILE_SIZE_16B, FACE_R_DIM, TILE_C_DIM, 4, false, 0);
}

/**
 * Reproduce the pack portion of compute_kernel_hw_startup with canonical,
 * CB-independent format and geometry.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_hw_cleanup_init_default_()
{
    constexpr std::uint32_t canonical_format = to_underlying(DataFormat::Float16_b);

    // Equivalent to llk_pack_init<PackMode::Default>(ocb): restore the ambient
    // Default ADDR_MOD table, Default pack MOP, packer strides, and PAC X.
    _llk_pack_init_<PackMode::Default, false, false, false>(canonical_format, FACE_R_DIM, TILE_C_DIM, 4, 1, /*skip_bh_tilize_workaround=*/false);

    // Equivalent to llk_pack_dest_init<DST_ACCUM_MODE, PackMode::Default>(ocb).
    _llk_pack_dest_init_<Dst, is_fp32_dest_acc_en>();
}

/**
 * @brief Quiesces all TRISCs and restores pack cfg banks to canonical Float16_b 32x32 geometry.
 * @tparam Dst Kernel DST_SYNC_MODE (compile-time; not modified by cleanup).
 * @tparam is_fp32_dest_acc_en Kernel DST_ACCUM_MODE (re-asserted, not changed).
 * @post Both cfg banks use Float16_b pack formats with 32x32 tiles and four faces; bank 0 is selected.
 * @post PackMode::Default MOP, ADDR_MOD, strides, PAC X, pack-dest counters, and Dest sync match startup.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_hw_cleanup_canonical_()
{
    hw_cleanup::start<PackThreadId>();

    hw_cleanup::select_cfg_state(0);
    _llk_pack_hw_cleanup_configure_current_bank_<is_fp32_dest_acc_en>();

    hw_cleanup::select_cfg_state(1);
    _llk_pack_hw_cleanup_configure_current_bank_<is_fp32_dest_acc_en>();

    hw_cleanup::select_cfg_state(0);

    _llk_pack_hw_cleanup_init_default_<Dst, is_fp32_dest_acc_en>();

    hw_cleanup::finish<PackThreadId>();
}
