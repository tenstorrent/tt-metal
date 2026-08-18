// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_template.h"
#include "experimental/llk_hw_cleanup.h"
#include "llk_unpack_common.h"

using namespace ckernel;

template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_hw_cleanup_configure_current_bank_()
{
    constexpr std::uint32_t canonical_format = to_underlying(DataFormat::Float16_b);
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        canonical_format,
        canonical_format,
        canonical_format,
        canonical_format,
        FACE_R_DIM,
        FACE_R_DIM,
        4,
        4,
        HW_CLEANUP_CANONICAL_TILE_SIZE_16B,
        HW_CLEANUP_CANONICAL_TILE_SIZE_16B);
}

/**
 * @brief Quiesces all TRISCs and restores unpack cfg banks to canonical Float16_b 32x32 geometry.
 * @tparam is_fp32_dest_acc_en Kernel DST_ACCUM_MODE (re-asserted, not changed).
 * @note On return both cfg banks use Float16_b 32x32 tiles with four faces (2048 bytes), and
 *       bank 0 is selected.
 * @note On return MOP CFG is NOP-poisoned to a 1x1 double-loop template.
 */
template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_hw_cleanup_canonical_()
{
    hw_cleanup::start<UnpackThreadId>();

    hw_cleanup::select_cfg_state(0);
    _llk_unpack_hw_cleanup_configure_current_bank_<is_fp32_dest_acc_en>();

    hw_cleanup::select_cfg_state(1);
    _llk_unpack_hw_cleanup_configure_current_bank_<is_fp32_dest_acc_en>();

    hw_cleanup::select_cfg_state(0);

    // 1x1 NOP double-loop so accidental MOP runs (template or unpack-template) are inert.
    ckernel_template(1, 1).program();

    hw_cleanup::finish<UnpackThreadId>();
}
