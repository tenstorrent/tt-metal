// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "fp32_state_sfpu.hpp"

#ifndef ARCH_BLACKHOLE
#error "SDPA FP32 streaming state is currently Blackhole-only"
#endif
static_assert(DST_ACCUM_MODE, "SDPA FP32 streaming state requires FP32 destination registers");

namespace sdpa::streaming {

// The owner must invalidate this cache after any pack configuration performed
// outside this object. Match the frozen C/D single-tile pack setup and preserve
// its skip-addrmod/skip-strides contract: full 32x32 tiles, initialized at startup.
struct Fp32PackConfig {
#ifdef TRISC_PACK
    uint32_t format_cb = 32;
    uint32_t width = 0;
#endif

    ALWI void invalidate() { PACK(format_cb = 32; width = 0;) }

    ALWI void single_tile(uint32_t cb) {
#ifdef TRISC_PACK
        if (format_cb == 32 || pack_src_format[format_cb] != pack_src_format[cb] ||
            pack_dst_format[format_cb] != pack_dst_format[cb]) {
            pack_reconfig_data_format(cb);
        }
        format_cb = cb;
        if (width == 1) {
            return;
        }
        width = 1;
        llk_pack_init<ckernel::PackMode::Default, false, true, true>(cb, 1);
#endif
    }
};

// Full-face FP32 unpack batching from the selected C/D implementation. MATH's
// extra zero-flag clears are mandatory even though UNPACK moves the data directly
// to DST. Restore the one-tile MOP before the following broadcast.
ALWI void state_unpack_mop(uint32_t faces) {
    UNPACK((
        ckernel_template(faces, 1, TT_OP_UNPACR(0, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1))
            .program()));
}

// Preconditions:
// - old/correction CBs are published and waited on by the caller; all use FP32
//   unpack-to-destination, never the 19-bit source-register path.
// - new_cb is reserved, not published, and already contains the current chunk;
//   its writes are fenced. PACK is in single-tile FP32 L1-accumulation mode.
// - input indices are relative to current read pointers, output indices to the
//   reserved write origin. This helper neither publishes nor consumes CBs.
// With an identity correction, correction_cb is not accessed at all. Four
// numerator tiles then fit in a half DST; nonidentity uses at most three.
template <int pairs, bool first_column = false>
ALWI void rescale_and_accumulate(
    uint32_t old_cb,
    uint32_t new_cb,
    uint32_t correction_cb,
    uint32_t old_index,
    uint32_t new_write_index,
    uint32_t correction_index,
    bool identity_correction = false) {
    // Four tiles are valid only for the identity path (no correction tile).
    static_assert(pairs >= 1 && pairs <= 4);
    static_assert(!first_column || pairs == 1);
    tile_regs_acquire();
    reconfig_data_format_skip_int8(old_cb, old_cb);
    unary_bcast_init<BroadcastType::NONE>(old_cb);
    if constexpr (pairs > 1) {
        state_unpack_mop(4 * pairs);
        unary_bcast<BroadcastType::NONE>(old_cb, old_index, 0);
        MATH(for (uint32_t tile = 1; tile < pairs; ++tile) {
            for (uint32_t face = 0; face < 4; ++face) {
                TT_ZEROACC(p_zeroacc::CLR_16, 1, 1, ADDR_MOD_3, get_dest_index_in_faces(tile, face));
            }
        })
        state_unpack_mop(4);
    } else {
        unary_bcast<BroadcastType::NONE>(old_cb, old_index, 0);
    }
    unary_bcast_uninit<BroadcastType::NONE>(old_cb);
    if (!identity_correction) {
        reconfig_data_format_skip_int8(correction_cb, correction_cb);
        unary_bcast_init<BroadcastType::COL>(correction_cb);
        unary_bcast<BroadcastType::COL>(correction_cb, correction_index, pairs);
        unary_bcast_uninit<BroadcastType::COL>(correction_cb);
    }
    tile_regs_commit();
    tile_regs_wait();
    if (!identity_correction) {
        if constexpr (first_column) {
            PACK((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                DST_SYNC_MODE, DST_ACCUM_MODE, sdpa_state_rescale_first_column, 0, VectorMode::C)));
        } else {
            PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, sdpa_state_rescale, (pairs), 0, VectorMode::None)));
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    }
    for (int p = 0; p < pairs; ++p) {
        pack_tile<true>(p, new_cb, new_write_index + p);
    }
    tile_regs_release();
}

// Consume numerator and first-column denominator rows, publish normalized rows.
// scratch_cb must be a one-tile FP32 CB with unpack-to-destination enabled.
// Zero/nonfinite denominators retain the frozen reciprocal behavior; no new
// epsilon, clamp, or fully-masked-row convention is introduced here.
template <uint32_t head_dim_tiles, uint32_t identity_cb = 32>
ALWI void normalize_rows(
    uint32_t sum_cb,
    uint32_t numerator_cb,
    uint32_t scratch_cb,
    uint32_t output_cb,
    uint32_t rows,
    Fp32PackConfig& pack) {
    static_assert(head_dim_tiles > 0);
    PACK((llk_pack_reconfig_l1_acc(0)));
    for (uint32_t row = 0; row < rows; ++row) {
        pack.invalidate();
        CircularBuffer(sum_cb).wait_front(1);
        if constexpr (identity_cb != 32) {
            CircularBuffer(identity_cb).wait_front(1);
        }
        CircularBuffer(scratch_cb).reserve_back(1);
        reconfig_data_format_skip_int8(sum_cb, sum_cb);
        tile_regs_acquire();
        unary_bcast_init<BroadcastType::NONE>(sum_cb);
        unary_bcast<BroadcastType::NONE>(sum_cb, 0, 0);
        unary_bcast_uninit<BroadcastType::NONE>(sum_cb);
        tile_regs_commit();
        tile_regs_wait();
        PACK(
            (SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, sdpa_state_reciprocal, 0, VectorMode::C)));
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        pack.single_tile(scratch_cb);
        pack_tile(0, scratch_cb);
        tile_regs_release();
        CircularBuffer(scratch_cb).push_back(1);
        CircularBuffer(sum_cb).pop_front(1);
        CircularBuffer(scratch_cb).wait_front(1);
        CircularBuffer(numerator_cb).wait_front(head_dim_tiles);
        CircularBuffer(output_cb).reserve_back(head_dim_tiles);
        pack.single_tile(output_cb);
        for (uint32_t col = 0; col < head_dim_tiles; ++col) {
            tile_regs_acquire();
            reconfig_data_format_skip_int8(numerator_cb, numerator_cb);
            unary_bcast_init<BroadcastType::NONE>(numerator_cb);
            unary_bcast<BroadcastType::NONE>(numerator_cb, col, 0);
            unary_bcast_uninit<BroadcastType::NONE>(numerator_cb);
            reconfig_data_format_skip_int8(scratch_cb, scratch_cb);
            unary_bcast_init<BroadcastType::COL>(scratch_cb);
            unary_bcast<BroadcastType::COL>(scratch_cb, 0, 1);
            unary_bcast_uninit<BroadcastType::COL>(scratch_cb);
            tile_regs_commit();
            tile_regs_wait();
            PACK((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                DST_SYNC_MODE, DST_ACCUM_MODE, sdpa_state_normalize, 0, VectorMode::None)));
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            pack_tile(0, output_cb);
            tile_regs_release();
        }
        CircularBuffer(output_cb).push_back(head_dim_tiles);
        CircularBuffer(numerator_cb).pop_front(head_dim_tiles);
        CircularBuffer(scratch_cb).pop_front(1);
    }
    pack_reconfig_data_format(scratch_cb);
    pack.invalidate();
}

}  // namespace sdpa::streaming
