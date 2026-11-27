// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// t2_raw_kwalk_hxw: kw::matmul_accumulate_interm generalized from 1 x sb_w to sb_h x sb_w subblocks.
//
// Same two mechanisms as the 1 x sb_w raw walk (genify_matmul_kwalk.hpp, Perf 1), now for every subblock shape:
//  1. DEST advance by address modifier. The stock _llk_math_matmul_ writes the DEST offset (SETC16) before every
//     output tile of the subblock. Here the offset is written once per subblock pass; the last MVMUL of each tile
//     carries an address modifier that moves the DEST carry register to the next tile. The stock tile order is
//     kept: reuse_a = sb_w >= sb_h walks rows (SrcB = A row tile reused across the row's sb_w tiles, SrcA cleared
//     per tile, SrcB cleared at each row end), otherwise columns (SrcA = B column tile reused down the column's
//     sb_h tiles, SrcB cleared per tile, SrcA cleared at each column end). DEST tile index = r * sb_w + c, so a
//     row walk and a one-column walk advance one tile (64 DEST rows) per tile. A multi-column column walk
//     (sb_h > sb_w > 1, 16-bit DEST only) is not expressible (the caller keeps the helper).
//  2. One reserve / one push of the partials block (no per-K-block drain on the unpack thread).
// The unpack side is the stock llk_unpack_AB_matmul(ct = sb_w, rt = sb_h, kt = block_k), exactly as the helper.
//
// MATMUL_OUT_STREAM (variant b'): on the last K-block the partials are pushed row group by row group and each
// fronted row group is copied (CopyTile -> PackTile, the identity epilogue) to the output CB one row group behind
// the walk, so the output streams out during the last K-block instead of after it. Bit-identical to the
// whole-block identity copy after the walk.
//
// Address modifiers used: ADDR_MOD_3, ADDR_MOD_6, ADDR_MOD_7 (math thread), re-armed after every init.
// Contract: full 32x32 tiles, no transpose, 1 <= sb_h * sb_w <= DEST half, sb_w >= sb_h or sb_w == 1.

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/ttnn/operations/genify_matmul/kernels/genify_matmul_kschedule.hpp"

namespace kwh {

template <uint32_t rt, uint32_t ct>
struct Walk {
    static constexpr bool reuse_a = ct >= rt;
    static constexpr uint32_t t_dim = reuse_a ? rt : ct;
    static constexpr uint32_t rut_dim = reuse_a ? ct : rt;
    // DEST tile index r * ct + c: consecutive in MVMUL order for a row walk and a one-column walk. A multi-column
    // column walk (sb_h > sb_w > 1, 16-bit DEST only) is not expressible here: the first attempt (DEST stride
    // sb_w tiles + re-base per column) was measured incorrect, so kwalk_expressible keeps the helper there.
    static constexpr bool contiguous = reuse_a || ct == 1;
    static constexpr uint32_t dest_step = 1;
};

#ifdef TRISC_MATH
constexpr uint32_t mm_phases = to_underlying(MATH_FIDELITY) == 0 ? 1 : to_underlying(MATH_FIDELITY);
constexpr uint8_t AM_ADVANCE = ADDR_MOD_3;  // next tile: dest cr += 64 * dest_step, dest = cr
constexpr uint8_t AM_RESET = ADDR_MOD_6;    // last tile of the pass (or of a column): every counter cleared
constexpr uint8_t AM_PHASE = ADDR_MOD_7;    // HiFi phase end: back to the tile base, next phase

template <uint32_t dest_step>
ALWI void set_addrmods() {
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = static_cast<int16_t>(64 * dest_step), .clr = 0, .cr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(AM_ADVANCE);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 0, .clr = 1, .cr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(AM_RESET);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 0, .clr = 0, .cr = 1},
        .fidelity = {.incr = 1, .clr = 0},
    }
        .set(AM_PHASE);
}

static bool pass_throttled = false;

template <uint32_t rt, uint32_t ct>
ALWI void pass_begin(uint32_t in0_cb, uint32_t in1_cb) {
    const bool thr = (*throttle_ptr % 2) != 0;
    if (!thr && throttled_mop_status != 0) {  // back from a throttled pass: stock MOP/replay + ours
        llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb, in1_cb, 0, ct, rt);
        throttled_mop_status = 0;
        set_addrmods<Walk<rt, ct>::dest_step>();
    }
    pass_throttled = thr;
    if (!thr) {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    }
}

template <uint32_t rt, uint32_t ct>
ALWI void math_call(uint32_t in0_cb, uint32_t in1_cb) {
    using W = Walk<rt, ct>;
    if (pass_throttled) {
        matmul_block_math_dynamic_throttle(in0_cb, in1_cb, 0, 0, ct, rt);
        return;
    }
    constexpr uint32_t clr_tile = W::reuse_a ? p_setrwc::CLR_A : p_setrwc::CLR_B;  // streamed operand
    constexpr uint32_t clr_row = W::reuse_a ? p_setrwc::CLR_B : p_setrwc::CLR_A;   // reused operand
#pragma GCC unroll 8
    for (uint32_t t = 0; t < W::t_dim; ++t) {
#pragma GCC unroll 8
        for (uint32_t u = 0; u < W::rut_dim; ++u) {
            for (uint32_t p = 0; p + 1 < mm_phases; ++p) {
                lltt::replay(ckernel::math::replay_buf_offset, 15);
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, AM_PHASE, 0);
            }
            lltt::replay(ckernel::math::replay_buf_offset, 15);
            if (u + 1 == W::rut_dim && t + 1 == W::t_dim) {
                TTI_MVMUL(clr_tile, 0, AM_RESET, 0);
            } else {
                TTI_MVMUL(clr_tile, 0, AM_ADVANCE, 0);
            }
        }
        if (t + 1 == W::t_dim) {
            TTI_SETRWC(clr_row, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
        } else {
            // end of a reuse row: release the reused operand, keep the DEST counter (= the next tile base)
            TTI_SETRWC(clr_row, 0, 0, 0, 0, 0);
        }
    }
}
#endif

struct DefaultIn0 {
    ALWI uint32_t operator()(uint32_t /*kb*/, uint32_t in0_cb) const { return in0_cb; }
};
struct NoPostKBlock {
    ALWI void operator()(uint32_t, uint32_t, bool) const {}
};

// Identity copy of `ntiles` fronted partials tiles (from the read pointer + src0) into out_cb (reserved by the
// caller, at offsets dst0..), DEST-half-sized batches. Leaves unpack / math in copy mode and the packer on
// out_cb's format with L1 accumulation off.
template <uint32_t batch>
ALWI void copy_tiles(uint32_t partials_cb, uint32_t out_cb, uint32_t src0, uint32_t dst0, uint32_t ntiles) {
    for (uint32_t t = 0; t < ntiles; t += batch) {
        const uint32_t n = ntiles - t < batch ? ntiles - t : batch;
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            copy_tile(partials_cb, src0 + t + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile<true>(i, out_cb, dst0 + t + i);
        }
        tile_regs_release();
    }
}

// Accumulate one (in0_num_subblocks x sb_h) x (in1_num_subblocks x sb_w) block over the gm_ks K deliveries into
// partials_cb (fp32, packer L1 acc), tile row-major, row stride block_n. Waits / pops in0 and in1 per K-block;
// reserves the whole block in partials_cb before. stream_out == false: pushes it after (it ends fronted there
// for the epilogue). stream_out == true (epilogue none, variant b'): see the head comment; out_cb receives the
// block row group by row group (reserve / push per row group) and partials_cb ends empty.
// Leaves packer L1 accumulation enabled (caller disables it, as with the helper).
template <
    uint32_t sb_h,
    uint32_t sb_w,
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t block_k,
    uint32_t num_k_blocks,
    uint32_t last_block_k,
    bool stream_out = false,
    typename In0SourceFn = DefaultIn0,
    typename PostKBlockFn = NoPostKBlock>
ALWI void matmul_accumulate_interm(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t partials_cb,
    uint32_t out_cb = 0,
    In0SourceFn in0_source = {},
    PostKBlockFn post_k_block = {}) {
    constexpr uint32_t block_m = in0_num_subblocks * sb_h;
    constexpr uint32_t block_n = in1_num_subblocks * sb_w;
    constexpr uint32_t block_tiles = block_m * block_n;
    constexpr uint32_t rg_tiles = sb_h * block_n;  // one row group
    constexpr uint32_t in0_block_tiles = block_m * block_k;
    constexpr uint32_t in1_block_tiles = block_k * block_n;
    constexpr uint32_t copy_batch = sb_h * sb_w;  // the walk's own DEST-half budget
    static_assert(Walk<sb_h, sb_w>::contiguous, "multi-column column walk (sb_h > sb_w > 1): use the helper");

    reconfig_data_format(in1_cb, in0_cb);  // SrcA = in1, SrcB = in0
    PACK((pack_reconfig_data_format(partials_cb)));
    matmul_block_init(in0_cb, in1_cb, /*transpose=*/0, sb_w, sb_h, block_k);
    MATH((set_addrmods<Walk<sb_h, sb_w>::dest_step>()));
    PACK((llk_pack_reconfig_l1_acc(0)));
    cb_reserve_back(partials_cb, block_tiles);

    using KS = gm_ks::Sched<block_k, num_k_blocks, last_block_k>;
    for (uint32_t kb = 0; kb < KS::n; ++kb) {
        const uint32_t inner = KS::kv(kb);
        const uint32_t a_stride = gm_ks::compact_a ? inner : block_k;  // A slot row stride (tiles)
        const uint32_t a_cb = in0_source(kb, in0_cb);
        const bool last_kb = kb == KS::n - 1;
        cb_wait_front(a_cb, in0_block_tiles);
        cb_wait_front(in1_cb, in1_block_tiles);
        if (kb == 1) {
            PACK((llk_pack_reconfig_l1_acc(1)));
        }
        for (uint32_t m = 0; m < in0_num_subblocks; ++m) {
            for (uint32_t n = 0; n < in1_num_subblocks; ++n) {
                tile_regs_acquire();
                MATH((pass_begin<sb_h, sb_w>(in0_cb, in1_cb)));
                uint32_t in0_index = m * sb_h * a_stride;
                uint32_t in1_index = n * sb_w;
                for (uint32_t k = 0; k < inner; ++k) {
#ifndef SKIP_COMPUTE
                    UNPACK((llk_unpack_AB_matmul(a_cb, in1_cb, in0_index, in1_index, sb_w, sb_h, block_k)));
                    MATH((math_call<sb_h, sb_w>(in0_cb, in1_cb)));
#endif
                    in0_index++;
                    in1_index += block_n;
                }
                tile_regs_commit();
                tile_regs_wait();
                // stream_out pushes row groups on the last K-block: offsets are relative to the moving write ptr
                const uint32_t pushed = (stream_out && last_kb) ? m * rg_tiles : 0;
                const uint32_t col_base = m * rg_tiles + n * sb_w - pushed;
                for (uint32_t r = 0; r < sb_h; ++r) {
                    for (uint32_t c = 0; c < sb_w; ++c) {
                        pack_tile<true>(r * sb_w + c, partials_cb, col_base + r * block_n + c);
                    }
                }
                tile_regs_release();
            }
            if constexpr (stream_out) {
                if (last_kb) {
                    cb_push_back(partials_cb, rg_tiles);
                    if (m > 0) {  // copy the previous row group, one behind the walk
                        cb_wait_front(partials_cb, rg_tiles);
                        reconfig_data_format_srca(in1_cb, partials_cb);
                        copy_init(partials_cb);
                        PACK((pack_reconfig_data_format(out_cb)));
                        PACK((llk_pack_reconfig_l1_acc(0)));
                        cb_reserve_back(out_cb, rg_tiles);
                        copy_tiles<copy_batch>(partials_cb, out_cb, 0, 0, rg_tiles);
                        cb_push_back(out_cb, rg_tiles);
                        cb_pop_front(partials_cb, rg_tiles);
                        // back to the walk
                        reconfig_data_format_srca(partials_cb, in1_cb);
                        PACK((pack_reconfig_data_format(partials_cb)));
                        PACK((llk_pack_reconfig_l1_acc(1)));
                        matmul_block_init(in0_cb, in1_cb, /*transpose=*/0, sb_w, sb_h, block_k);
                        MATH((set_addrmods<Walk<sb_h, sb_w>::dest_step>()));
                    }
                }
            }
        }
        cb_pop_front(a_cb, in0_block_tiles);
        cb_pop_front(in1_cb, in1_block_tiles);
        post_k_block(kb, KS::n, last_kb);
    }
    if constexpr (stream_out) {
        // the last row group
        cb_wait_front(partials_cb, rg_tiles);
        reconfig_data_format_srca(in1_cb, partials_cb);
        copy_init(partials_cb);
        PACK((pack_reconfig_data_format(out_cb)));
        PACK((llk_pack_reconfig_l1_acc(0)));
        cb_reserve_back(out_cb, rg_tiles);
        copy_tiles<copy_batch>(partials_cb, out_cb, 0, 0, rg_tiles);
        cb_push_back(out_cb, rg_tiles);
        cb_pop_front(partials_cb, rg_tiles);
    } else {
        cb_push_back(partials_cb, block_tiles);
    }
}

}  // namespace kwh
