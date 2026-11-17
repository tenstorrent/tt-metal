// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Perf 1 graduation from perf_experiments/matmul_kwalk_efficiency (focus whole op 466.3 -> 441.5 us, bit-identical).
// kw::matmul_accumulate_interm -- the K-walk of genify_matmul's matmul_accumulate_block for the
// (TileRowMajor, packer_l1_acc, LastBlockTarget::Interm, 1 x sb_w subblock) configuration, i.e. what
// compute_kernel_lib::matmul_block<false, true, Interm, TileRowMajor, Short, WaitAndPop, WaitAndPop, ...,
// caller_owns_pack_target> computes, with two mechanism changes. Bit-identical result.
//
// RAW-LLK justification (bypasses compute_kernel_lib::matmul_block; capability, not ergonomics):
//  1. DEST advance by address modifier. The stock math call (_llk_math_matmul_, used by every
//     matmul_block flavour incl. matmul_block_no_mop) writes the DEST offset register (TT_SETC16 via
//     math::set_dst_write_addr) before EVERY output tile's MVMUL sequence; each such write costs ~1 FPU
//     cycle on Blackhole (measured: 16 + ~1.3 cyc per LoFi tile-K). Here the offset is written once per
//     subblock pass; the 16th MVMUL of each tile carries an address modifier that moves the DEST
//     counter's carry register to the next tile (cr += 64 rows) -- the last tile of the subblock clears
//     it. The first 15 MVMULs are the stock replay buffer written by llk_math_matmul_init. No helper or
//     LLK API exposes a multi-tile matmul without the per-tile SETC16.
//  2. One reserve / one push of the partials block (the helper's caller_owns_pack_target protocol):
//     no per-row-group reserve/push and no per-K-block drain wait/pop on the UNPACK thread (which made
//     the unpacker wait for the packer to finish each K-block before unpacking the next).
//  The packer data format / L1-acc state is also set once (K-block 0 / 1) instead of per subblock.
//
// Dynamic throttle: the firmware flag is read once per subblock pass (stock reads it per call); a
// throttled pass runs the stock throttled call and the custom state is re-armed on the way back.
// Address modifiers used: ADDR_MOD_3, ADDR_MOD_6, ADDR_MOD_7 (math thread), re-armed by the init.
// Contract: full 32x32 tiles, no transpose, sb_h == 1 (reuse-A path), 1 <= sb_w <= DEST half.

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/ttnn/operations/genify_matmul/kernels/genify_matmul_kschedule.hpp"

namespace kw {

#ifdef TRISC_MATH
constexpr uint32_t mm_phases = to_underlying(MATH_FIDELITY) == 0 ? 1 : to_underlying(MATH_FIDELITY);
constexpr uint8_t AM_ADVANCE = ADDR_MOD_3;  // tile t < ct-1 last MVMUL: dest cr += 64, dest = cr
constexpr uint8_t AM_RESET = ADDR_MOD_6;    // tile ct-1 last MVMUL: every counter cleared
constexpr uint8_t AM_PHASE = ADDR_MOD_7;    // HiFi phase end: back to the tile base, next phase

ALWI void set_addrmods() {
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 64, .clr = 0, .cr = 1},
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

ALWI void pass_begin(uint32_t in0_cb, uint32_t in1_cb, uint32_t ct) {
    const bool thr = (*throttle_ptr % 2) != 0;
    if (!thr && throttled_mop_status != 0) {  // back from a throttled pass: stock MOP/replay + ours
        llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb, in1_cb, 0, ct, 1);
        throttled_mop_status = 0;
        set_addrmods();
    }
    pass_throttled = thr;
    if (!thr) {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    }
}

template <uint32_t ct>
ALWI void math_call(uint32_t in0_cb, uint32_t in1_cb) {
    if (pass_throttled) {
        matmul_block_math_dynamic_throttle(in0_cb, in1_cb, 0, 0, ct, 1);
        return;
    }
#pragma GCC unroll 8
    for (uint32_t t = 0; t < ct; ++t) {
        for (uint32_t p = 0; p + 1 < mm_phases; ++p) {
            lltt::replay(ckernel::math::replay_buf_offset, 15);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, AM_PHASE, 0);
        }
        lltt::replay(ckernel::math::replay_buf_offset, 15);
        if (t + 1 < ct) {
            TTI_MVMUL(p_setrwc::CLR_A, 0, AM_ADVANCE, 0);
        } else {
            TTI_MVMUL(p_setrwc::CLR_A, 0, AM_RESET, 0);
        }
    }
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
}
#endif

struct DefaultIn0 {
    ALWI uint32_t operator()(uint32_t /*kb*/, uint32_t in0_cb) const { return in0_cb; }
};
struct NoPostKBlock {
    ALWI void operator()(uint32_t, uint32_t, bool) const {}
};

// Accumulate one (in0_num_subblocks x 1) x (in1_num_subblocks x sb_w) block over the K deliveries of the
// gm_ks schedule (Perf 2 head ramp: per-delivery width kv, A row stride kv under GM_COMPACT_A; without the
// ramp, num_k_blocks K-blocks with the ragged last_block_k last)
// into partials_cb (fp32, packer L1 acc), tile row-major, row stride block_n. Waits / pops in0 and in1
// per K-block; reserves the whole block in partials_cb before and pushes it after (it ends fronted
// there for the epilogue). Leaves packer L1 accumulation enabled (caller disables it, as with the helper).
// The boot hw_configure (compute_kernel_hw_startup<SrcOrder::Reverse>) is the caller's.
template <
    uint32_t sb_w,
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t block_k,
    uint32_t num_k_blocks,
    uint32_t last_block_k,
    typename In0SourceFn = DefaultIn0,
    typename PostKBlockFn = NoPostKBlock>
ALWI void matmul_accumulate_interm(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t partials_cb,
    In0SourceFn in0_source = {},
    PostKBlockFn post_k_block = {}) {
    constexpr uint32_t block_m = in0_num_subblocks;
    constexpr uint32_t block_n = in1_num_subblocks * sb_w;
    constexpr uint32_t block_tiles = block_m * block_n;
    constexpr uint32_t in0_block_tiles = block_m * block_k;
    constexpr uint32_t in1_block_tiles = block_k * block_n;

    reconfig_data_format(in1_cb, in0_cb);  // SrcA = in1, SrcB = in0
    PACK((pack_reconfig_data_format(partials_cb)));
    matmul_block_init(in0_cb, in1_cb, /*transpose=*/0, sb_w, 1, block_k);
    MATH((set_addrmods()));
    PACK((llk_pack_reconfig_l1_acc(0)));
    cb_reserve_back(partials_cb, block_tiles);

    using KS = gm_ks::Sched<block_k, num_k_blocks, last_block_k>;
    for (uint32_t kb = 0; kb < KS::n; ++kb) {
        const uint32_t inner = KS::kv(kb);
        const uint32_t a_stride = gm_ks::compact_a ? inner : block_k;  // A slot row stride (tiles)
        const uint32_t a_cb = in0_source(kb, in0_cb);
        cb_wait_front(a_cb, in0_block_tiles);
        cb_wait_front(in1_cb, in1_block_tiles);
        if (kb == 1) {
            PACK((llk_pack_reconfig_l1_acc(1)));
        }
        for (uint32_t m = 0; m < block_m; ++m) {
            for (uint32_t n = 0; n < in1_num_subblocks; ++n) {
                tile_regs_acquire();
                MATH((pass_begin(in0_cb, in1_cb, sb_w)));
                uint32_t in0_index = m * a_stride;
                uint32_t in1_index = n * sb_w;
                for (uint32_t k = 0; k < inner; ++k) {
#ifndef SKIP_COMPUTE
                    UNPACK((llk_unpack_AB_matmul(a_cb, in1_cb, in0_index, in1_index, sb_w, 1, block_k)));
                    MATH((math_call<sb_w>(in0_cb, in1_cb)));
#endif
                    in0_index++;
                    in1_index += block_n;
                }
                tile_regs_commit();
                tile_regs_wait();
                const uint32_t col_base = m * block_n + n * sb_w;
                for (uint32_t c = 0; c < sb_w; ++c) {
                    pack_tile<true>(c, partials_cb, col_base + c);
                }
                tile_regs_release();
            }
        }
        cb_pop_front(a_cb, in0_block_tiles);
        cb_pop_front(in1_cb, in1_block_tiles);
        post_k_block(kb, KS::n, kb == KS::n - 1);
    }
    cb_push_back(partials_cb, block_tiles);
}

}  // namespace kw
