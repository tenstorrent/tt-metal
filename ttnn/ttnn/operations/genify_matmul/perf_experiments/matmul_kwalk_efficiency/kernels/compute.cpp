// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// K-walk bench compute: genify_matmul's matmul_accumulate_block (regime R1, epilogue != none) in
// isolation. A (cb_act, bf16 -> SrcB) and B (cb_weight, bfp4_b -> SrcA) arrive one K-block at a time;
// the block (block_m x block_n tiles, sb_h x sb_w subblocks) is accumulated in fp32 in cb_partials by
// packer L1 accumulation, the last K-block landing there too (LastBlockTarget::Interm), fronted for the
// epilogue. The bench's epilogue is a plain fp32 copy to cb_out under CHECK_OUT, a pop otherwise.
//
// KW_VARIANT
//   0  baseline: compute_kernel_lib::matmul_block with the op's exact template args (helper-owned
//      per-row-group reserve/push on cb_partials + per-K-block drain wait/pop on the UNPACK thread).
//   1  same helper, caller_owns_pack_target=true: one reserve before / one push after the walk, no
//      per-K-block drain (the helper's documented mode for TileRowMajor + packer_l1_acc + Interm).
//   2  raw K-walk (bypasses matmul_block -- see RAW-LLK note below): variant 1's CB protocol plus the
//      packer config hoisted out of the walk (pack data format once; L1-acc toggled once at K-block 1
//      instead of re-issued per subblock).
//
// RAW-LLK note (variant 2): bypasses compute_kernel_lib::matmul_block. Mechanism: the helper re-issues
// pack_reconfig_data_format(interm) + llk_pack_reconfig_l1_acc() before EVERY subblock pack (588 per
// block here) even though neither changes inside a K-block; hoisting them needs a helper change.
//
// Ablations (any variant that walks itself, i.e. 2): SKIP_COMPUTE drops the matmul LLK calls,
// ABL_NOPACK drops the pack_tile calls (DEST handshake kept).

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/experimental/matmul_custom.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "kw_matmul_walk.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_act = get_compile_time_arg_val(0);
constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
constexpr uint32_t cb_out = get_compile_time_arg_val(2);
constexpr uint32_t cb_partials = get_compile_time_arg_val(3);
constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(4);
constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(5);
constexpr uint32_t sb_h = get_compile_time_arg_val(6);
constexpr uint32_t sb_w = get_compile_time_arg_val(7);
constexpr uint32_t block_k = get_compile_time_arg_val(8);
constexpr uint32_t num_k_blocks = get_compile_time_arg_val(9);
constexpr uint32_t last_block_k = get_compile_time_arg_val(10);
constexpr uint32_t block_m = in0_num_subblocks * sb_h;
constexpr uint32_t block_n = in1_num_subblocks * sb_w;
constexpr uint32_t block_tiles = block_m * block_n;
constexpr uint32_t act_block_tiles = block_m * block_k;
constexpr uint32_t weight_block_tiles = block_k * block_n;

#ifndef KW_VARIANT
#define KW_VARIANT 0
#endif

struct LastKBlockInnerDim {
    ALWI uint32_t operator()(uint32_t block, uint32_t bk) const {
        return (block == num_k_blocks - 1) ? last_block_k : bk;
    }
};

template <bool caller_owns>
ALWI void helper_walk(
    CircularBuffer& act_buf, CircularBuffer& weight_buf, CircularBuffer& out_buf, CircularBuffer& interm_buf) {
    const auto shape =
        ckl::MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks, sb_h, sb_w, block_k, num_k_blocks, 1);
    ckl::matmul_block<
        /*transpose=*/false,
        /*packer_l1_acc=*/true,
        ckl::LastBlockTarget::Interm,
        ckl::OutputCBLayout::TileRowMajor,
        ckl::matmul_config::InitMode::Short,
        ckl::InputPolicy::WaitAndPopPerKBlock,
        ckl::InputPolicy::WaitAndPopPerKBlock,
        ckl::NoPostCompute,
        ckl::NoPreKBlock,
        ckl::NoPostKBlock,
        /*untilize_block_ct_dim=*/0,
        LastKBlockInnerDim,
        ckl::NoIn0Source,
        ckl::NoIn1BaseOffset,
        caller_owns>(
        act_buf,
        weight_buf,
        out_buf,
        interm_buf,
        shape,
        ckl::NoPostCompute{},
        ckl::NoPreKBlock{},
        /*in1_per_core_w=*/block_n,
        /*out_row_width=*/block_n,
        ckl::NoPostKBlock{},
        LastKBlockInnerDim{},
        ckl::NoIn0Source{},
        ckl::NoIn1BaseOffset{});
}

// Variant 2: raw K-walk, same math / pack order as the helper (subblock row-major, K inner).
// MM_FLAVOR 0: matmul_block (MOP + dynamic throttle check); 1: matmul_block_no_mop (replay, static
// MM_THROTTLE -- diagnostic); 2: matmul_block's unpack + llk_math_matmul<MATH_FIDELITY, 0> directly
// (no per-call throttle-flag read -- diagnostic).
#ifndef MM_FLAVOR
#define MM_FLAVOR 0
#endif
// MM_FLAVOR 4 (candidate "dest_advance"): the math side of a 1 x sb_w matmul call without the per-tile
// SETC16 dest-offset write. Stock _llk_math_matmul_ issues set_dst_write_addr (TT_SETC16 on the DEST
// offset register) before EVERY output tile's MOP; measured ~1 FPU cycle each (diag_one_setc). Here the
// DEST offset is written once per subblock pass; each tile's last MVMUL carries an address modifier
// that moves the DEST counter's carry register to the next tile (cr += 64 rows) instead of clearing
// it, and the subblock's last tile clears it (back to tile 0 for the next K step). The first 15 MVMULs
// of the tile are the stock replay buffer (replay_buf_offset, written by llk_math_matmul_init); only
// the 16th is issued here with the chosen modifier. Same MVMUL sequence, same SrcA/SrcB clears, same
// fidelity phases -> bit-identical result. Dynamic throttle: the firmware flag is read once per pass;
// when set, the pass falls back to the stock throttled call (and the custom state is re-armed after).
#ifdef TRISC_MATH
constexpr uint32_t kw_phases = to_underlying(MATH_FIDELITY) == 0 ? 1 : to_underlying(MATH_FIDELITY);
constexpr uint8_t KW_AM_ADVANCE = ADDR_MOD_3;  // last MVMUL of tile t < ct-1: dest cr += 64, dest = cr
constexpr uint8_t KW_AM_RESET = ADDR_MOD_6;    // last MVMUL of tile ct-1: all counters cleared
constexpr uint8_t KW_AM_PHASE = ADDR_MOD_7;    // HiFi: phase end, back to the tile base, next phase
ALWI void kw_set_addrmods() {
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 64, .clr = 0, .cr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(KW_AM_ADVANCE);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 0, .clr = 1, .cr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(KW_AM_RESET);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 0, .clr = 0, .cr = 1},
        .fidelity = {.incr = 1, .clr = 0},
    }
        .set(KW_AM_PHASE);
}
static bool kw_throttled = false;
ALWI void kw_pass_begin() {
    const bool thr = (*throttle_ptr % 2) != 0;
    if (!thr && throttled_mop_status != 0) {  // back from a throttled pass: restore stock MOP + ours
        llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(cb_act, cb_weight, 0, sb_w, sb_h);
        throttled_mop_status = 0;
        kw_set_addrmods();
    }
    kw_throttled = thr;
    if (!thr) {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    }
}
ALWI void kw_math_call() {
    if (kw_throttled) {
        matmul_block_math_dynamic_throttle(cb_act, cb_weight, 0, 0, sb_w, sb_h);
        return;
    }
#pragma GCC unroll 8
    for (uint32_t t = 0; t < sb_w; ++t) {
        for (uint32_t p = 0; p + 1 < kw_phases; ++p) {
            lltt::replay(ckernel::math::replay_buf_offset, 15);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, KW_AM_PHASE, 0);
        }
        lltt::replay(ckernel::math::replay_buf_offset, 15);
        if (t + 1 < sb_w) {
            TTI_MVMUL(p_setrwc::CLR_A, 0, KW_AM_ADVANCE, 0);
        } else {
#ifdef KW_CLR_AB
            // KW_CLR_AB: the subblock's last MVMUL also releases SrcB (CLR_AB), replacing the
            // separate SETRWC(CLR_B, SET_ABD_F); KW_AM_RESET already zeroes every counter.
            TTI_MVMUL(p_setrwc::CLR_AB, 0, KW_AM_RESET, 0);
#else
            TTI_MVMUL(p_setrwc::CLR_A, 0, KW_AM_RESET, 0);
#endif
        }
    }
#ifndef KW_CLR_AB
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
#endif
}
#endif

ALWI void raw_walk() {
    reconfig_data_format(cb_weight, cb_act);  // SrcA = in1, SrcB = in0
    PACK((pack_reconfig_data_format(cb_partials)));
#if MM_FLAVOR == 1
    mm_no_mop_init_short(cb_act, cb_weight, false, sb_w, sb_h, block_k);
#else
    matmul_block_init(cb_act, cb_weight, /*transpose=*/0, sb_w, sb_h, block_k);
#endif
#if MM_FLAVOR == 4
    static_assert(sb_h == 1, "dest_advance: 1 x sb_w subblocks (reuse-A path) only");
    MATH((kw_set_addrmods()));
#endif
    PACK((llk_pack_reconfig_l1_acc(0)));
    for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
        const uint32_t inner = (kb == num_k_blocks - 1) ? last_block_k : block_k;
        cb_wait_front(cb_act, act_block_tiles);
        cb_wait_front(cb_weight, weight_block_tiles);
        if (kb == 1) {
            PACK((llk_pack_reconfig_l1_acc(1)));
        }
        uint32_t in0_sb_base = 0;
        for (uint32_t m = 0; m < in0_num_subblocks; ++m) {
            for (uint32_t n = 0; n < in1_num_subblocks; ++n) {
                tile_regs_acquire();
#if MM_FLAVOR == 4
                MATH((kw_pass_begin()));
#endif
#ifndef SKIP_COMPUTE
                uint32_t in0_index = in0_sb_base;
                uint32_t in1_index = n * sb_w;
                for (uint32_t k = 0; k < inner; ++k) {
#if MM_FLAVOR == 0
                    matmul_block(cb_act, cb_weight, in0_index, in1_index, 0, false, sb_w, sb_h, block_k);
#elif MM_FLAVOR == 1
                    matmul_block_no_mop(cb_act, cb_weight, in0_index, in1_index, 0, false, sb_w, sb_h, block_k);
#elif MM_FLAVOR == 2
                    UNPACK((llk_unpack_AB_matmul(cb_act, cb_weight, in0_index, in1_index, sb_w, sb_h, block_k)));
                    MATH((llk_math_matmul<MATH_FIDELITY, 0>(0, sb_w, sb_h)));
#elif MM_FLAVOR == 4
                    UNPACK((llk_unpack_AB_matmul(cb_act, cb_weight, in0_index, in1_index, sb_w, sb_h, block_k)));
                    MATH((kw_math_call()));
#else
                    // DIAGNOSTIC ONLY (wrong output): one dest-address write per call instead of per tile.
                    UNPACK((llk_unpack_AB_matmul(cb_act, cb_weight, in0_index, in1_index, sb_w, sb_h, block_k)));
                    MATH((math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0)));
                    for (uint32_t t = 0; t < sb_w; ++t) {
                        MATH((ckernel_template::run()));
                    }
                    MATH((TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F)));
#endif
                    in0_index++;
                    in1_index += block_n;
                }
#endif
                tile_regs_commit();
                tile_regs_wait();
#ifndef ABL_NOPACK
                for (uint32_t r = 0; r < sb_h; ++r) {
                    const uint32_t row_base = (m * sb_h + r) * block_n + n * sb_w;
                    for (uint32_t c = 0; c < sb_w; ++c) {
                        pack_tile<true>(r * sb_w + c, cb_partials, row_base + c);
                    }
                }
#endif
                tile_regs_release();
            }
            in0_sb_base += sb_h * block_k;
        }
        cb_pop_front(cb_act, act_block_tiles);
        cb_pop_front(cb_weight, weight_block_tiles);
    }
}

void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act, cb_weight, cb_out);
    CircularBuffer act_buf(cb_act), weight_buf(cb_weight), out_buf(cb_out), interm_buf(cb_partials);
    {
        DeviceZoneScopedN("kwalk");
#if KW_VARIANT == 0
        helper_walk<false>(act_buf, weight_buf, out_buf, interm_buf);
#elif KW_VARIANT == 1
        interm_buf.reserve_back(block_tiles);
        helper_walk<true>(act_buf, weight_buf, out_buf, interm_buf);
        interm_buf.push_back(block_tiles);
#elif KW_VARIANT == 2
        interm_buf.reserve_back(block_tiles);
        raw_walk();
        interm_buf.push_back(block_tiles);
#elif KW_VARIANT == 3
        // the graduation artifact: kw::matmul_accumulate_interm (kw_matmul_walk.hpp)
        static_assert(sb_h == 1);
        kw::matmul_accumulate_interm<sb_w, in0_num_subblocks, in1_num_subblocks, block_k, num_k_blocks, last_block_k>(
            cb_act, cb_weight, cb_partials);
#else
#error "unknown KW_VARIANT"
#endif
    }
    PACK((llk_pack_reconfig_l1_acc(0)));
    interm_buf.wait_front(block_tiles);
#ifdef CHECK_OUT
    // Bench epilogue: fp32 copy of the accumulated block to cb_out, tile row-major.
    reconfig_data_format_srca(cb_partials);
    pack_reconfig_data_format(cb_out);
    copy_init(cb_partials);
    for (uint32_t t = 0; t < block_tiles; ++t) {
        tile_regs_acquire();
        copy_tile(cb_partials, t, 0);
        tile_regs_commit();
        out_buf.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        out_buf.push_back(1);
    }
#endif
    interm_buf.pop_front(block_tiles);
}
