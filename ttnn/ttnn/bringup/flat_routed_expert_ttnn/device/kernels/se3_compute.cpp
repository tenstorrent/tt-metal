// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// M-split big-M streamed-expert compute (TRISC): like se2_compute.cpp, but each core owns NP gate/up pairs (and its
// M-group's rows only), so a sub-block is MT tiles of the group's rows and the weight slice is NP pairs wide:
//   gate/up(v): DST[m * 2NP + {gate_p, up_p}] += x(v) [MT x KBLK per K-block, streamed] @ in1 gu block
//               [KBLK x 2NP] over all NK_GU K-blocks; h_p = silu(gate_p) * up_p on the PACK thread's SFPU, packed to
//               H_LOCAL as [MT x NP].
//   down(v):    DST[m * PCD + c] += h_all(v) [MT x KT_D] @ in1 down blocks [KBLK_D x PCD], in row passes of RT_D
//               tiles (RT_D * PCD <= 8); h_all is row-major within K-blocks of 8 tiles.
// A weight block is freed (popped) only after its last use, gu(e, S - 1) / d(e, S - 1); since the ring only pops from
// the front and gu(e + 1, 0) runs before d(e, S - 1) (pipelined order), fully used blocks are popped lazily in ring
// order, which lets the next expert's blocks land while the current one finishes.
//
// CT: 0 KBLK, 1 MT, 2 NK_GU, 3 KT_D, 4 KBLK_D, 5 NUM_EXPERTS, 6 S, 7 SLOT_TILES, 8 PCD, 9 RT_D, 10 NP, 11 PIPE (S = 1
//     only: the weight stream comes in consumption order gu(0), gu(1), d(0), gu(2), d(1), ... and every block is freed
//     right after its (only) use, in stream order), 12 RING_SLOTS (in1 ring capacity in blocks; 0: one expert)
// Define SE_GU_ONLY for the spatially pipelined variant: gate/up only (KT_D = 0, down runs on other cores).
#include "se_meta.hpp"
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#if defined(SE_GU_L1ACC) || defined(SE_RH_W) || defined(SE_RH_X)
#include "api/compute/tile_move_copy.h"
#endif
#include "api/compute/reconfig_data_format.h"
// Gate/up activation (SE_ACT), on the PACK thread's SFPU over the raw gate / up accumulators in DST:
//   0 SiLU-GLU silu(g) * u (default)          1 SwiGLU-OAI (clamp(u, +-7) + 1) * g' sigmoid(1.702 g'), g' = min(g, 7)
//   2 SiTU-GLU 4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25) (Kimi K3)
//   3 clamped SiLU-GLU silu(min(g, 10)) * clamp(u, +-10) (DeepSeek V4)    4 GeGLU gelu_tanh(g) * u (Gemma 4)
#ifndef SE_ACT
#define SE_ACT 0
#endif
#ifdef TRISC_PACK
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#ifdef SE_HG  // (half-width gate / up tiles: the face-wise GLU, se_glu_faces.hpp)
#include "se_glu_faces.hpp"
#endif
#if defined(SE_RH_W) || defined(SE_RH_Q)  // (the h options: routing weight, fp8 q-dq, half merge)
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "se_rh.hpp"
#endif
#if SE_ACT == 1
#include "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/swiglu_sfpu.h"
#elif SE_ACT == 2
#include "ckernel_sfpu_situ_glu.h"
#elif SE_ACT == 3
#include "ckernel_sfpu_clamped_silu_glu.h"
#endif
#endif
#if SE_ACT == 4
#include "api/compute/eltwise_unary/gelu.h"
#endif
#if defined(SE_ZONES) || defined(SE_WAITZ)
#include "tools/profiler/kernel_profiler.hpp"
#endif

// (perf probe SE_NO_MATH: no matmul - every CB handshake, unpack-side pops and pack stay; garbage outputs)
#ifdef SE_NO_MATH
#define matmul_block(...) ((void)0)
#endif

constexpr uint32_t x_cb = tt::CBIndex::c_0;
constexpr uint32_t in1_cb = tt::CBIndex::c_1;
constexpr uint32_t h_all_cb = tt::CBIndex::c_2;
constexpr uint32_t h_local_cb = tt::CBIndex::c_3;
constexpr uint32_t out_cb = tt::CBIndex::c_16;
constexpr uint32_t part_cb = tt::CBIndex::c_5;  // SE_GU_L1ACC: bf16 gate/up partials, MT x 2 NP tiles
// SE_XMT: row tiles of an x block (all M-groups' rows; this core computes MT of them from row GROUP * MT, RT 0)
#ifndef SE_XMT
#define SE_XMT get_compile_time_arg_val(1)
#endif
constexpr uint32_t xmt = SE_XMT;
uint32_t x_row0 = 0;  // first tile of this core's rows in an x block

constexpr uint32_t kblk = get_compile_time_arg_val(0);
constexpr uint32_t mt = get_compile_time_arg_val(1);
constexpr uint32_t nk_gu = get_compile_time_arg_val(2);
constexpr uint32_t kt_d = get_compile_time_arg_val(3);
constexpr uint32_t kblk_d = get_compile_time_arg_val(4);
constexpr uint32_t num_experts = get_compile_time_arg_val(5);
constexpr uint32_t sub = get_compile_time_arg_val(6);
constexpr uint32_t slot = get_compile_time_arg_val(7);
constexpr uint32_t pcd = get_compile_time_arg_val(8);
constexpr uint32_t rt_d = get_compile_time_arg_val(9);
constexpr uint32_t np = get_compile_time_arg_val(10);
#ifdef SE_HG_NU  // 16-column units: NU weight tiles a row, HPC h tiles a row (hg_lead: this core's first unit is odd)
constexpr uint32_t gw = SE_HG_NU, hpc = SE_HPC;
#else
constexpr uint32_t gw = 2 * np;  // gate/up block width
constexpr uint32_t hpc = np;     // h tiles a row
#endif
uint32_t hg_lead = 0;
// SE_HG: the DST tile (of a row's gw) that holds local h tile j: its first unit's (lead: unit 0 is a right half)
FORCE_INLINE uint32_t h_slot(uint32_t j) { return j == 0 ? 0 : 2 * j - hg_lead; }

#if defined(SE_RH_W) || defined(SE_RH_Q)
// The h options (config routed_weight / fp8_act_quant; the plan: one 32-row sub-block, gw gate/up DEST tiles + tile
// SE_RH_WS), on the fp32 DEST between the activation and h's pack (se_rh.hpp):
//   SE_RH_W  the sub-block's routing-weight tile (CB SE_RH_RW_CB, Float32, unpacked straight into DEST tile WS before
//            the K loop; se5_recv.cpp builds it) multiplies every h tile of the row
//   SE_RH_Q  each whole h tile is fp8-quantized in place (scratch: tile WS)
//   SE_RH_X  odd 16-column units: a split h tile's halves meet on the core whose LAST h tile is a left half (tail):
//            the lead core (first h tile a right half) packs its half (Float32) to SE_RH_XOUT_CB, its receiver sends it
//            to the tail core's SE_RH_XIN_CB; the tail core packs its own half to SE_RH_HOWN_CB and, after the
//            sub-block, merges both into one DEST tile, quantizes it and packs it as its h tile hpc - 1
constexpr uint32_t rh_ws = SE_RH_WS;
static_assert(rh_ws >= gw && rh_ws < 4, "the h options: the weight / scratch tile after the gate/up tiles");
uint32_t rh_lead = 0, rh_tail = 0;
#ifdef SE_RH_W
constexpr uint32_t rh_rw_cb = SE_RH_RW_CB;
#endif
#ifdef SE_RH_X
constexpr uint32_t rh_hown_cb = SE_RH_HOWN_CB, rh_xin_cb = SE_RH_XIN_CB, rh_xout_cb = SE_RH_XOUT_CB;
#endif

// SFPU steps at a runtime DEST tile (0..2) of the pack thread's section
FORCE_INLINE void rh_wmul_at(uint32_t t) {
#ifdef SE_RH_W
    if (t == 0) {
        PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, rh_wmul, (0, rh_ws), 0, VectorMode::RC_custom)));
    } else if (t == 1) {
        PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, rh_wmul, (1, rh_ws), 0, VectorMode::RC_custom)));
    } else {
        PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, rh_wmul, (2, rh_ws), 0, VectorMode::RC_custom)));
    }
#endif
}
FORCE_INLINE void rh_quant_at(uint32_t t) {
#ifdef SE_RH_Q
    if (t == 0) {
        PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, rh_quant, (0, rh_ws), 0, VectorMode::RC_custom)));
    } else if (t == 1) {
        PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, rh_quant, (1, rh_ws), 0, VectorMode::RC_custom)));
    } else {
        PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, rh_quant, (2, rh_ws), 0, VectorMode::RC_custom)));
    }
#endif
}

// (math) the sub-block's routing-weight tile into DEST tile WS, in the acquire the K loop then accumulates into
FORCE_INLINE void rh_acquire_with_weight(uint32_t rows) {
#ifdef SE_RH_W
    cb_wait_front(rh_rw_cb, 1);
    reconfig_data_format_srca(in1_cb, rh_rw_cb);
    copy_tile_to_dst_init_short(rh_rw_cb);
    tile_regs_acquire();
    copy_tile(rh_rw_cb, 0, rh_ws);
    cb_pop_front(rh_rw_cb, 1);
    reconfig_data_format_srca(rh_rw_cb, in1_cb);
    matmul_block_init(x_cb, in1_cb, SE_WT, gw, rows, kblk);
#else
    tile_regs_acquire();
#endif
}

// (pack, after the activation) the routing weight on every h tile, then the fp8 q-dq of the whole ones
FORCE_INLINE void rh_dst() {
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
    for (uint32_t j = 0; j < hpc; ++j) {
        rh_wmul_at(h_slot(j));
    }
    for (uint32_t j = rh_lead; j + rh_tail < hpc; ++j) {
        rh_quant_at(h_slot(j));
    }
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
}

#ifdef SE_RH_X
// (tail core, after the sub-block's pack) its own half (CB HOWN) and the partner's (CB XIN) into DEST tiles 0 / 1,
// merged (right faces from the partner), quantized, packed as h tile hpc - 1 of the h_local block
FORCE_INLINE void rh_phase2() {
    cb_wait_front(rh_hown_cb, 1);
    cb_wait_front(rh_xin_cb, 1);
    reconfig_data_format_srca(in1_cb, rh_hown_cb);
    copy_tile_to_dst_init_short(rh_hown_cb);
    tile_regs_acquire();
    copy_tile(rh_hown_cb, 0, 0);
    copy_tile(rh_xin_cb, 0, 1);
    tile_regs_commit();
    cb_pop_front(rh_hown_cb, 1);
    cb_pop_front(rh_xin_cb, 1);
    reconfig_data_format_srca(rh_hown_cb, in1_cb);
    matmul_block_init(x_cb, in1_cb, SE_WT, gw, mt, kblk);  // (the caller records the matmul state)
    tile_regs_wait();
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
    PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, rh_merge, (0, 1), 0, VectorMode::RC_custom)));
    PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, rh_quant, (0, rh_ws), 0, VectorMode::RC_custom)));
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    pack_reconfig_data_format(out_cb, h_local_cb);
    pack_tile<true>(0, h_local_cb, (hpc - 1) * mt);
    pack_reconfig_data_format(h_local_cb, out_cb);
    tile_regs_release();
}
#endif
#endif
constexpr bool gu_pipe = get_compile_time_arg_val(11) != 0;
constexpr uint32_t ring_ct = get_compile_time_arg_val(12);
constexpr uint32_t nk_dd = kt_d / kblk_d;
constexpr uint32_t bpe = nk_gu + nk_dd;             // weight blocks per expert
constexpr uint32_t ring = ring_ct ? ring_ct : bpe;  // in1 ring capacity (blocks)
constexpr uint32_t h_all_tiles = kt_d * mt;
constexpr uint32_t hk = 8;  // h_all K-block width (the gather layout)
#ifndef SE_DST_TILES
#define SE_DST_TILES 8  // DST half: 8 bf16 tiles, 4 with fp32 accumulation (SE_DST_TILES 4)
#endif
#ifdef SE_GU_RP
constexpr uint32_t gu_dst_tiles = SE_GU_RP * gw;  // row passes: one pass's rows in DST
#else
constexpr uint32_t gu_dst_tiles = mt * gw;
#endif
static_assert(
    gu_dst_tiles <= SE_DST_TILES && rt_d * pcd <= 8 && mt % rt_d == 0 && kt_d % kblk_d == 0 && kt_d % hk == 0);

uint32_t popped = 0;               // weight blocks popped from the in1 ring (absolute block index of its front)
uint32_t gu_done = 0, d_done = 0;  // experts whose gate/up (down) blocks have all had their last use
uint32_t gu_last = 0, d_last = 0;  // blocks of expert gu_done (d_done) already through their last use
uint32_t consumed = 0;             // PIPE: blocks through their last use (in stream order)

// Stream position of expert e's gate/up block b and down block j.
FORCE_INLINE uint32_t pos_gu(uint32_t e, uint32_t b) {
    return gu_pipe ? (e ? nk_gu + (e - 1) * bpe : 0) + b : e * bpe + b;
}
FORCE_INLINE uint32_t pos_d(uint32_t e, uint32_t j) {
    return gu_pipe ? nk_gu + e * bpe + (e + 1 < num_experts ? nk_gu : 0) + j
                   : e * bpe + nk_gu + j;  // d(e) follows gu(e + 1)
}
uint32_t cur_ct = gw, cur_rt = mt;

FORCE_INLINE void set_mm(uint32_t ct, uint32_t rt) {
    if (ct != cur_ct || rt != cur_rt) {
        matmul_block_init(h_all_cb, in1_cb, SE_WT, ct, rt, kblk);
        cur_ct = ct;
        cur_rt = rt;
    }
}

// Tile index of absolute weight block a relative to the ring's read pointer, after waiting for it to land. Block a
// sits in slot a % BPE, and the unpacker does not wrap at the end of the ring, so a block past the wrap point gets a
// negative index (the unpacker's address arithmetic is modulo 2^32).
FORCE_INLINE uint32_t wblock(uint32_t a) {
    cb_wait_front(in1_cb, (a - popped + 1) * slot);
    return static_cast<uint32_t>(static_cast<int32_t>(a % ring) - static_cast<int32_t>(popped % ring)) * slot;
}

#ifdef SE_DYN
// Dynamic schedule (se_dyn.hpp): load l's block b is stream block l * NK_GU + b and sits in ring slot
// region(l) * NK_GU + b; blocks are popped by count right after their last use (a pinned load retires after later
// loads), and the receiver grants slots by count, so the read pointer is just popped % ring.
uint32_t d_popped = 0;
FORCE_INLINE uint32_t wblock_dyn(uint32_t p, uint32_t phys) {
    if (p + 1 > d_popped) {  // else it has landed already (popped <= pushed)
        cb_wait_front(in1_cb, (p + 1 - d_popped) * slot);
    }
    return static_cast<uint32_t>(static_cast<int32_t>(phys) - static_cast<int32_t>(d_popped % ring)) * slot;
}
#endif

FORCE_INLINE void pop_used() {
    if constexpr (gu_pipe) {
        while (popped < consumed) {
            cb_pop_front(in1_cb, slot);
            ++popped;
        }
        return;
    }
    while (true) {
        const uint32_t e = popped / bpe, j = popped % bpe;
        const bool used = j < nk_gu ? (gu_done > e || (gu_done == e && j < gu_last))
                                    : (d_done > e || (d_done == e && j - nk_gu < d_last));
        if (!used) {
            break;
        }
        cb_pop_front(in1_cb, slot);
        ++popped;
    }
}

// Gate/up activation on DST tiles [0, N) (gate t, up t + 1 -> t), on the PACK thread's SFPU (after tile_regs_wait).
// SE_HG: tiles t, t + 1 hold [gate | up] of the h tile's left / right 16 columns: both activations land in tile t.
FORCE_INLINE void act_dst(uint32_t n) {
#ifndef SE_NO_ACT
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
#ifdef SE_HG
    // (one call site: two inlined copies of the SFPU body in this loop make the SFPI compiler's rvtt_synth_renumber
    // pass ICE; unit u of a row goes to its h tile's slot, half (u + lead) & 1)
    for (uint32_t t = 0; t < n; ++t) {
        const uint32_t u = t % gw, out = t - u + h_slot((u + hg_lead) / 2);
        PACK((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_glu_faces,
            (DST_ACCUM_MODE, SE_ACT, 8),
            t,
            t,
            out,
            VectorMode::C,
            (u + hg_lead) & 1u)));
    }
#elif SE_ACT == 1 || SE_ACT == 2 || SE_ACT == 3
    for (uint32_t t = 0; t < n; t += 2) {  // one binary op: gate t, up t + 1 -> t
#if SE_ACT == 1
        PACK((ckernel::llk_math_eltwise_binary_sfpu_swiglu<DST_ACCUM_MODE>(t, t + 1, t)));
#elif SE_ACT == 2
        PACK((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_situ_glu,
            (DST_ACCUM_MODE, 8, sfpu::SituGluConfigKimi),
            t,
            t + 1,
            t,
            VectorMode::RC)));
#else
        PACK((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_clamped_silu_glu,
            (DST_ACCUM_MODE, 8, sfpu::ClampedSiluGluConfigDsV4),
            t,
            t + 1,
            t,
            VectorMode::RC)));
#endif
    }
#else
    for (uint32_t t = 0; t < n; t += 2) {  // unary gate activation, then gate * up
#if SE_ACT == 4
        gelu_tanh_tile_pack(t);
#else
        silu_tile_pack(t);
#endif
    }
    for (uint32_t t = 0; t < n; t += 2) {
        PACK((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary_mul,
            (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE),
            t,
            t + 1,
            t,
            VectorMode::RC)));
    }
#endif
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
#endif
}

// ROWS: row tiles of the sub-block that hold tokens (dynamic counts: the last sub-block of an entry may hold fewer;
// the rest of the block is neither computed nor packed, its rows are never written out).
FORCE_INLINE void gate_up(uint32_t v, uint32_t e, bool last, uint32_t ph0 = 0, uint32_t rows = mt) {
    {
#ifdef SE_ZONES
        DeviceZoneScopedN("SE_GU_MM");
#endif
        if (rows != cur_rt || cur_ct != gw) {  // (x is in0 here: re-init against x_cb)
            matmul_block_init(x_cb, in1_cb, SE_WT, gw, rows, kblk);
            cur_ct = gw;
            cur_rt = rows;
        }
#ifdef SE_GU_L1ACC
        // K in passes of SE_GU_L1ACC K-blocks: each pass accumulates in (bf16) DST, then is packed into the bf16
        // partials CB with packer L1 accumulation (the first pass overwrites); a long bf16 DST accumulation over
        // all of K biases the result upwards (ties-away rounding: ~1.085 norm gain at K 7168, ~1.009 with passes of
        // 8 K tiles, test_dest_gain_probe.py). The sum returns to DST for the activation below.
        constexpr uint32_t grp = SE_GU_L1ACC;
        cb_reserve_back(part_cb, mt * gw);
        pack_reconfig_data_format(out_cb, part_cb);
#elif defined(SE_RH_W)
        rh_acquire_with_weight(rows);  // (the routing-weight tile into DEST tile WS first; the matmul re-initialized)
        cur_ct = gw;
        cur_rt = rows;
#else
        tile_regs_acquire();
#endif
        for (uint32_t b = 0; b < nk_gu; ++b) {
#ifdef SE_GU_L1ACC
            if (b % grp == 0) {
                tile_regs_acquire();
            }
#endif
#ifdef SE_WAITZ
            if (v == SE_WAITZ) {
                {
                    DeviceZoneScopedN("W_X");
                    cb_wait_front(x_cb, xmt * kblk);
                }
                {
                    DeviceZoneScopedN("W_W");
                    wblock(pos_gu(e, b));
                }
            }
#endif
            cb_wait_front(x_cb, xmt * kblk);
#ifdef SE_DYN
            const uint32_t w = wblock_dyn(e * nk_gu + b, ph0 + b);
#else
            const uint32_t w = wblock(pos_gu(e, b));
#endif
            for (uint32_t k = 0; k < kblk; ++k) {
                matmul_block(x_cb, in1_cb, x_row0 + k, w + k * gw, 0, SE_WT, gw, rows, kblk);
            }
            cb_pop_front(x_cb, xmt * kblk);
            if (last) {  // free the block for the next expert as soon as possible
#ifdef SE_DYN
                cb_pop_front(in1_cb, slot);
                ++d_popped;
#else
                gu_last = b + 1;
                ++consumed;
                pop_used();
#endif
            }
#ifdef SE_GU_L1ACC
            if (b % grp == grp - 1 || b + 1 == nk_gu) {
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_l1_acc(b >= grp ? 1 : 0);
                for (uint32_t i = 0; i < rows * gw; ++i) {
                    pack_tile<true>(i, part_cb, i);
                }
                // the next pass re-accumulates the same slots: its read-modify-write must see this pass's writes
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK));
                tile_regs_release();
            }
#endif
        }
#ifdef SE_GU_L1ACC
        pack_reconfig_l1_acc(0);
        cb_push_back(part_cb, mt * gw);
        // the sum back into DST (unpack srcA: the weights' format -> the bf16 partials), then the matmul state again
        reconfig_data_format_srca(in1_cb, part_cb);
        copy_tile_to_dst_init_short(part_cb);
        tile_regs_acquire();
        cb_wait_front(part_cb, mt * gw);
        for (uint32_t i = 0; i < rows * gw; ++i) {
            copy_tile(part_cb, i, i);
        }
        cb_pop_front(part_cb, mt * gw);
        tile_regs_commit();
        reconfig_data_format_srca(part_cb, in1_cb);
        pack_reconfig_data_format(part_cb, out_cb);
        matmul_block_init(x_cb, in1_cb, SE_WT, gw, rows, kblk);
        cur_ct = gw;
        cur_rt = rows;
#else
        tile_regs_commit();
#endif
    }
#ifndef SE_DYN
    if (last) {
        ++gu_done;
        gu_last = 0;
        pop_used();
    }
#endif
    {
#ifdef SE_ZONES
        DeviceZoneScopedN("SE_GU_ACT_PACK");
#endif
        cb_reserve_back(h_local_cb, mt * hpc);
        tile_regs_wait();
#ifndef SE_NO_ACT
        act_dst(rows * gw);
#endif
#if defined(SE_RH_W) || defined(SE_RH_Q)
        rh_dst();  // (one row tile: mt 1)
#endif
        pack_reconfig_data_format(out_cb, h_local_cb);
#if defined(SE_GU_ONLY) && defined(SE_RH_X)
        // split h tiles: the lead core's first (right half) to the partner, the tail core's last (left half) kept for
        // the merge (both Float32); the whole ones into h_local at their [p][m] place (mt 1)
        for (uint32_t p = 0; p < hpc; ++p) {
            const bool send = p == 0 && rh_lead, keep = p + 1 == hpc && rh_tail;
            if (send || keep) {
                const uint32_t cb = send ? rh_xout_cb : rh_hown_cb;
                cb_reserve_back(cb, 1);
                pack_reconfig_data_format(h_local_cb, cb);
                pack_tile(h_slot(p), cb);
                pack_reconfig_data_format(cb, h_local_cb);
                cb_push_back(cb, 1);
            } else {
                pack_tile<true>(h_slot(p), h_local_cb, p * mt);
            }
        }
#elif defined(SE_GU_ONLY)
        for (uint32_t p = 0; p < hpc; ++p) {     // [p][m]: K-tile-major, one contiguous run on the down cores
            for (uint32_t m = 0; m < mt; ++m) {  // (rows past ROWS: whatever DST holds, never used)
                pack_tile(m < rows ? m * gw + h_slot(p) : h_slot(p), h_local_cb);
            }
        }
#else
        for (uint32_t t = 0; t < mt * gw; t += 2) {
            pack_tile(t, h_local_cb);  // [m][p]
        }
#endif
        pack_reconfig_data_format(h_local_cb, out_cb);
        tile_regs_release();
#ifdef SE_RH_X
        if (rh_tail) {  // the split last h tile: merged with the partner's half, quantized, packed as tile hpc - 1
            rh_phase2();
            cur_ct = gw;
            cur_rt = mt;
        }
#endif
        cb_push_back(h_local_cb, mt * hpc);
    }
}

#if defined(SE_GU_RP) && defined(SE_DYN) && defined(SE_GU_ONLY)
uint32_t x_front = 0;  // x blocks popped (the ring's read slot = x_front % SE_XSLOTS)
// Row passes (SE_GU_RP row tiles each): the sub-block keeps MT rows for the whole pipeline (x delivery, h exchange,
// down passes cost ~3.5 us per sub-block on K3 whatever its rows), but DST (4 fp32 tiles) only holds RP x 2 NP
// gate/up tiles, so the K loop runs once per row pass over the same x blocks, which stay in the x ring until the
// final pass (host: X_SLOTS >= NK_GU) and the same resident weights (popped in the final pass).
FORCE_INLINE void gate_up_rp(uint32_t e, bool last, uint32_t ph0, uint32_t rows) {
    constexpr uint32_t rp = SE_GU_RP;
    constexpr uint32_t xblk = xmt * kblk;
    constexpr uint32_t xs = SE_XSLOTS;  // x ring blocks: block b of this sub-block sits in slot (front + b) % xs
    const uint32_t npass = (rows + rp - 1) / rp;
#ifdef SE_WAITZ  // (sampled wait zones: one sub-block in 16)
    static uint32_t calls = 0;
    const bool wz = calls++ % 16 == SE_WAITZ;
    if (wz) {
        DeviceZoneScopedN("W_H");
        cb_reserve_back(h_local_cb, mt * hpc);
    }
#endif
    cb_reserve_back(h_local_cb, mt * hpc);
    for (uint32_t ps = 0; ps < npass; ++ps) {
        const uint32_t r0 = ps * rp, rr = rows - r0 < rp ? rows - r0 : rp;
        const bool fin = ps + 1 == npass;
        {
#ifdef SE_ZONES
            DeviceZoneScopedN("SE_GU_MM");
#endif
            if (rr != cur_rt || cur_ct != gw) {
                matmul_block_init(x_cb, in1_cb, SE_WT, gw, rr, kblk);
                cur_ct = gw;
                cur_rt = rr;
            }
#ifdef SE_WAITZ
            if (wz) {
                DeviceZoneScopedN("GU_ACQ");
                tile_regs_acquire();
            } else {
                tile_regs_acquire();
            }
#else
            tile_regs_acquire();
#endif
            for (uint32_t b = 0; b < nk_gu; ++b) {
                uint32_t xo = 0;
#ifdef SE_WAITZ
                if (wz) {
                    DeviceZoneScopedN("W_X");
                    cb_wait_front(x_cb, fin ? xblk : (b + 1) * xblk);
                }
                if (wz) {
                    DeviceZoneScopedN("W_W");
                    wblock_dyn(e * nk_gu + b, ph0 + b);
                }
#endif
                if (fin) {  // the final pass consumes the blocks: block b is at the front
                    cb_wait_front(x_cb, xblk);
                } else {  // (unpacker indices do not wrap: a block past the ring end gets a negative offset)
                    cb_wait_front(x_cb, (b + 1) * xblk);
                    const uint32_t f = x_front % xs;
                    xo = static_cast<uint32_t>(static_cast<int32_t>((f + b) % xs) - static_cast<int32_t>(f)) * xblk;
                }
                const uint32_t w = wblock_dyn(e * nk_gu + b, ph0 + b);
                for (uint32_t k = 0; k < kblk; ++k) {
                    matmul_block(x_cb, in1_cb, xo + x_row0 + r0 * kblk + k, w + k * gw, 0, SE_WT, gw, rr, kblk);
                }
                if (fin) {
                    cb_pop_front(x_cb, xblk);
                    ++x_front;
                    if (last) {
                        cb_pop_front(in1_cb, slot);
                        ++d_popped;
                    }
                }
            }
            tile_regs_commit();
        }
        tile_regs_wait();
        act_dst(rr * gw);
        pack_reconfig_data_format(out_cb, h_local_cb);
        for (uint32_t p = 0; p < hpc; ++p) {  // [p][m] as gate_up: this pass's rows r0.. of each K tile
            for (uint32_t m = 0; m < rr; ++m) {
                pack_tile<true>(m * gw + h_slot(p), h_local_cb, p * mt + r0 + m);
            }
        }
        pack_reconfig_data_format(h_local_cb, out_cb);
        tile_regs_release();
    }
    cb_push_back(h_local_cb, mt * hpc);  // (rows past ROWS: never written, never used)
}
#endif

FORCE_INLINE void down(uint32_t v) {
    const uint32_t e = v / sub;
    const bool last = v % sub == sub - 1;
    {
#ifdef SE_ZONES
        DeviceZoneScopedN("SE_H_WAIT");
#endif
        cb_wait_front(h_all_cb, h_all_tiles);
    }
    {
#ifdef SE_ZONES
        DeviceZoneScopedN("SE_DOWN3");
#endif
        set_mm(pcd, rt_d);
        cb_reserve_back(out_cb, mt * pcd);
        for (uint32_t r = 0; r < mt; r += rt_d) {
            tile_regs_acquire();
            uint32_t w = 0;
            for (uint32_t kk = 0; kk < kt_d; ++kk) {
                if (kk % kblk_d == 0) {
                    w = wblock(pos_d(e, kk / kblk_d));
                }
                matmul_block(
                    h_all_cb,
                    in1_cb,
                    (kk / hk) * mt * hk + r * hk + kk % hk,
                    w + (kk % kblk_d) * pcd,
                    0,
                    SE_WT,
                    pcd,
                    rt_d,
                    hk);
                if (last && r + rt_d == mt && kk % kblk_d == kblk_d - 1) {
                    d_last = kk / kblk_d + 1;
                    ++consumed;
                    pop_used();
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < rt_d * pcd; ++i) {
                pack_tile<true>(i, out_cb, r * pcd + i);  // row-major [MT x PCD]
            }
            tile_regs_release();
        }
        cb_pop_front(h_all_cb, h_all_tiles);
        cb_push_back(out_cb, mt * pcd);
    }
    if (last) {
        ++d_done;
        d_last = 0;
        pop_used();
    }
}

void kernel_main() {
    constexpr uint32_t num_v = num_experts * sub;
    const uint32_t grp = get_arg_val<uint32_t>(0);  // M-group (RT 0)
#ifdef SE_HG_NU
    hg_lead = get_arg_val<uint32_t>(1);  // (RT 1: this core's first unit is odd)
#endif
#if defined(SE_RH_W) || defined(SE_RH_Q)
#ifdef SE_RH_X
    rh_lead = hg_lead;
    rh_tail = get_arg_val<uint32_t>(2);  // (RT 2: this core's last h tile is a left half)
#endif
#ifdef SE_GU_RP
#error "the h options run one gate/up pass per sub-block (the plan's mt 1, no row passes)"
#endif
#endif
    x_row0 = grp * mt * kblk;
    compute_kernel_hw_startup<SrcOrder::Reverse>(x_cb, in1_cb, out_cb);
#ifdef SE_PACK_SRND  // stochastic rounding in the packer (the default rounds bf16 -> bfp8 ties away from zero)
    UNPACK((_llk_unpack_configure_stoch_rnd_<StochRndType::Pack>()));
#endif
    matmul_block_init(x_cb, in1_cb, SE_WT, gw, mt, kblk);
#ifndef SE_NO_ACT
#if defined(SE_HG)  // (every face-wise GLU uses SiTU's sigmoid: tanh_init, no vConstFloatPrgm for the sigmoid)
    PACK((SFPU_BINARY_INIT_FN_NO_ARGS(situ_glu, sfpu::situ_glu_init)));
#elif SE_ACT == 1
    PACK((ckernel::llk_math_eltwise_binary_sfpu_swiglu_init()));
#elif SE_ACT == 2
    PACK((SFPU_BINARY_INIT_FN_NO_ARGS(situ_glu, sfpu::situ_glu_init)));
#elif SE_ACT == 3
    PACK((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::clamped_silu_glu_init)));
#else
#if SE_ACT == 4
    gelu_tanh_tile_init_pack();
#else
    silu_tile_init_pack();
#endif
    PACK((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::MUL))));
#endif
#endif
    // Pipelined order: gate/up of v + 1 before down of v hides the h exchange of v.
#ifdef SE_GU_ONLY
#ifdef SE_DYN
    // Dynamic counts: CB 6 holds the schedule (se_meta.hpp layout, from se5_recv.cpp): per entry its sub-blocks and
    // its gate/up load (stream place), that load's ring region and whether this is the load's last use.
    cb_wait_front(tt::CBIndex::c_6, 1);
    const uint32_t n_act = read_tile_value(tt::CBIndex::c_6, 0, 0);
    for (uint32_t a = 0, v = 0; a < n_act; ++a) {
        const uint32_t subs = read_tile_value(tt::CBIndex::c_6, 0, SE_META_SUBS + a);
        const uint32_t gu = read_tile_value(tt::CBIndex::c_6, 0, SE_META_GU + a);
        const uint32_t ld = gu & 0xFF, ph0 = ((gu >> 8) & 0xFF) * nk_gu;
        const bool last_use = (gu >> 16) & 1;
        const uint32_t lmt = read_tile_value(tt::CBIndex::c_6, 0, SE_META_LMT + a);
        for (uint32_t s = 0; s < subs; ++s, ++v) {
            // this group's row tiles holding tokens (none: one garbage row, the down cores never write it out)
            const uint32_t lr = s + 1 == subs ? lmt : xmt;
            const uint32_t rows = lr > grp * mt ? (lr - grp * mt < mt ? lr - grp * mt : mt) : 1;
#ifdef SE_GU_RP
            gate_up_rp(ld, last_use && s + 1 == subs, ph0, rows);
#else
            gate_up(v, ld, last_use && s + 1 == subs, ph0, rows);
#endif
        }
    }
    cb_pop_front(tt::CBIndex::c_6, 1);
#else
    for (uint32_t v = 0; v < num_v; ++v) {
        gate_up(v, v / sub, v % sub == sub - 1);
    }
#endif
#else
    gate_up(0, 0, sub == 1);
    for (uint32_t v = 0; v < num_v; ++v) {
        if (v + 1 < num_v) {
            gate_up(v + 1, (v + 1) / sub, (v + 1) % sub == sub - 1);
        }
        down(v);
    }
#endif
}
