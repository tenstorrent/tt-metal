// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Flat expert gate/up compute (TRISC): each gate/up core owns NP gate/up pairs of all MT row tiles of a sub-block:
//   DST[m * 2NP + {gate_p, up_p}] += x [MT x KBLK per K-block, streamed] @ in1 gu block [KBLK x 2NP] over all NK_GU
//   K-blocks; h_p = act(gate_p, up_p) on the PACK thread's SFPU, packed to H_LOCAL as [NP x MT] (K-tile-major).
// The down projection runs on the down cores (down_compute.cpp). The schedule (CB 6, schedule_meta.hpp) names each
// entry's weight load and ring region; a load's blocks are popped right after its last use.
//
// CT: 0 KBLK, 1 MT, 2 NK_GU, 3 SLOT_TILES, 4 NP, 5 RING_SLOTS (in1 ring capacity in blocks; 0: one load)
#include "schedule_meta.hpp"
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
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
#ifdef SE_HG  // (half-width gate / up tiles: the face-wise GLU, glu_faces.hpp)
#include "glu_faces.hpp"
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

constexpr uint32_t x_cb = tt::CBIndex::c_0;
constexpr uint32_t in1_cb = tt::CBIndex::c_1;
constexpr uint32_t h_local_cb = tt::CBIndex::c_3;
constexpr uint32_t out_cb = tt::CBIndex::c_16;
constexpr uint32_t kblk = get_compile_time_arg_val(0);
constexpr uint32_t mt = get_compile_time_arg_val(1);
constexpr uint32_t nk_gu = get_compile_time_arg_val(2);
constexpr uint32_t slot = get_compile_time_arg_val(3);
constexpr uint32_t np = get_compile_time_arg_val(4);
#ifdef SE_HG_NU  // 16-column units: NU weight tiles a row, HPC h tiles a row (hg_lead: this core's first unit is odd)
constexpr uint32_t gw = SE_HG_NU, hpc = SE_HPC;
#else
constexpr uint32_t gw = 2 * np;  // gate/up block width
constexpr uint32_t hpc = np;     // h tiles a row
#endif
uint32_t hg_lead = 0;
// SE_HG: the DST tile (of a row's gw) that holds local h tile j: its first unit's (lead: unit 0 is a right half)
FORCE_INLINE uint32_t h_slot(uint32_t j) { return j == 0 ? 0 : 2 * j - hg_lead; }
constexpr uint32_t ring_ct = get_compile_time_arg_val(5);
constexpr uint32_t ring = ring_ct ? ring_ct : nk_gu;  // in1 ring capacity (blocks)
#ifndef SE_DST_TILES
#define SE_DST_TILES 8  // DST half: 8 bf16 tiles, 4 with fp32 accumulation (SE_DST_TILES 4)
#endif
#ifdef SE_GU_RP
constexpr uint32_t gu_dst_tiles = SE_GU_RP * gw;  // row passes: one pass's rows in DST
#else
constexpr uint32_t gu_dst_tiles = mt * gw;
#endif
static_assert(gu_dst_tiles <= SE_DST_TILES);

uint32_t cur_ct = gw, cur_rt = mt;  // the matmul's current block shape

// Dynamic schedule (expert_schedule.hpp): load l's block b is stream block l * NK_GU + b and sits in ring slot
// region(l) * NK_GU + b; blocks are popped by count right after their last use (a pinned load retires after later
// loads), and the receiver grants slots by count, so the read pointer is just popped % ring.
uint32_t d_popped = 0;
FORCE_INLINE uint32_t wblock_dyn(uint32_t p, uint32_t phys) {
    if (p + 1 > d_popped) {  // else it has landed already (popped <= pushed)
        cb_wait_front(in1_cb, (p + 1 - d_popped) * slot);
    }
    return static_cast<uint32_t>(static_cast<int32_t>(phys) - static_cast<int32_t>(d_popped % ring)) * slot;
}

// Gate/up activation on DST tiles [0, N) (gate t, up t + 1 -> t), on the PACK thread's SFPU (after tile_regs_wait).
// SE_HG: tiles t, t + 1 hold [gate | up] of the h tile's left / right 16 columns: both activations land in tile t.
FORCE_INLINE void act_dst(uint32_t n) {
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
}

// ROWS: row tiles of the sub-block that hold tokens (dynamic counts: the last sub-block of an entry may hold fewer;
// the rest of the block is neither computed nor packed, its rows are never written out).
FORCE_INLINE void gate_up(uint32_t e, bool last, uint32_t ph0 = 0, uint32_t rows = mt) {
    {
        if (rows != cur_rt || cur_ct != gw) {  // (x is in0 here: re-init against x_cb)
            matmul_block_init(x_cb, in1_cb, false, gw, rows, kblk);
            cur_ct = gw;
            cur_rt = rows;
        }
        tile_regs_acquire();
        for (uint32_t b = 0; b < nk_gu; ++b) {
            cb_wait_front(x_cb, mt * kblk);
            const uint32_t w = wblock_dyn(e * nk_gu + b, ph0 + b);
            for (uint32_t k = 0; k < kblk; ++k) {
                matmul_block(x_cb, in1_cb, k, w + k * gw, 0, false, gw, rows, kblk);
            }
            cb_pop_front(x_cb, mt * kblk);
            if (last) {  // free the block for the next expert as soon as possible
                cb_pop_front(in1_cb, slot);
                ++d_popped;
            }
        }
        tile_regs_commit();
    }
    {
        cb_reserve_back(h_local_cb, mt * hpc);
        tile_regs_wait();
        act_dst(rows * gw);
        pack_reconfig_data_format(out_cb, h_local_cb);
        for (uint32_t p = 0; p < hpc; ++p) {     // [p][m]: K-tile-major, one contiguous run on the down cores
            for (uint32_t m = 0; m < mt; ++m) {  // (rows past ROWS: whatever DST holds, never used)
                pack_tile(m < rows ? m * gw + h_slot(p) : h_slot(p), h_local_cb);
            }
        }
        pack_reconfig_data_format(h_local_cb, out_cb);
        tile_regs_release();
        cb_push_back(h_local_cb, mt * hpc);
    }
}

#ifdef SE_GU_RP
uint32_t x_front = 0;  // x blocks popped (the ring's read slot = x_front % SE_XSLOTS)
// Row passes (SE_GU_RP row tiles each): the sub-block keeps MT rows for the whole pipeline (x delivery, h exchange,
// down passes cost ~3.5 us per sub-block on K3 whatever its rows), but DST (4 fp32 tiles) only holds RP x 2 NP
// gate/up tiles, so the K loop runs once per row pass over the same x blocks, which stay in the x ring until the
// final pass (host: X_SLOTS >= NK_GU) and the same resident weights (popped in the final pass).
FORCE_INLINE void gate_up_rp(uint32_t e, bool last, uint32_t ph0, uint32_t rows) {
    constexpr uint32_t rp = SE_GU_RP;
    constexpr uint32_t xblk = mt * kblk;
    constexpr uint32_t xs = SE_XSLOTS;  // x ring blocks: block b of this sub-block sits in slot (front + b) % xs
    const uint32_t npass = (rows + rp - 1) / rp;
    cb_reserve_back(h_local_cb, mt * hpc);
    for (uint32_t ps = 0; ps < npass; ++ps) {
        const uint32_t r0 = ps * rp, rr = rows - r0 < rp ? rows - r0 : rp;
        const bool fin = ps + 1 == npass;
        {
            if (rr != cur_rt || cur_ct != gw) {
                matmul_block_init(x_cb, in1_cb, false, gw, rr, kblk);
                cur_ct = gw;
                cur_rt = rr;
            }
            tile_regs_acquire();
            for (uint32_t b = 0; b < nk_gu; ++b) {
                uint32_t xo = 0;
                if (fin) {  // the final pass consumes the blocks: block b is at the front
                    cb_wait_front(x_cb, xblk);
                } else {  // (unpacker indices do not wrap: a block past the ring end gets a negative offset)
                    cb_wait_front(x_cb, (b + 1) * xblk);
                    const uint32_t f = x_front % xs;
                    xo = static_cast<uint32_t>(static_cast<int32_t>((f + b) % xs) - static_cast<int32_t>(f)) * xblk;
                }
                const uint32_t w = wblock_dyn(e * nk_gu + b, ph0 + b);
                for (uint32_t k = 0; k < kblk; ++k) {
                    matmul_block(x_cb, in1_cb, xo + r0 * kblk + k, w + k * gw, 0, false, gw, rr, kblk);
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

void kernel_main() {
#ifdef SE_HG_NU
    hg_lead = get_arg_val<uint32_t>(0);  // (RT 0: this core's first unit is odd)
#endif
    compute_kernel_hw_startup<SrcOrder::Reverse>(x_cb, in1_cb, out_cb);
#ifdef SE_PACK_SRND  // stochastic rounding in the packer (the default rounds bf16 -> bfp8 ties away from zero)
    UNPACK((_llk_unpack_configure_stoch_rnd_<StochRndType::Pack>()));
#endif
    matmul_block_init(x_cb, in1_cb, false, gw, mt, kblk);
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
    // Dynamic counts: CB 6 holds the schedule (schedule_meta.hpp layout, from gate_up_core_reader.cpp): per entry its
    // sub-blocks and its gate/up load (stream place), that load's ring region and whether this is the load's last use.
    cb_wait_front(tt::CBIndex::c_6, 1);
    const uint32_t n_act = read_tile_value(tt::CBIndex::c_6, 0, 0);
    for (uint32_t a = 0; a < n_act; ++a) {
        const uint32_t subs = read_tile_value(tt::CBIndex::c_6, 0, SE_META_SUBS + a);
        const uint32_t gu = read_tile_value(tt::CBIndex::c_6, 0, SE_META_GU + a);
        const uint32_t ld = gu & 0xFF, ph0 = ((gu >> 8) & 0xFF) * nk_gu;
        const bool last_use = (gu >> 16) & 1;
        const uint32_t lmt = read_tile_value(tt::CBIndex::c_6, 0, SE_META_LMT + a);
        for (uint32_t s = 0; s < subs; ++s) {
            // row tiles holding tokens (none: one garbage row, the down cores never write it out)
            const uint32_t lr = s + 1 == subs ? lmt : mt;
            const uint32_t rows = lr ? (lr < mt ? lr : mt) : 1;
#ifdef SE_GU_RP
            gate_up_rp(ld, last_use && s + 1 == subs, ph0, rows);
#else
            gate_up(ld, last_use && s + 1 == subs, ph0, rows);
#endif
        }
    }
    cb_pop_front(tt::CBIndex::c_6, 1);
}
