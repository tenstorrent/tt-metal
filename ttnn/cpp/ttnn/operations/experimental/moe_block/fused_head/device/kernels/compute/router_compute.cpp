// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The router cores' compute kernel (router_rows.hpp; under RT_ROUTE also the dispatch prep's route words,
// route_words.hpp).
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/router_addr.hpp"

#include <cstdint>
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/router_rows.hpp"
#ifdef RT_ROUTE
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/route_words.hpp"
#endif
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/topk.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_binary_sfpu.h"
#ifdef RT_SQRTSOFTPLUS
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/sqrt.h"
#endif
#include "api/debug/kernel_profiler.hpp"
constexpr uint32_t KT = get_compile_time_arg_val(0), ET = get_compile_time_arg_val(1), G = get_compile_time_arg_val(2);
constexpr uint32_t KBR = get_compile_time_arg_val(3), K = get_compile_time_arg_val(4);
constexpr uint32_t XR = 0, WR = 1, HOLD = 2, ARRF = 3, ARRB = 4, BIAS = 5, SC = 6, FINC = 7, PSYNC = 8;
constexpr uint32_t BEST = 9, IDX = 11;
constexpr uint32_t EM = rows_::em_of(ET);
#ifdef TRISC_MATH
namespace ckernel::sfpu {
// a score tile's top-K keys: the fp32 score with its low mantissa bits (EM) replaced by the expert id (the uint16 id
// tile two DEST tiles on): ttnn.topk's fused keys at fp32 - 9 bits instead of bf16, raw INT32 words
template <uint32_t M>
inline void rt_key() {
    TOPK_SFPENCC_ALL_LANES_ON();
    sfpi::vConstIntPrgm0 = M;
    for (int i = 0; i < 32; i++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 128);
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);  // the id (the high half of a uint16 in 32-bit DEST: junk)
        TTI_SFPOR(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TTI_SFPXOR(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);  // the score's low bits cleared
        TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}
}  // namespace ckernel::sfpu
#endif
// args: nops, ops (kind | entry << 8: 1 hand, 2 fin), per entry: arrivals to add (the fin: G - 1, from the zone's
// slots 0 ..; a hand-off: 0); then the top-K's done word. Every tile here is TRANSPOSED: logits^T [ET][32 tokens] - a
// token is a column.
void kernel_main() {
    // (the PACK -> MATH word: zero before anything - L1 scratch, PACK sets it only after the top-K)
    MATH((*reinterpret_cast<volatile uint32_t*>(RT_MSYNC) = 0));
    compute_kernel_hw_startup<SrcOrder::Reverse>(WR, XR, HOLD);
    pack_reconfig_data_format(HOLD);
#ifdef RT_ENG
    // (the matmul below is the router cores' own path: off here)
    cb_wait_front(12, 1);  // (the engine's partial logits are in my HOLD: RDY - UNPACK; PACK on word 112)
    PACK(({
        volatile uint32_t* hw =
            reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(1 + get_arg_val<uint32_t>(0) + G) + 112);
        while (*hw == 0) {
        }
        *hw = 0;  // (re-arm)
    }));
    if (false)
#endif
    {
        DeviceZoneScopedN("SE-RMMUL");
        for (uint32_t kb = 0; kb < KT / KBR; ++kb) {  // K-outer: every token tile's partial accumulated in L1 (fp32)
            cb_wait_front(XR, G * KBR);
            cb_wait_front(WR, KBR * ET);
            PACK((llk_pack_reconfig_l1_acc(kb > 0 ? 1 : 0)));
            for (uint32_t e0 = 0; e0 < ET;
                 ++e0) {                    // (logits^T [e0] x [G token tiles]: W_r^T x x^T; half-sync DEST: the
                constexpr uint32_t re = 1;  // packs of one block under the next block's math)
                matmul_block_init(WR, XR, 1, G, re, KBR);
                tile_regs_acquire();
                for (uint32_t kk = 0; kk < KBR; ++kk) {
                    matmul_block(WR, XR, e0 * KBR + kk, kk * G, 0, true, G, re, KBR);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t r = 0; r < re; ++r) {
                    for (uint32_t g = 0; g < G; ++g) {
                        pack_tile<true>(r * G + g, HOLD, g * ET + e0 + r);
                    }
                }
                tile_regs_release();
            }
            cb_pop_front(XR, G * KBR);
            cb_pop_front(WR, KBR * ET);
        }
    }
    PACK((llk_pack_reconfig_l1_acc(0)));
    DeviceZoneScopedN("SE-RFIN");
    cb_reserve_back(PSYNC, 1);
    cb_push_back(PSYNC, 1);
    cb_wait_front(PSYNC, 1);  // (UNPACK: the partials are in L1)
    cb_pop_front(PSYNC, 1);
    cb_wait_front(BIAS, ET);
    const uint32_t nops = get_arg_val<uint32_t>(0), et0 = 1 + nops;
    for (uint32_t o = 0; o < nops; ++o) {
        const uint32_t op = get_arg_val<uint32_t>(1 + o), kind = op & 0xFF, e = op >> 8;
        if (kind == 1) {
            cb_reserve_back(FINC, 1);
            cb_push_back(FINC, 1);
            continue;
        }
        const uint32_t na = get_arg_val<uint32_t>(et0 + e);  // (the one fin: the owner's - G - 1 sources)
        cb_reserve_back(SC, ET);
        cb_wait_front(IDX, ET);
        pack_reconfig_data_format(HOLD);
        // the sources' partials in landing order (ARRB: the slot): all but the last summed into my block's HOLD as they
        // land (fp32 on the SFPU, packed back in place), the last in the passes below - after the last arrival only its
        // add is left
        uint32_t last = 0;
        for (uint32_t a = 0; a < na; ++a) {
            cb_wait_front(ARRB, 1);
            const uint32_t sl = read_tile_value(ARRB, 0, 0);
            cb_pop_front(ARRB, 1);
            if (a + 1 == na) {
                last = sl;
                break;
            }
            for (uint32_t i0 = 0; i0 < ET; i0 += 2) {
                const uint32_t n = ET - i0 < 2 ? ET - i0 : 2;
                tile_regs_acquire();
                reconfig_data_format_srca(HOLD);
                copy_init(HOLD);
                for (uint32_t j = 0; j < n; ++j) {
                    copy_tile(HOLD, e * ET + i0 + j, j);
                }
                reconfig_data_format_srca(HOLD, ARRF);
                copy_init(ARRF);
                for (uint32_t j = 0; j < n; ++j) {
                    copy_tile(ARRF, sl * ET + i0 + j, 2 + j);
                }
                add_binary_tile_init();
                for (uint32_t j = 0; j < n; ++j) {
                    add_binary_tile(j, 2 + j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < n; ++j) {
                    pack_tile<true>(j, HOLD, e * ET + i0 + j);
                }
                tile_regs_release();
            }
            cb_reserve_back(PSYNC, 1);  // (PACK -> UNPACK: the sums are in HOLD before the next pass reads them)
            cb_push_back(PSYNC, 1);
            cb_wait_front(PSYNC, 1);
            cb_pop_front(PSYNC, 1);
        }
        {
            DeviceZoneScopedN("SE-RFARR");
        }  // (a mark: every arrival in - the fin from here to the top-K)
        // (all in DEST at fp32: the partials, the bias unpacked straight to DEST, the adds on the SFPU - the FPU's
        // srcA / srcB hold TF32, a 10-bit mantissa: the sums and the bias add through them moved scores by up to ~1e-3
        // and the picks of close experts with them)
        // (a score tile a pass: DEST 0 its score, 1 its key, 2 / 3 the operands - the score is kept: packed back over
        // its logits in HOLD, where the joint rows read the picks' weights exactly; the key's low bits are the id)
        for (uint32_t i = 0; i < ET; ++i) {
            tile_regs_acquire();
            reconfig_data_format_srca(HOLD);
            copy_init(HOLD);
            copy_tile(HOLD, e * ET + i, 0);
            if (na) {  // (+ the last source's partial)
                reconfig_data_format_srca(HOLD, ARRF);
                copy_init(ARRF);
                copy_tile(ARRF, last * ET + i, 2);
                add_binary_tile_init();
                add_binary_tile(0, 2, 0);
            }
            {  // the owner: scores = sigmoid(logits) (RT_SQRTSOFTPLUS: sqrt(softplus)), biased = scores + bias (bias^T:
               // per row)
#ifdef RT_SQRTSOFTPLUS
                softplus_tile_init();  // (INP_FLOAT32: the fp32 polynomial, as stock ttnn.softplus on fp32; beta 1,
                                       // threshold 20)
                softplus_tile(0, 0x3F800000u, 0x3F800000u, 0x41A00000u);
                sqrt_tile_init();
                sqrt_tile(0);
#else
                sigmoid_tile_init();
                sigmoid_tile(0);
#endif
                reconfig_data_format_srca(ARRF, BIAS);
                copy_init(BIAS);
                copy_tile(BIAS, i, 2);
                add_binary_tile_init();
                add_binary_tile(0, 2, 1);
                // -> the top-K's keys: the expert ids into the low mantissa bits (rt_key: the id tile two on, DEST 3)
                reconfig_data_format_srca(BIAS, IDX);
                copy_init(IDX);
                copy_tile(IDX, i, 3);
                MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::rt_key<EM>, 1, VectorMode::RC_custom)));
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(1, SC, i);
            pack_tile<true>(0, HOLD, e * ET + i);
            tile_regs_release();
        }
        cb_push_back(SC, ET);
        // the owner's 32 tokens' top-K on the SFPU (ttnn.topk's bitonic network), each token a column, fp32 straight
        // to DEST (no srcA rounding): the running top K of every column in DEST tile 0, each next 32 experts copied
        // into tile 1 as keys (rt_key: the expert id in the low mantissa bits - one word a datum, no id tiles riding
        // the swaps); the 64-wide column sorted into K-runs (bitonic phases 0 .. log K - 1), then merged run against
        // run down to one sorted K-run on top (log(64 / K) merge + rebuild rounds) - no full 64-wide sorts, nothing
        // packed until the end
        DeviceZoneScopedN("SE-RTOPK");
        cb_wait_front(SC, ET);
        // (the network's run KS: K's power of two, at least 4 - K 6 sorts the top 8, the rows take its first K)
        constexpr uint32_t KS = K <= 4 ? 4 : K <= 8 ? 8 : K <= 16 ? 16 : 32;
        constexpr uint32_t LK = KS == 8 ? 3 : KS == 16 ? 4 : KS == 32 ? 5 : 2, NM = 6 - LK;
        constexpr auto TIE = ckernel::TopkTieOrder::Unset;
        tile_regs_acquire();
        if constexpr (ET == 1) {  // (one tile of candidates: no running top K is parked in DEST tile 2 - the
                                  // column's other half is -inf keys, below every score; K <= 32 never reaches them)
            fill_tile_init();
            fill_tile_int<DataFormat::Int32>(0, 0xFF800000u);
        }
        reconfig_data_format_srca(SC);
        copy_init(SC);
        ckernel::topk_tile_init<true, false>();
        for (uint32_t t = 0; t < ET; t += 2) {
            const bool odd = t + 1 == ET;     // (a last tile alone: against the running top K, a full round)
            const bool fold = t > 0 && !odd;  // (a pair after the first: its run folded into the running top K)
            if (odd) {
                if constexpr (ET > 1) {
                    copy_dest_values<DataFormat::Int32>(2, 0);  // (the running top K, parked at t - 2)
                }
                copy_tile(SC, t, 1);
            } else {
                copy_tile(SC, t, 0);
                copy_tile(SC, t + 1, 1);
            }
            ckernel::topk_local_sort<false, DST_ACCUM_MODE, true, false, TIE>(0, 0, LK - 1);
            for (uint32_t m = 0; m + 1 < NM; ++m) {
                ckernel::topk_merge<false, false, DST_ACCUM_MODE, true, false, TIE>(0, m, KS);
                ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, TIE>(0, 0, m, KS, LK, false);
            }
            if (!fold) {  // the top K of the 64 a column on top: tile 0's run, descending
                ckernel::topk_merge<false, false, DST_ACCUM_MODE, true, false, TIE>(0, NM - 1, KS);
                ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, TIE>(0, 0, NM - 1, KS, LK, true);
            } else {  // the pair's top K into tile 1's run, ascending; the running top K (descending) into tile 0 ->
                      // one bitonic merge (max on top) + rebuild of K
                ckernel::topk_merge<true, false, DST_ACCUM_MODE, true, false, TIE>(0, NM - 1, KS);
                ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, TIE>(0, 0, NM - 1, KS, LK, false);
                copy_dest_values<DataFormat::Int32>(2, 0);
                ckernel::topk_merge<false, false, DST_ACCUM_MODE, true, false, TIE>(0, NM - 1, KS);
                ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, TIE>(0, 0, NM - 1, KS, LK, true);
            }
            if (t + 2 < ET) {
                copy_dest_values<DataFormat::Int32>(
                    0, 2);  // (parked in DEST tile 2: fused keys leave the id tiles unused)
            }
        }
        tile_regs_commit();
        cb_reserve_back(BEST, 1);
        tile_regs_wait();
        pack_reconfig_data_format(BEST);
        pack_tile(0, BEST);
        tile_regs_release();
        cb_push_back(BEST, 1);
        const uint32_t t1 = et0 + G, words = get_arg_val<uint32_t>(t1);
        PACK(({  // (the last best pair packed: the joint rows can go - BRISC and NCRISC told)
            *reinterpret_cast<volatile uint32_t*>(words) = 1;
            *reinterpret_cast<volatile uint32_t*>(words + 64) = 1;
        }));
        {  // the joint rows' last two shares (tokens 16 .. 23 the MATH thread, 24 .. 31 the PACK thread; soft float):
           // the BEST page is in L1 once PACK passed its push - PACK tells MATH (RT_MSYNC)
            const uint32_t bvc = get_arg_val<uint32_t>(t1 + 1), olc = get_arg_val<uint32_t>(t1 + 2);
            const uint32_t scl = get_arg_val<uint32_t>(t1 + 3), sv = get_arg_val<uint32_t>(t1 + 4);
            PACK(({
                *reinterpret_cast<volatile uint32_t*>(RT_MSYNC) = 1;
                rows_::make<K, ET>(bvc, sv, olc, scl, 24, 32);
                *reinterpret_cast<volatile uint32_t*>(words + 48) = 1;
            }));
            MATH(({
                volatile uint32_t* ms = reinterpret_cast<volatile uint32_t*>(RT_MSYNC);
                while (*ms == 0) {
                }
                rows_::make<K, ET>(bvc, sv, olc, scl, 16, 24);
                *reinterpret_cast<volatile uint32_t*>(words + 32) = 1;
            }));
        }
#ifdef RT_ROUTE
        // dispatch prep: the route words of my 32 owned tokens from the joint rows (BRISC's word 80: all shares
        // written), into L1 for the BRISC's write (word 96); off the routing chain (the UNPACK thread is idle here)
        const uint32_t rl1 = get_arg_val<uint32_t>(t1 + 5), tok0 = get_arg_val<uint32_t>(t1 + 6);
        const uint32_t ol1 = get_arg_val<uint32_t>(t1 + 2);
        UNPACK(({
            volatile uint32_t* rd = reinterpret_cast<volatile uint32_t*>(words + 80);
            while (*rd == 0) {
            }
            *rd = 0;
            volatile uint32_t* o = reinterpret_cast<volatile uint32_t*>(rl1);
            for (uint32_t c = 0; c < 32; ++c) {
                rw_::word(o, c, reinterpret_cast<volatile const uint16_t*>(ol1 + c * 4 * K), tok0 + c);
            }
            *reinterpret_cast<volatile uint32_t*>(words + 96) = 1;
        }));
#endif
    }
}
