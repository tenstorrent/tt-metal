// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The shared expert's matmul cores' compute kernel (one kernel a column group).
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/se_defs.hpp"

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#ifdef SE_PREP
#include "api/compute/pack_untilize.h"
#endif
#include "api/debug/kernel_profiler.hpp"
#if defined(SE_GLU) && defined(TRISC_PACK)
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#if SE_GLU == 2
#include "ckernel_sfpu_situ_glu.h"
#elif SE_GLU == 3
#include "ckernel_sfpu_clamped_silu_glu.h"
#elif SE_GLU == 4
#include "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/swiglu_sfpu.h"
#endif
namespace ckernel::sfpu {
// (the model's activation constants, compile-time: SiTU's betas, the clamp limit, SwiGLU-OAI's alpha / limit; their
// defines come with the kernel source - se_engine3.glu_defines)
struct SeGluCfg {
    static constexpr float beta_gate = SE_GLU_BETA, beta_up = SE_GLU_LBETA;
    static constexpr float limit = SE_GLU_LIMIT, clamp_limit = SE_GLU_LIMIT, alpha = SE_GLU_ALPHA;
};
}  // namespace ckernel::sfpu
#endif
constexpr uint32_t MT = get_compile_time_arg_val(0), KB = get_compile_time_arg_val(1),
                   NSG = get_compile_time_arg_val(2);
constexpr uint32_t NPG = get_compile_time_arg_val(3), NCH = get_compile_time_arg_val(4),
                   HT = get_compile_time_arg_val(5);
constexpr uint32_t POMAX = get_compile_time_arg_val(6), KBD = get_compile_time_arg_val(7),
                   NSD = get_compile_time_arg_val(8);
constexpr uint32_t NPD = get_compile_time_arg_val(9), WSLOT = get_compile_time_arg_val(10),
                   PWDMAX = get_compile_time_arg_val(11);
constexpr uint32_t G = get_compile_time_arg_val(12), SPS = get_compile_time_arg_val(13);  // (shards, steps a shard)
constexpr uint32_t GX = get_compile_time_arg_val(14), HA = GX * MT * POMAX;  // (h-all: [column][MT][POMAX])
// Words [0, NC) - the per-group block (gate / down tables, h column offsets, HT) and the RS / prep tables' prefix every
// core of the group shares - are this kernel's common args (sent once a chip, not once a core); the rest of a core's
// tables are its runtime args from NC on. Indices stay absolute (the host's tables point at them).
constexpr uint32_t SE_NC = get_compile_time_arg_val(15);
FORCE_INLINE uint32_t se_arg(uint32_t i) {
    return i < SE_NC ? get_common_arg_val<uint32_t>(i) : get_arg_val<uint32_t>(i - SE_NC);
}
constexpr uint32_t XIN = 0, WIN = 1, HOWN = 2, HALL = 3, OUT = 4, ACC = 5;
// (runtime args: the column's gate / up pass widths [NPG], then down pass widths [NPD])
#ifdef SE_RTR
// the router's matmul at HiFi3 (the engine runs HiFi2: the routing picks need HiFi3) - matmul_block's LLK calls with
// the fidelity fixed
ALWI void rmm_init(uint32_t in0, uint32_t in1, uint32_t ct, uint32_t kt) {
    state_configure(in1, in0, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, 1, ct, 1, kt)));
    MATH((llk_math_matmul_init<MathFidelity::HiFi3, MM_THROTTLE>(in0, in1, 1, ct, 1)));
}
ALWI void rmm(uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1, uint32_t idst, uint32_t ct, uint32_t kt) {
    state_configure(in1, in0, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul(in0, in1, i0, i1, ct, 1, kt)));
    MATH((llk_math_matmul<MathFidelity::HiFi3, MM_THROTTLE>(idst, ct, 1)));
}
#endif
void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(XIN, WIN, HOWN);
#ifdef SE_RTR
    {  // the router's logits first (every chain starts here): my expert tiles x my MT rows over the chip's K slice
        DeviceZoneScopedN("SE-RTRMM");
        constexpr uint32_t etc = SE_ETC;  // (every core the same count; past the experts: unused tiles)
        {  // (a K chunk's ETC x MT partials in DEST, then fp32 into RO with the packer's L1 accumulation)
            reconfig_data_format(SE_RX, SE_RW);
            pack_reconfig_data_format(SE_RO);
            // (row groups of SE_RG rows: ETC x SE_RG tiles in DEST - all rows at once where they fit)
            rmm_init(SE_RW, SE_RX, SE_RG, SE_KC);
            cb_reserve_back(SE_RO, etc * SE_MTR);
            for (uint32_t rd = 0; rd < SE_KTO / SE_KC; ++rd) {
                cb_wait_front(SE_RX, SE_KC * SE_MTR);
                cb_wait_front(SE_RW, etc * SE_KC);
                PACK((llk_pack_reconfig_l1_acc(rd > 0 ? 1 : 0)));
                for (uint32_t g = 0; g < SE_MTR; g += SE_RG) {
                    tile_regs_acquire();
                    for (uint32_t e = 0; e < etc; ++e) {
                        for (uint32_t k = 0; k < SE_KC; ++k) {
                            rmm(SE_RW, SE_RX, e * SE_KC + k, k * SE_MTR + g, e * SE_RG, SE_RG, SE_KC);
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t e = 0; e < etc; ++e) {
                        for (uint32_t r = 0; r < SE_RG; ++r) {
                            pack_tile<true>(e * SE_RG + r, SE_RO, e * SE_MTR + g + r);
                        }
                    }
                    tile_regs_release();
                }
                cb_pop_front(SE_RX, SE_KC * SE_MTR);
                cb_pop_front(SE_RW, etc * SE_KC);
            }
            PACK((llk_pack_reconfig_l1_acc(0)));
            cb_push_back(SE_RO, etc * SE_MTR);
            reconfig_data_format(WIN, XIN);  // (back to gate / up's formats)
        }
    }
#endif
#ifdef SE_GLU
#if SE_GLU == 2
    PACK((SFPU_BINARY_INIT_FN_NO_ARGS(situ_glu, sfpu::situ_glu_init)));
#elif SE_GLU == 3
    PACK((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::clamped_silu_glu_init)));
#elif SE_GLU == 4
    PACK((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::swiglu_init)));
#else
    silu_tile_init_pack();
    PACK((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::MUL))));
#endif
#endif
    uint32_t opos = 0;  // (OUTF tiles pushed: its blocks end at multiples of 8 - the NCRISC reads it page by page,
                        // pack_tile does not wrap; the chunks' fins continue where the last one left it)
    for (uint32_t m = 0; m < NCH; ++m) {
        // gate / up: per pass all of K in DEST, the GLU, h packed once
        {
            DeviceZoneScopedN("SE-GU");
            if (NCH > 1 && m > 0) {
                reconfig_data_format(WIN, XIN);  // (the last chunk's down / RS left other formats)
            }
            cb_reserve_back(HOWN, MT * POMAX);
            // shard-major: shard k (own first, then as each lands) through every pass; a pass's fp32 partial is packed
            // onto its fixed ACC slot (MT x pw tiles at the pass's offset) with the packer's L1 accumulation - shard 0
            // writes it, shards 1 .. G - 2 add onto it in L1 - and only the last shard brings the sum back into DEST
            // (unpacked straight to DEST, fp32) under its own matmuls; then the GLU and h. Shard G - 2 pushes each
            // pass's slot as it is done (the write pointer then sits at the next pass's slot); the chunk's tail is
            // pushed and popped as padding, so every chunk starts at the region's start
            if (G > 1) {
                cb_reserve_back(ACC, SE_ACCP);
            }
            uint32_t apushed = 0;  // (ACC pages pushed this chunk - the pack offsets are relative to the write pointer)
            for (uint32_t k = 0; k < G; ++k) {
                uint32_t hoff = 0, loff = 0, aoff = 0;
                for (uint32_t pg = 0; pg < NPG; ++pg) {
                    const uint32_t pw = se_arg(pg), n = MT * pw;
                    tile_regs_acquire();
                    if (k + 1 == G && k > 0) {
                        cb_wait_front(ACC, n);
                        reconfig_data_format_srca(WIN, ACC);
                        copy_init(ACC);
                        for (uint32_t t = 0; t < n; ++t) {
                            copy_tile(ACC, t, t);
                        }
                        cb_pop_front(ACC, n);
                        reconfig_data_format_srca(ACC, WIN);
                    }
                    matmul_block_init(XIN, WIN, 0, pw, MT, KB);
                    for (uint32_t st = 0; st < SPS; ++st) {
                        cb_wait_front(XIN, MT * KB);
                        cb_wait_front(WIN, WSLOT);
                        for (uint32_t kk = 0; kk < KB; ++kk) {
                            matmul_block(XIN, WIN, kk, kk * pw, 0, false, pw, MT, KB);
                        }
                        cb_pop_front(XIN, MT * KB);
                        cb_pop_front(WIN, WSLOT);
                    }
                    tile_regs_commit();
                    if (k + 1 < G) {
                        tile_regs_wait();
                        pack_reconfig_data_format(ACC);
                        PACK((llk_pack_reconfig_l1_acc(k > 0 ? 1 : 0)));
                        for (uint32_t t = 0; t < n; ++t) {
                            pack_tile<true>(t, ACC, aoff + t - apushed);
                        }
                        PACK((llk_pack_reconfig_l1_acc(0)));
                        tile_regs_release();
                        aoff += n;
                        if (k + 2 == G) {  // (its sum is complete but for the last shard's: the last shard may read it)
                            cb_push_back(ACC, n);
                            apushed += n;
                        }
                        continue;
                    }
                    tile_regs_wait();
#ifdef SE_LAT
                    if (pg >= SE_NGU) {  // (a latent-down pass: no GLU, the chip's latent columns packed bf16 to LAT)
                        pack_reconfig_data_format(SE_LAT);
                        for (uint32_t r = 0; r < MT; ++r) {
                            for (uint32_t t = 0; t < pw; ++t) {
                                pack_tile<true>(r * pw + t, SE_LAT, r * SE_LWMAX + loff + t);
                            }
                        }
                        tile_regs_release();
                        loff += pw;
                        continue;
                    }
#endif
                    pack_reconfig_data_format(HOWN);
#ifdef SE_GLU
                    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
#if SE_GLU == 2
                    for (uint32_t t = 0; t < MT * pw; t += 2) {
                        PACK((SFPU_BINARY_CALL(
                            DST_SYNC_MODE,
                            DST_ACCUM_MODE,
                            calculate_situ_glu,
                            (DST_ACCUM_MODE, 8, sfpu::SeGluCfg),
                            t,
                            t + 1,
                            t,
                            VectorMode::RC)));
                    }
#elif SE_GLU == 3  // (silu(min(gate, 10)) x clamp(up, +-10): GLM-5.3's swiglu_limit, DeepSeek V4's)
                    for (uint32_t t = 0; t < MT * pw; t += 2) {
                        PACK((SFPU_BINARY_CALL(
                            DST_SYNC_MODE,
                            DST_ACCUM_MODE,
                            calculate_clamped_silu_glu,
                            (DST_ACCUM_MODE, 8, sfpu::SeGluCfg),
                            t,
                            t + 1,
                            t,
                            VectorMode::RC)));
                    }
#elif SE_GLU == 4  // (SwiGLU-OAI: (clamp(up, +-limit) + 1) x g sigmoid(alpha g), g = min(gate, limit) - MiniMax M3)
                    for (uint32_t t = 0; t < MT * pw; t += 2) {
                        PACK((SFPU_BINARY_CALL(
                            DST_SYNC_MODE,
                            DST_ACCUM_MODE,
                            calculate_swiglu,
                            (DST_ACCUM_MODE, 8, sfpu::SeGluCfg, true),
                            t,
                            t + 1,
                            t,
                            VectorMode::RC)));
                    }
#else
                    for (uint32_t t = 0; t < MT * pw; t += 2) {
                        silu_tile_pack(t);
                    }
                    for (uint32_t t = 0; t < MT * pw; t += 2) {
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
                    for (uint32_t r = 0; r < MT; ++r) {
                        for (uint32_t t = 0; t < pw; t += 2) {
                            pack_tile<true>(r * pw + t, HOWN, r * POMAX + hoff + t / 2);
                        }
                    }
                    tile_regs_release();
                    hoff += pw / 2;
                }
            }
            cb_push_back(HOWN, MT * POMAX);
            if (G > 1 && apushed < SE_ACCP) {  // (the region's tail: padding, so the next chunk starts at its start)
                cb_push_back(ACC, SE_ACCP - apushed);
                cb_wait_front(ACC, SE_ACCP - apushed);
                cb_pop_front(ACC, SE_ACCP - apushed);
            }
#ifdef SE_LAT
            cb_reserve_back(SE_LAT, SE_LATP);  // (the latent: the BRISC hands it to the ports, the NCRISC writes mine;
            cb_push_back(SE_LAT, SE_LATP);     // with chunks a chunk's LAT + LATZ, the next chunk's packed past them)
#endif
        }
#ifdef SE_RS
        {
            DeviceZoneScopedN("SE-DOWNRS");
            // (down + reduce-scatter) the host's program: pass i - all of K in DEST over h-all and the streamed W_down;
            // row r's tiles packed at base_r: into HOLD (fp32: an entry that adds an arrival here) or FINB (bf16: a
            // start entry, exchanged as is); hand e - entry e's start sub-block is complete (a FINC page: the BRISC's
            // hand-off); fin e - HOLD + the arrival(s) in fp32 DEST, packed bf16 into FINB (relay: handed off) or OUTF
            // (fin: y). Partials accumulate fp32 in L1; only the exchanged sub-blocks are bf16 (matmul_reduce_scatter's
            // rule).
            const uint32_t r0 = NPG + NPD + GX + 1, nops = se_arg(r0), pt0 = r0 + 1 + nops;
            const uint32_t et0 = pt0 + NPD * (1 + MT);
            uint32_t npass = 0, hin = 0;  // (h-all columns waited for: the first pass follows the exchange's rounds)
            if (NCH > 1 &&
                m > 0) {  // (the ports read the last chunk's FINB: mine to write again - the NCRISC's credit)
                cb_wait_front(SE_FFREE, 1);
                cb_pop_front(SE_FFREE, 1);
            }
            for (uint32_t o = 0; o < nops; ++o) {
                const uint32_t op = se_arg(r0 + 1 + o), kind = op & 0xFF, idx = op >> 8;
                if (kind == 0) {
                    const uint32_t pd = idx, pw = se_arg(pt0 + pd * (1 + MT));
                    reconfig_data_format(WIN, HALL);
                    matmul_block_init(HALL, WIN, 0, pw, MT, POMAX);
                    uint32_t c = 0;
                    tile_regs_acquire();
                    for (uint32_t i = 0; i < NSD; ++i) {
                        cb_wait_front(WIN, WSLOT);
                        for (uint32_t k = 0; k < KBD; ++k) {
                            const uint32_t kt = i * KBD + k;
                            while (kt >= se_arg(NPG + NPD + c + 1)) {
                                ++c;
                            }
                            if (c >= hin) {
                                hin = c + 1;
                                cb_wait_front(HALL, hin * MT * POMAX);
                            }
                            const uint32_t a0 = c * MT * POMAX + kt - se_arg(NPG + NPD + c);
                            matmul_block(HALL, WIN, a0, k * pw, 0, false, pw, MT, POMAX);
                        }
                        cb_pop_front(WIN, WSLOT);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t r = 0; r < MT; ++r) {
                        const uint32_t bw = se_arg(pt0 + pd * (1 + MT) + 1 + r);
                        const uint32_t ocb = (bw >> 31) ? SE_HOLD : SE_FINB, base = bw & 0x7FFFFFFF;
                        pack_reconfig_data_format(ocb);
                        for (uint32_t t = 0; t < pw; ++t) {
                            pack_tile<true>(r * pw + t, ocb, base + t);
                        }
                    }
                    tile_regs_release();
                    cb_reserve_back(SE_PSYNC, 1);  // (pass done: the UNPACK of a later finish may read its HOLD tiles)
                    cb_push_back(SE_PSYNC, 1);
                    ++npass;
                } else if (kind == 1) {
                    cb_reserve_back(SE_FINC, 1);
                    cb_push_back(SE_FINC, 1);
#ifdef SE_PREP
                } else if (kind == 3) {
                    // dispatch prep: my rows of the chip's owned tokens (x own / the latent) untilized into PO, a unit
                    // (n tiles of one row tile, contiguous in L1 at PU + off) at a time; the NCRISC writes PO's rows
                    // into the dispatch buffer. After the last pass: under the reduce-scatter's waits for arrivals.
                    DeviceZoneScopedN("SE-PREP");
#ifdef SE_PREP_LAST
                    // (with chunks: every chunk's units in the last chunk's op - no prep writes under the W re-reads)
                    const uint32_t nu = m + 1 == NCH ? se_arg(idx) : 0, u0 = 0, u1 = NCH * nu;
#else
                    const uint32_t nu = se_arg(idx), u0 = m * nu, u1 = (m + 1) * nu;
#endif
                    uint32_t pu0 = 0;
                    UNPACK((pu0 = get_local_cb_interface(SE_PU).fifo_rd_ptr));
                    uint32_t lrdy = 0;  // (chunks' latent arrivals waited for)
                    if (nu) {
                        reconfig_data_format_srca(SE_PU);  // (the last pass left srcA at the weights' format)
                        pack_reconfig_data_format(SE_PO);
                        pack_untilize_init<SE_PREP_BCT, SE_PREP_FULL>(SE_PU, SE_PO);
                    }
                    for (uint32_t u = u0; u < u1; ++u) {  // (chunk u / nu's units)
                        const uint32_t off = se_arg(idx + 1 + 2 * u), fl = se_arg(idx + 2 + 2 * u);
                        const uint32_t n = fl & 0xFFFF;
#ifdef SE_LRDY
                        if ((fl >> 16) &&
                            lrdy < u / nu + 1) {  // (a unit of the latent's arrivals: once they all landed)
                            lrdy = u / nu + 1;
                            cb_wait_front(SE_LRDY, lrdy);
                        }
#endif
                        cb_reserve_back(SE_PO, SE_PREP_FULL);
                        for (uint32_t c = 0; c * SE_PREP_BCT < n; ++c) {
                            UNPACK((
                                get_local_cb_interface(SE_PU).fifo_rd_ptr = pu0 + (off + c * SE_PREP_BCT * 2048) / 16));
                            pack_untilize_block<SE_PREP_BCT, SE_PREP_FULL>(SE_PU, 1, SE_PO, c);
                        }
                        cb_push_back(SE_PO, SE_PREP_FULL);
                    }
                    if (nu) {
                        pack_untilize_uninit(SE_PO);
                    }
                    UNPACK((get_local_cb_interface(SE_PU).fifo_rd_ptr = pu0));
                    (void)lrdy;
#ifdef SE_XFREE
                    cb_reserve_back(SE_XFREE, 1);  // (x own's ring: this chunk's slot read - the BRISC may reload it)
                    cb_push_back(SE_XFREE, 1);
#endif
#endif
                } else {
                    const uint32_t e6 = et0 + 6 * idx, cnt = se_arg(e6), mode = se_arg(e6 + 1);
                    const uint32_t sf = se_arg(e6 + 2), sb = se_arg(e6 + 3);
                    const uint32_t ho = se_arg(e6 + 4), fbo = se_arg(e6 + 5);
                    cb_wait_front(SE_PSYNC, npass);
                    // (sf / sb: the arrivals' zone offsets, tiles - every slot before them pushed with them)
                    if (mode & 1) {
                        cb_wait_front(SE_ARRF, sf + cnt);
                    }
                    if (mode & 2) {
                        cb_wait_front(SE_ARRB, sb + cnt);
                    }
                    const uint32_t a1 = (mode & 1) ? SE_ARRF : SE_ARRB, s1 = (mode & 1) ? sf : sb;
                    const uint32_t ocb = (mode & 4) ? SE_OUTF : SE_FINB;
                    pack_reconfig_data_format(ocb);
                    for (uint32_t i0 = 0, n = 0; i0 < cnt; i0 += n) {
                        const uint32_t lim = (mode & 4) ? SE_DT - opos % SE_DT : SE_DT;
                        n = cnt - i0 < lim ? cnt - i0 : lim;
                        if (mode & 4) {
                            opos += n;
                        }
                        tile_regs_acquire();
                        reconfig_data_format(SE_HOLD, a1);
                        add_init(SE_HOLD, a1);
                        for (uint32_t j = 0; j < n; ++j) {
                            add_tiles(SE_HOLD, a1, ho + i0 + j, s1 + i0 + j, j);
                        }
                        if ((mode & 3) == 3) {  // (the second arrival: DEST -> srcA at fp32 width, + the bwd tile)
                            reconfig_data_format(SE_HOLD, SE_ARRB);
                            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(SE_ARRB);
                            for (uint32_t j = 0; j < n; ++j) {
                                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(SE_ARRB, sb + i0 + j, j);
                            }
                        }
                        tile_regs_commit();
                        if (mode & 4) {
                            cb_reserve_back(SE_OUTF, n);
                        }
                        tile_regs_wait();
                        for (uint32_t j = 0; j < n; ++j) {
                            if (mode & 4) {
                                pack_tile(j, SE_OUTF);
                            } else {
                                pack_tile<true>(j, SE_FINB, fbo + i0 + j);
                            }
                        }
                        tile_regs_release();
                        if (mode & 4) {
                            cb_push_back(SE_OUTF, n);
                        }
                    }
                    if (!(mode & 4)) {
                        cb_reserve_back(SE_FINC, 1);
                        cb_push_back(SE_FINC, 1);
                    }
                }
            }
            cb_pop_front(HALL, HA);
            if constexpr (NCH > 1) {  // (the chunk's syncs and arrival zones popped: the next chunk's from their start)
                cb_wait_front(SE_PSYNC, npass);
                cb_pop_front(SE_PSYNC, npass);
                uint32_t mf = 0;
                for (uint32_t e = 0; e < SE_NENT; ++e) {
                    mf |= se_arg(et0 + 6 * e + 1);  // (the RS's entries: TP of them)
                }
                if (mf & 1) {
                    cb_wait_front(SE_ARRF, SE_ZF);
                    cb_pop_front(SE_ARRF, SE_ZF);
                }
                if (mf & 2) {
                    cb_wait_front(SE_ARRB, SE_ZB);
                    cb_pop_front(SE_ARRB, SE_ZB);
                }
            }
        }
#else
        DeviceZoneScopedN("SE-DOWN");
        // down: per pass all of K in DEST over h-all and the streamed W_down blocks, y packed once
        cb_wait_front(HALL, HA);
        for (uint32_t pd = 0; pd < NPD; ++pd) {
            const uint32_t pw = se_arg(NPG + pd);
            matmul_block_init(HALL, WIN, 0, pw, MT, POMAX);  // (h rows strided POMAX within a column's block)
            uint32_t c = 0;                                  // (the column block of K tile k: k in [h0[c], h0[c + 1]))
            tile_regs_acquire();
            for (uint32_t i = 0; i < NSD; ++i) {
                cb_wait_front(WIN, WSLOT);
                for (uint32_t k = 0; k < KBD; ++k) {
                    const uint32_t kt = i * KBD + k;
                    while (kt >= se_arg(NPG + NPD + c + 1)) {
                        ++c;
                    }
                    const uint32_t a0 = c * MT * POMAX + kt - se_arg(NPG + NPD + c);
                    matmul_block(HALL, WIN, a0, k * pw, 0, false, pw, MT, POMAX);
                }
                cb_pop_front(WIN, WSLOT);
            }
            tile_regs_commit();
            cb_reserve_back(OUT, MT * PWDMAX);
            tile_regs_wait();
            for (uint32_t r = 0; r < MT; ++r) {
                for (uint32_t t = 0; t < pw; ++t) {
                    pack_tile<true>(r * pw + t, OUT, r * PWDMAX + t);
                }
            }
            tile_regs_release();
            cb_push_back(OUT, MT * PWDMAX);
        }
        cb_pop_front(HALL, HA);
#endif
    }
}
