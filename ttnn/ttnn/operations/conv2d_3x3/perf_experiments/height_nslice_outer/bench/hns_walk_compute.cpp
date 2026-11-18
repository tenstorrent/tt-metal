// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated bench (single core, compute only, every operand resident in L1): the per-core compute of conv2d_3x3's
// HEIGHT K-blocked walk vs an N-slice-outer whole-K walk, at semantic res2.fuse_conv.1's shape (bm 5, 9 K-blocks of
// bk 8 = one tap each, center taps (kw = 1) as copied input tiles, N 8, 2 M-blocks per core), bias + ReLU, LoFi,
// fp32 DEST, half-sync DEST (4 fp32 tiles). Bench-only (perf_experiments/height_nslice_outer).
//
// mode 0 (baseline, the op today): per M-block, compute_kernel_lib::matmul_block exactly as conv2d_3x3 instantiates it
//   (Interm target, packer L1 accumulation, TileRowMajor, ShortAfterPreKBlock, WaitAndPopPerKBlock x2, uniform 1 x 4
//   subblocks), PreKBlockFn = the weight round's push plus the K-block's lean fast tilize (none on the 3 center
//   K-blocks, which the In0SourceFn reads from the center CB), then add_bias_bcast_rows with ReLU into the output.
// mode 1 (N-slice-outer): per M-block, a prelude tilizes the 6 non-center K-blocks into a resident [6][bm][bk] in0
//   (one lean tilize call per K-block, as the baseline's hook), then per N-slice of s columns: one weight round of
//   K x s tiles, and per output subblock (CT list of (r0, h, c0, w)) one whole-K DEST accumulation over the 9 K-blocks
//   (6 from in0, 3 from the center CB; row stride bk in both), then
//     bias_mode 0 (pass):  fp32 pack into [bm, N] partials; after the M-block the baseline's bias pass;
//     bias_mode 1 (fold):  + bias in DEST (add_reuse_dest DEST_TO_SRCA from a row-broadcast bias block built once),
//                          ReLU pack straight into the output; matmul re-init per subblock;
//     bias_mode 2 (slice): fp32 pack into a [bm, s] slice partial; after the slice its bias pass into the output;
//     bias_mode 3 (fold, tf32): as 1, with the ALU's SrcA format switched to fp32 around the add, so DEST moves into
//     SrcA
//                          as tf32, the precision at which the baseline's bias pass unpacks its fp32 partials.
//   Odd slices walk the subblock list backwards, so a slice starts at the shape the previous one ended on (mode 2:
//   every slice walks it forwards).
//
// RAW LLK / HELPER BYPASS (mode 1): compute_kernel_lib::matmul_block runs one MatmulBlockShape per call over a
// contiguous [bm, K] in0 and a [K, N] in1 block, waits / pops its in0 per K-block, and packs with L1 accumulation per
// K-block. The N-slice walk needs a whole-K DEST accumulation per subblock whose in0 K-blocks come from two CBs (the
// tilized taps and the copied center taps), a shorter last row group, and a pack (or bias add) per subblock at an
// absolute output offset. Same per-tile matmul LLK calls as the helper.

#include <cstdint>
#include <type_traits>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "../op_copy/kernels/conv2d_3x3_lean_tilize.hpp"

namespace ckl = compute_kernel_lib;

namespace {

constexpr uint32_t CB_IN0 = 0;    // bf8 tilize target (mode 0: one K-block; mode 1: the 6 tilized K-blocks)
constexpr uint32_t CB_W = 1;      // bf8 weights, resident shard of every round
constexpr uint32_t CB_RM = 2;     // bf16 row-major im2col rows of one K-block (resident; every tilized K-block)
constexpr uint32_t CB_CTR = 3;    // bf8 center K-blocks [3][bm][bk] (resident)
constexpr uint32_t CB_BIAS = 4;   // bf8 bias tiles (row 0 = bias), resident
constexpr uint32_t CB_PART = 5;   // fp32 partials
constexpr uint32_t CB_OUT = 6;    // bf8 output shard [m_iters * bm, N]
constexpr uint32_t CB_BIASF = 7;  // bf8 row-broadcast bias (fold)

template <uint32_t bm, uint32_t bk, uint32_t N, bool center, uint32_t tw>
struct KBlockHook {
    DataflowBuffer* rm;
    DataflowBuffer* in0;
    ALWI void operator()(uint32_t block, uint32_t /*num_k_blocks*/, bool /*is_last*/) const {
        // The weight writer's round (resident shard: the push only moves the CB pointers).
        cb_reserve_back(CB_W, bk * N);
        cb_push_back(CB_W, bk * N);
        if (center && block % 3 == 1) {
            // Center K-block: the reader's copied input tiles, nothing to tilize (In0SourceFn reads CB_CTR).
            cb_reserve_back(CB_CTR, bm * bk);
            cb_push_back(CB_CTR, bm * bk);
            return;
        }
        pack_reconfig_l1_acc(0);
        cb_reserve_back(CB_RM, bm * bk);
        cb_push_back(CB_RM, bm * bk);
        {
            MaybeDeviceZoneScope("compute_tilize");
            conv2d_3x3::lean_tilize::tilize<tw, CB_RM, CB_IN0>(*rm, *in0, bm * (bk / tw));
        }
    }
};

template <bool center>
struct CenterIn0 {
    ALWI uint32_t operator()(uint32_t block, uint32_t in0_cb_id) const {
        return center && block % 3 == 1 ? CB_CTR : in0_cb_id;
    }
};

struct Sub {
    uint32_t r0, h, c0, w;
};

}  // namespace

void kernel_main() {
    constexpr uint32_t bm = get_compile_time_arg_val(0);
    constexpr uint32_t bk = get_compile_time_arg_val(1);
    constexpr uint32_t N = get_compile_time_arg_val(2);
    constexpr uint32_t nkb = get_compile_time_arg_val(3);
    constexpr uint32_t m_iters = get_compile_time_arg_val(4);
    constexpr uint32_t mode = get_compile_time_arg_val(5);
    constexpr uint32_t s = get_compile_time_arg_val(6);
    constexpr uint32_t bias_mode = get_compile_time_arg_val(7);
    constexpr bool wide = get_compile_time_arg_val(8) != 0;
    // center: the sem layout (9 K-blocks of one tap, kw = 1 K-blocks from CB_CTR); else every K-block is tilized.
    constexpr bool center = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t tw = get_compile_time_arg_val(10);    // tilize block width (one tap)
    constexpr uint32_t sb_h = get_compile_time_arg_val(11);  // the K-blocked walk's _out_subblock
    constexpr uint32_t sb_w = get_compile_time_arg_val(12);
    constexpr uint32_t tilized_kb = center ? 6 : nkb;
    constexpr uint32_t tblocks = bm * (bk / tw);
    constexpr uint32_t K = bk * nkb;
    static_assert(N % s == 0 && bk % tw == 0 && (!center || nkb == 9));

    compute_kernel_hw_startup(CB_RM, CB_IN0);

    DataflowBuffer rm(CB_RM);
    DataflowBuffer in0_dfb(CB_IN0);
    CircularBuffer in0(CB_IN0);
    CircularBuffer w(CB_W);
    CircularBuffer partials(CB_PART);
    CircularBuffer out(CB_OUT);
    CircularBuffer bias(CB_BIAS);

    // The bias is resident: front it once.
    cb_reserve_back(CB_BIAS, N);
    cb_push_back(CB_BIAS, N);
    bias.wait_front(N);

    if constexpr (mode != 0 && (bias_mode == 1 || bias_mode == 3)) {
        // Row-broadcast bias block, built once (bf8: identical rows keep row 0's shared exponents, so exact).
        MaybeDeviceZoneScope("compute_bias_bcast");
        reconfig_data_format_srcb(CB_BIAS);
        pack_reconfig_data_format(CB_BIASF);
        unary_bcast_init<BroadcastType::ROW>(CB_BIAS);
        cb_reserve_back(CB_BIASF, N);
        for (uint32_t n = 0; n < N; ++n) {
            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(CB_BIAS, n, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_BIASF);
            tile_regs_release();
        }
        cb_push_back(CB_BIASF, N);
        unary_bcast_uninit<BroadcastType::ROW>(CB_BIAS);
        cb_wait_front(CB_BIASF, N);
    }

    for (uint32_t m = 0; m < m_iters; ++m) {
        if constexpr (mode == 0) {
            {
                MaybeDeviceZoneScope("compute_matmul_walk");
                ckl::matmul_block<
                    false,
                    true,
                    ckl::LastBlockTarget::Interm,
                    ckl::OutputCBLayout::TileRowMajor,
                    ckl::matmul_config::InitMode::ShortAfterPreKBlock,
                    ckl::InputPolicy::WaitAndPopPerKBlock,
                    ckl::InputPolicy::WaitAndPopPerKBlock,
                    ckl::NoPostCompute,
                    KBlockHook<bm, bk, N, center, tw>,
                    ckl::NoPostKBlock,
                    0,
                    ckl::NoKBlockInnerDimFn,
                    CenterIn0<center>>(
                    in0,
                    w,
                    out,
                    partials,
                    ckl::MatmulBlockShape::of(bm / sb_h, N / sb_w, sb_h, sb_w, bk, nkb),
                    {},
                    KBlockHook<bm, bk, N, center, tw>{&rm, &in0_dfb},
                    N,
                    N,
                    {},
                    {},
                    CenterIn0<center>{});
            }
            MaybeDeviceZoneScope("compute_bias_add");
            pack_reconfig_l1_acc(0);
            pack_relu_config(ReluConfig::zero());
            ckl::add_bias_bcast_rows<ckl::BiasBroadcast::RowBroadcast, ckl::OutputCBLayout::TileRowMajor>(
                partials, bias, out, ckl::BiasAddShape::of(bm / sb_h, N / sb_w, sb_h, sb_w, N));
            pack_relu_config(ReluConfig::none());
        } else {
            {
                // Prelude: every non-center K-block tilized into the resident in0 ([6][bm][bk]).
                MaybeDeviceZoneScope("compute_tilize");
                pack_reconfig_l1_acc(0);
                if constexpr (center) {
                    cb_reserve_back(CB_CTR, 3 * bm * bk);
                    cb_push_back(CB_CTR, 3 * bm * bk);
                }
                for (uint32_t t = 0; t < tilized_kb; ++t) {
                    cb_reserve_back(CB_RM, bm * bk);
                    cb_push_back(CB_RM, bm * bk);
                    conv2d_3x3::lean_tilize::tilize<tw, CB_RM, CB_IN0>(rm, in0_dfb, tblocks);
                }
            }
            MaybeDeviceZoneScope("compute_matmul_walk");
            in0.wait_front(tilized_kb * bm * bk);
            if constexpr (center) {
                cb_wait_front(CB_CTR, 3 * bm * bk);
            }
            if constexpr (bias_mode == 0) {
                partials.reserve_back(bm * N);
            } else {
                out.reserve_back(bm * N);
            }
            if constexpr (bias_mode != 2) {
                pack_relu_config(bias_mode == 1 || bias_mode == 3 ? ReluConfig::zero() : ReluConfig::none());
            }
            reconfig_data_format(CB_W, CB_IN0);
            pack_reconfig_data_format(bias_mode == 1 || bias_mode == 3 ? CB_OUT : CB_PART);
            // Subblock shapes are compile-time (runtime h / w make every matmul_block call generic: measured ~2x the
            // per-call cost on 1 x 1 / 1 x 2 subblocks).
            uint32_t cur = 0;
            auto run_sub = [&](auto H, auto Wd, uint32_t r0, uint32_t c0, uint32_t y) {
                constexpr uint32_t h = decltype(H)::value;
                constexpr uint32_t sw = decltype(Wd)::value;
                if (cur != h * 16 + sw) {
                    ckernel::matmul_block_init(CB_IN0, CB_W, false, sw, h, bk);
                    cur = h * 16 + sw;
                }
                tile_regs_acquire();
                uint32_t in1_index = c0;
                if constexpr (center) {
#pragma GCC unroll 1
                    for (uint32_t kh = 0; kh < 3; ++kh) {
                        uint32_t idx = (2 * kh) * bm * bk + r0 * bk;
                        for (uint32_t k = 0; k < bk; ++k) {
                            ckernel::matmul_block(CB_IN0, CB_W, idx + k, in1_index, 0, false, sw, h, bk);
                            in1_index += s;
                        }
                        idx = kh * bm * bk + r0 * bk;
                        for (uint32_t k = 0; k < bk; ++k) {
                            ckernel::matmul_block(CB_CTR, CB_W, idx + k, in1_index, 0, false, sw, h, bk);
                            in1_index += s;
                        }
                        idx = (2 * kh + 1) * bm * bk + r0 * bk;
                        for (uint32_t k = 0; k < bk; ++k) {
                            ckernel::matmul_block(CB_IN0, CB_W, idx + k, in1_index, 0, false, sw, h, bk);
                            in1_index += s;
                        }
                    }
                } else {
#pragma GCC unroll 1
                    for (uint32_t kb = 0; kb < nkb; ++kb) {
                        const uint32_t idx = kb * bm * bk + r0 * bk;
                        for (uint32_t k = 0; k < bk; ++k) {
                            ckernel::matmul_block(CB_IN0, CB_W, idx + k, in1_index, 0, false, sw, h, bk);
                            in1_index += s;
                        }
                    }
                }
                if constexpr (bias_mode == 1 || bias_mode == 3) {
                    if constexpr (bias_mode == 3) {
                        reconfig_data_format_srca(CB_PART);
                    }
                    // + bias in DEST: DEST -> SrcA, the row-broadcast bias block -> SrcB (bf8, as in0: no SrcB
                    // format change).
                    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_BIASF);
                    for (uint32_t r = 0; r < h; ++r) {
                        for (uint32_t c = 0; c < sw; ++c) {
                            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                                CB_BIASF, y * s + c0 + c, r * sw + c);
                        }
                    }
                    if constexpr (bias_mode == 3) {
                        reconfig_data_format_srca(CB_W);
                    }
                    cur = 0;  // the matmul state is gone
                }
                tile_regs_commit();
                tile_regs_wait();
                if constexpr (bias_mode == 0) {
                    ckl::pack_subblock_row_strided(0, CB_PART, r0 * N + y * s + c0, N, h, sw);
                } else if constexpr (bias_mode == 1 || bias_mode == 3) {
                    ckl::pack_subblock_row_strided(0, CB_OUT, r0 * N + y * s + c0, N, h, sw);
                } else {
                    ckl::pack_subblock_row_strided(0, CB_PART, r0 * s + c0, s, h, sw);
                }
                tile_regs_release();
            };
            constexpr uint32_t W1 = s < 4 ? s : 4;
            using I1 = std::integral_constant<uint32_t, 1>;
            using IW = std::integral_constant<uint32_t, W1>;
            // Tall plan: TALL-row groups of TALL x 1 subblocks per column, then the tail rows as 1 x W1 subblocks.
            constexpr uint32_t TALL = bm < 4 ? bm : 4;
            constexpr uint32_t GROUPS = bm / TALL;
            constexpr uint32_t TAIL0 = GROUPS * TALL;
            using IT = std::integral_constant<uint32_t, TALL>;
            for (uint32_t y = 0; y < N / s; ++y) {
                cb_reserve_back(CB_W, K * s);
                cb_push_back(CB_W, K * s);
                w.wait_front(K * s);
                if constexpr (bias_mode == 2) {
                    partials.reserve_back(bm * s);
                }
                if constexpr (wide) {
                    for (uint32_t r = 0; r < bm; ++r) {
                        for (uint32_t c = 0; c < s; c += W1) {
                            run_sub(I1{}, IW{}, r, c, y);
                        }
                    }
                } else {
                    const bool rev = mode == 1 && (y & 1);
                    if (rev) {
                        for (uint32_t r = TAIL0; r < bm; ++r) {
                            for (uint32_t c = 0; c < s; c += W1) {
                                run_sub(I1{}, IW{}, r, c, y);
                            }
                        }
                    }
                    for (uint32_t g = 0; g < GROUPS; ++g) {
                        for (uint32_t c = 0; c < s; ++c) {
                            run_sub(IT{}, I1{}, g * TALL, c, y);
                        }
                    }
                    if (!rev) {
                        for (uint32_t r = TAIL0; r < bm; ++r) {
                            for (uint32_t c = 0; c < s; c += W1) {
                                run_sub(I1{}, IW{}, r, c, y);
                            }
                        }
                    }
                }
                w.pop_front(K * s);
                if constexpr (bias_mode == 2) {
                    // The slice's bias pass: [bm, s] fp32 -> + bias -> ReLU -> output columns y*s.. (absolute packs).
                    partials.push_back(bm * s);
                    partials.wait_front(bm * s);
                    reconfig_data_format_srca(CB_PART);
                    reconfig_data_format_srcb(CB_BIAS);
                    pack_reconfig_data_format(CB_OUT);
                    pack_relu_config(ReluConfig::zero());
                    add_bcast_rows_init(CB_PART, CB_BIAS);
                    for (uint32_t r = 0; r < bm; ++r) {
                        tile_regs_acquire();
                        for (uint32_t c = 0; c < s; ++c) {
                            add_tiles_bcast_rows(CB_PART, CB_BIAS, r * s + c, y * s + c, c);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t c = 0; c < s; ++c) {
                            pack_tile<true>(c, CB_OUT, r * N + y * s + c);
                        }
                        tile_regs_release();
                    }
                    partials.pop_front(bm * s);
                    pack_relu_config(ReluConfig::none());
                    reconfig_data_format(CB_W, CB_IN0);
                    pack_reconfig_data_format(CB_PART);
                    cur = 0;
                }
            }
            in0.pop_front(tilized_kb * bm * bk);
            if constexpr (center) {
                cb_pop_front(CB_CTR, 3 * bm * bk);
            }
            if constexpr (bias_mode == 0) {
                partials.push_back(bm * N);
                MaybeDeviceZoneScope("compute_bias_add");
                pack_relu_config(ReluConfig::zero());
                ckl::add_bias_bcast_rows<ckl::BiasBroadcast::RowBroadcast, ckl::OutputCBLayout::TileRowMajor>(
                    partials, bias, out, ckl::BiasAddShape::of(bm / sb_h, N / sb_w, sb_h, sb_w, N));
                pack_relu_config(ReluConfig::none());
            } else {
                out.push_back(bm * N);
                pack_relu_config(ReluConfig::none());
            }
        }
    }
}
