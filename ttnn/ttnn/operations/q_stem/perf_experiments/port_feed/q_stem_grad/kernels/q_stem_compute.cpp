// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// q_stem compute (every compute-rect core). Block ops:
//   s1_matmul_block     (stage-1 cores) q block = x round(s) (m x Q_t, in s1_kb K-blocks) @ resident q_b slice
//                       (Q_t x cols) -> cb_q_slice (m x cols, row-major), never popped here (the ports and the
//                       local-delivery writer read it in place)
//   kpart_combine_block (K-split, owners) q block = own partial (cb_q_kpart) + the partner groups' partials
//                       (cb_q_recv1 [+ cb_q_recv2]) -> cb_q_slice: one fused eltwise chain (FPU add, + SFPU add for
//                       a third group). Partners' matmul packs into cb_q_kpart, which their writer ships to the owner.
//   prefix_matmul_block per landed row: nope (1 x Dn_t) @ wkv_b1[h] (Dn_t x s3n) -> cb_prefix_out, per N-block
//   rope_block (NoPE)   copy of the row's Dr_t rope tiles -> cb_rope_out
//   rope_block (RoPE)   rot = rope (Dr_t x 1 tiles) @ trans (1 tile, rotate_half_meta) -> cb_rope_tmp (matmul_block,
//                       rope retained in cb_landing), then ONE fused chain per tile: D0 = rope * cos, D1 = rot * sin,
//                       D0 += D1 (SFPU add) -> cb_rope_out. cos / sin rows (t mod qrt) indexed in the resident tables.
//   untilize_row_block  (ROW_MAJOR output only) cb_prefix_out N-block -> cb_prefix_rm, cb_rope_out -> cb_rope_rm:
//                       32 sticks per block for the writer. A separate block op after the producers, so a rotated
//                       rope (RoPE) slots in front of it unchanged. full_stick (one N-block per row): the rope
//                       tiles are packed behind the prefix into cb_prefix_out and the whole [prefix | rope] row
//                       block is untilized once -> 32 complete output sticks.
// All through kernel_lib helpers (matmul_block, eltwise copy, untilize).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/ttnn/operations/q_stem/perf_experiments/port_feed/q_stem_grad/kernels/q_stem_common.hpp"

using namespace compute_kernel_lib;

namespace {
// in1 (resident q_b slice, fronted once) offset of K-block kb: kb * s1_kb rows of `cols` tiles.
// Fixed in1 offset (a resident head's weights inside cb_wkv).
struct TileOffset {
    uint32_t tiles;
    ALWI uint32_t operator()(uint32_t) const { return tiles; }
};
// q_b slice streaming: the reader pushes the resident slice one K-block (s1_kb x cols tiles) at a time, so the
// first block's matmul starts on K-block 0 while later K-blocks are still in flight. Cumulative wait (the slice is
// never popped until stage 1 ends); after the first block every wait is already satisfied.
struct QbKBlockWait {
    uint32_t cb, tiles_per_kblock;
    ALWI void operator()(uint32_t kb, uint32_t, bool) const { cb_wait_front(cb, (kb + 1) * tiles_per_kblock); }
};
struct QbKBlockOffset {
    uint32_t tiles_per_kblock;
    ALWI uint32_t operator()(uint32_t block) const { return block * tiles_per_kblock; }
};
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x_round = get_compile_time_arg_val(0);
    constexpr uint32_t cb_qb_slice = get_compile_time_arg_val(1);
    constexpr uint32_t cb_q_slice = get_compile_time_arg_val(2);
    constexpr uint32_t cb_q_partial = get_compile_time_arg_val(3);
    constexpr uint32_t cb_landing = get_compile_time_arg_val(4);
    constexpr uint32_t cb_wkv = get_compile_time_arg_val(5);
    constexpr uint32_t cb_prefix_out = get_compile_time_arg_val(6);
    constexpr uint32_t cb_rope_out = get_compile_time_arg_val(7);
    constexpr uint32_t Q_t = get_compile_time_arg_val(8);
    constexpr uint32_t s1_kb = get_compile_time_arg_val(9);
    constexpr uint32_t nkb = get_compile_time_arg_val(10);
    constexpr uint32_t m = get_compile_time_arg_val(11);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(12);
    constexpr uint32_t rows_per_head = get_compile_time_arg_val(13);
    constexpr uint32_t Dn_t = get_compile_time_arg_val(14);
    constexpr uint32_t Dr_t = get_compile_time_arg_val(15);
    constexpr uint32_t N_t = get_compile_time_arg_val(16);
    constexpr uint32_t s3n = get_compile_time_arg_val(17);
    constexpr uint32_t n_nb = get_compile_time_arg_val(18);
    constexpr bool wkv_resident = get_compile_time_arg_val(19) != 0;
    constexpr uint32_t sbw3 = get_compile_time_arg_val(20);
    constexpr bool out_rm = get_compile_time_arg_val(21) != 0;
    constexpr uint32_t cb_prefix_rm = get_compile_time_arg_val(22);
    constexpr uint32_t cb_rope_rm = get_compile_time_arg_val(23);
    // full stick (ROW_MAJOR, n_nb == 1): the rope tiles are packed behind the prefix in cb_prefix_out and the row
    // block [prefix | rope] is untilized once into cb_prefix_rm (one write per output stick in the writer)
    constexpr bool full_stick = get_compile_time_arg_val(24) != 0;
    constexpr uint32_t cb_rope_dst = full_stick ? cb_prefix_out : cb_rope_out;
    constexpr uint32_t ts = get_compile_time_arg_val(25);  // stage-3 consumption order (q_stem::RowOrder)
    constexpr bool k_blocked = nkb > 1;
    // RoPE (rope_mode == rope): resident trans tile + cos / sin rows of the quarter (qrt x Dr_t each)
    constexpr bool rope = get_named_compile_time_arg_val("rope") != 0;
    constexpr uint32_t cb_cos = get_named_compile_time_arg_val("cb_cos");
    constexpr uint32_t cb_sin = get_named_compile_time_arg_val("cb_sin");
    constexpr uint32_t cb_trans = get_named_compile_time_arg_val("cb_trans");
    constexpr uint32_t cb_rope_tmp = get_named_compile_time_arg_val("cb_rope_tmp");
    constexpr uint32_t qrt = get_named_compile_time_arg_val("qrt");
    constexpr uint32_t sbh_rot = get_named_compile_time_arg_val("sbh_rot");  // rotation matmul subblock height
    // K-split stage 1 (regime R6): ksplit K groups; Q_t above is this group's K slice
    constexpr uint32_t ksplit = get_named_compile_time_arg_val("ksplit");
    constexpr uint32_t cb_q_kpart = get_named_compile_time_arg_val("cb_q_kpart");
    constexpr uint32_t cb_q_recv1 = get_named_compile_time_arg_val("cb_q_recv1");
    constexpr uint32_t cb_q_recv2 = get_named_compile_time_arg_val("cb_q_recv2");
    constexpr uint32_t cb_s1_out = ksplit > 1 ? cb_q_kpart : cb_q_slice;  // the matmul's output

    const uint32_t cols = get_arg_val<uint32_t>(0);
    const uint32_t sbw1 = get_arg_val<uint32_t>(1);
    const uint32_t hpc = get_arg_val<uint32_t>(2);
    const uint32_t n_r = get_arg_val<uint32_t>(3);
    [[maybe_unused]] const uint32_t kgroup = get_arg_val<uint32_t>(4);  // K group (0: owner of q)

    CircularBuffer x_buf(cb_x_round), qb_buf(cb_qb_slice), q_buf(cb_s1_out), q_partial_buf(cb_q_partial);
    CircularBuffer landing_buf(cb_landing), wkv_buf(cb_wkv), prefix_buf(cb_prefix_out);

    if (cols > 0) {
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_round, cb_qb_slice, cb_s1_out);
    } else {
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_landing, cb_wkv, cb_prefix_out);
    }

    // ---- stage 1 ----------------------------------------------------------------------------------------------------
    if (cols > 0) {
        MaybeDeviceZoneScope("S1");  // stage 1: x rounds (waited inside matmul_block) + matmul
        // resident in1 (NoWaitNoPop below: the lifecycle is mine), waited per K-block (QbKBlockWait)
        const auto shape = MatmulBlockShape::of(m, cols / sbw1, 1, sbw1, s1_kb, nkb);
        // kpart_combine_block: q block (m x cols) = sum of the K groups' partials, into cb_q_slice
        auto kpart_combine_block = [&]() {
            if constexpr (ksplit == 2) {
                add<input(cb_q_kpart), input(cb_q_recv1), output(cb_q_slice)>(IterationShape::tiles(m * cols));
            } else if constexpr (ksplit == 3) {
                eltwise_chain(
                    IterationShape::tiles(m * cols),
                    BinaryFpu<BinaryFpuOp::Add, input(cb_q_kpart), input(cb_q_recv1), Dst::D0>{},
                    CopyTile<input(cb_q_recv2), Dst::D1>{},
                    AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
                    PackTile<output(cb_q_slice)>{});
            }
        };
        for (uint32_t j = 0; j < num_blocks; ++j) {
#ifdef ABLATE_S1_COMPUTE
            for (uint32_t kb = 0; kb < nkb; ++kb) {
                cb_wait_front(cb_x_round, m * s1_kb);
                cb_pop_front(cb_x_round, m * s1_kb);
            }
            cb_reserve_back(cb_s1_out, m * cols);
            cb_push_back(cb_s1_out, m * cols);
            if constexpr (ksplit > 1) {
                if (kgroup == 0) {
                    kpart_combine_block();
                }
            }
            continue;
#endif
            matmul_block<
                false,
                false,
                LastBlockTarget::Out,
                OutputCBLayout::SubblockMajor,
                matmul_config::InitMode::Short,
                InputPolicy::WaitAndPopPerKBlock,
                InputPolicy::NoWaitNoPop,
                NoPostCompute,
                QbKBlockWait,
                NoPostKBlock,
                0,
                NoKBlockInnerDimFn,
                NoIn0Source,
                QbKBlockOffset>(
                x_buf,
                qb_buf,
                q_buf,
                k_blocked ? q_partial_buf : q_buf,
                shape,
                {},
                QbKBlockWait{cb_qb_slice, s1_kb * cols},
                cols,
                0,
                {},
                {},
                {},
                QbKBlockOffset{s1_kb * cols});
            if constexpr (ksplit > 1) {
                if (kgroup == 0) {
                    kpart_combine_block();
                }
            }
        }
        cb_pop_front(cb_qb_slice, Q_t * cols);
    }

    // ---- stage 3 ----------------------------------------------------------------------------------------------------
    // Consumption order = landing order (q_stem::RowOrder): remote rows t < ts row-major across the n_r remote heads,
    // the local heads head-major, then the late remote rows. R1: every remote head's weights resident (head u at in1
    // offset u * Dn_t * N_t); with late rows (a core holding both kinds) the local heads stack behind them (local k at
    // (n_r + k) * head tiles) and everything is popped at the end, else each local head is popped after its rows.
    const auto shape3 = MatmulBlockShape::of(1, s3n / sbw3, 1, sbw3, Dn_t, 1);
    constexpr uint32_t head_tiles = Dn_t * N_t;
    MaybeDeviceZoneScope("S3");  // stage 3: landed rows (waited inside the helpers) + prefix matmul + rope + untilize
    CircularBuffer trans_buf(cb_trans), rope_tmp_buf(cb_rope_tmp);
    const auto shape_rot = MatmulBlockShape::of(Dr_t / sbh_rot, 1, sbh_rot, 1, 1, 1);
    constexpr uint32_t cs_tiles = qrt * Dr_t;
    if constexpr (rope) {
        if (hpc > 0) {  // resident for every rope_block (NoWaitNoPop / caller-managed below)
            cb_wait_front(cb_trans, 1);
            cb_wait_front(cb_cos, cs_tiles);
            cb_wait_front(cb_sin, cs_tiles);
        }
    }
    // rope_block: the row's Dr_t rope tiles at the front of cb_landing -> cb_rope_dst
    auto rope_block = [&](uint32_t t) {
        if constexpr (rope) {
            matmul_block<
                false,
                false,
                LastBlockTarget::Out,
                OutputCBLayout::SubblockMajor,
                matmul_config::InitMode::Short,
                InputPolicy::WaitAndRetainOnLastBlock,
                InputPolicy::NoWaitNoPop>(landing_buf, trans_buf, rope_tmp_buf, rope_tmp_buf, shape_rot);
            const uint32_t cs = (t % qrt) * Dr_t;  // the row's cos / sin tiles (cos rows follow the output rows)
            eltwise_chain(
                IterationShape::tiles(Dr_t),
                BinaryFpu<
                    BinaryFpuOp::Mul,
                    input(cb_landing),
                    input(
                        cb_cos,
                        WaitPolicy::None,
                        PopPolicy::None,
                        InputTileMapping::Block,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Offset),
                    Dst::D0>{0, cs},
                BinaryFpu<
                    BinaryFpuOp::Mul,
                    input(cb_rope_tmp),
                    input(
                        cb_sin,
                        WaitPolicy::None,
                        PopPolicy::None,
                        InputTileMapping::Block,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Offset),
                    Dst::D1>{0, cs},
                AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
                PackTile<output(cb_rope_dst)>{});
        } else {
            copy<input(cb_landing), output(cb_rope_dst)>(IterationShape::tiles(Dr_t));
        }
    };
    auto row_block = [&](uint32_t wkv_offset, uint32_t t) {
        for (uint32_t nb = 0; nb < n_nb; ++nb) {
#ifdef ABLATE_S3
            if constexpr (!wkv_resident) {
                cb_wait_front(cb_wkv, Dn_t * s3n);
                cb_pop_front(cb_wkv, Dn_t * s3n);
            }
            cb_wait_front(cb_landing, Dn_t);
            cb_reserve_back(cb_prefix_out, s3n);
            cb_push_back(cb_prefix_out, s3n);
            if constexpr (out_rm && !full_stick) {
                untilize<s3n, cb_prefix_out, cb_prefix_rm>(1);
            }
            continue;
#endif
            if constexpr (wkv_resident) {
                matmul_block<
                    false,
                    false,
                    LastBlockTarget::Out,
                    OutputCBLayout::SubblockMajor,
                    matmul_config::InitMode::Short,
                    InputPolicy::WaitAndRetainOnLastBlock,
                    InputPolicy::NoWaitNoPop,
                    NoPostCompute,
                    NoPreKBlock,
                    NoPostKBlock,
                    0,
                    NoKBlockInnerDimFn,
                    NoIn0Source,
                    TileOffset>(
                    landing_buf,
                    wkv_buf,
                    prefix_buf,
                    prefix_buf,
                    shape3,
                    {},
                    {},
                    0,
                    0,
                    {},
                    {},
                    {},
                    TileOffset{wkv_offset});
            } else {
                matmul_block<
                    false,
                    false,
                    LastBlockTarget::Out,
                    OutputCBLayout::SubblockMajor,
                    matmul_config::InitMode::Short,
                    InputPolicy::WaitAndRetainOnLastBlock,
                    InputPolicy::WaitAndPopPerKBlock>(landing_buf, wkv_buf, prefix_buf, prefix_buf, shape3);
            }
            if constexpr (out_rm && !full_stick) {
                untilize<s3n, cb_prefix_out, cb_prefix_rm>(1);  // untilize_row_block (prefix N-block)
            }
        }
        cb_pop_front(cb_landing, Dn_t);  // the row's nope, retained across the N-blocks
#ifdef ABLATE_S3
        cb_wait_front(cb_landing, Dr_t);
        cb_pop_front(cb_landing, Dr_t);
        cb_reserve_back(cb_rope_dst, Dr_t);
        cb_push_back(cb_rope_dst, Dr_t);
#else
        rope_block(t);
#endif
        if constexpr (full_stick) {
            untilize<s3n + Dr_t, cb_prefix_out, cb_prefix_rm>(1);  // untilize_row_block ([prefix | rope])
        } else if constexpr (out_rm) {
            untilize<Dr_t, cb_rope_out, cb_rope_rm>(1);  // untilize_row_block (rope)
        }
    };
    const q_stem::RowOrder order(n_r, hpc, rows_per_head, ts);
    // late remote rows after the local heads (a core holding both): no early pops (host sizes cb_wkv for every head)
    const bool stack_local = n_r > 0 && order.ts < rows_per_head;
    for (uint32_t c = 0; c < order.total(); ++c) {
        uint32_t u, t;
        order.at(c, u, t);
        if (u < n_r) {
            if constexpr (wkv_resident) {
                if (c == 0) {
                    cb_wait_front(cb_wkv, n_r * head_tiles);
                }
            }
            row_block(u * head_tiles, t);
            if constexpr (wkv_resident) {
                if (!stack_local && c + 1 == order.first_local()) {
                    cb_pop_front(cb_wkv, n_r * head_tiles);  // last remote row
                }
            }
        } else {
            const uint32_t below = stack_local ? u : 0;  // heads resident in front of this one
            if constexpr (wkv_resident) {
                if (t == 0) {
                    cb_wait_front(cb_wkv, (below + 1) * head_tiles);
                }
            }
            row_block(below * head_tiles, t);
            if constexpr (wkv_resident) {
                if (!stack_local && t + 1 == rows_per_head) {
                    cb_pop_front(cb_wkv, head_tiles);
                }
            }
        }
    }
    if constexpr (wkv_resident) {
        if (stack_local) {
            cb_pop_front(cb_wkv, hpc * head_tiles);
        }
    }
    if constexpr (rope) {
        if (hpc > 0) {
            cb_pop_front(cb_trans, 1);
            cb_pop_front(cb_cos, cs_tiles);
            cb_pop_front(cb_sin, cs_tiles);
        }
    }
}
