// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert rows, compute of one core of one expert group. Per job (one expert, at most 32 routed rows):
//  P1: for each of this core's W0/W1 column groups: [g0 u0 g1 u1] = x_job . [W0 j0, W1 j0, W0 j1, W1 j1] over all
//      hidden tiles, activation on the packer's SFPU, a(j0), a(j1) -> cb_a; a half block-column (the last group of a
//      ring position with an odd column count in the compact layout) is [g0 u0] = x_job . [W0 j0, W1 j0] -> a(j0).
//  P2: a2 (every intermediate column of the job, from the exchange) . W2 for each of this core's 4-tile output groups
//      (2 tiles for the half-width last a2a iteration), the W2 rows in moe_compute's per-core rotated order -> cb_rows
//      (4 bf16 tiles; the writer sends the real rows).
// Each output tile sums the same K rows in the same order as the ring program (compute.cpp), at ct_dim 2 where it
// uses 2.
// Order P1(0), P1(1), P2(0), P1(2), P2(1), ..., P2(J - 1).
// A job of several row tiles (M > 1) streams its weights once for all its row tiles: P1 walks x in chunks of KC hidden
// tiles (chunk-major weights) and P2 walks each output group's W2 rows in chunks of KC2; between chunks each (group,
// row tile) leaves its DEST tiles in cb_part (FP32 with FP32 DEST) and the next chunk loads them back before it
// accumulates, so every output tile is summed over K in the same order as with one row tile per job. With a single
// chunk (KC covers x, KC2 the W2 rows; always with M = 1) nothing is spilled.
#include <cstdint>
#include "../moe_ring_common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/fill.h"
#include "../moe_activation.h"

namespace {

constexpr uint32_t BT = get_named_compile_time_arg_val("block_tiles");
constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");

// Weight tiles of one pass arrive in blocks of BT; a run (one group) ends with a short block.
struct WeightStream {
    uint32_t left = 0, cur = 0;
    bool held = false;
    FORCE_INLINE uint32_t take(uint32_t n) {
        if (left == 0) {
            if (held) {
                cb_pop_front(cb_w, BT);
            }
            cb_wait_front(cb_w, BT);
            held = true;
            left = BT;
            cur = 0;
        }
        const uint32_t i = cur;
        cur += n;
        left -= n;
        return i;
    }
    FORCE_INLINE void end_run() {
        if (held) {
            cb_pop_front(cb_w, BT);
        }
        held = false;
        left = 0;
    }
};

}  // namespace

void kernel_main() {
    constexpr uint32_t Ht = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t Nt = get_named_compile_time_arg_val("intermediate_tiles");
    constexpr uint32_t ring = get_named_compile_time_arg_val("ring_cores");
    constexpr bool has_bias = get_named_compile_time_arg_val("has_bias") == 1;
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
    constexpr uint32_t cb_a = get_named_compile_time_arg_val("cb_a");
    constexpr uint32_t cb_a2 = get_named_compile_time_arg_val("cb_a2");
    constexpr uint32_t cb_rows = get_named_compile_time_arg_val("cb_rows");
    constexpr uint32_t cb_ctl = get_named_compile_time_arg_val("cb_ctl");
    constexpr uint32_t cb_ones = get_named_compile_time_arg_val("cb_ones");
    constexpr uint32_t a_tiles = get_named_compile_time_arg_val("a_tiles");
    constexpr uint32_t M = get_named_compile_time_arg_val("row_tiles");
    constexpr uint32_t KC = get_named_compile_time_arg_val("chunk_tiles");
    constexpr uint32_t SB = get_named_compile_time_arg_val("chunk_blocks");
    constexpr uint32_t KC2 = get_named_compile_time_arg_val("w2_chunk_rows");
    constexpr uint32_t cb_part = get_named_compile_time_arg_val("cb_part");
    constexpr auto activation =
        ttnn::experimental::prim::detail::MoEActivationFunction(get_named_compile_time_arg_val("activation_function"));
    static_assert(BT % 4 == 0, "a block holds whole 4-tile K rows");
    constexpr auto cols_lut = moe_ring::make_shard_lut<Nt, ring>();

    const uint32_t ng = get_arg_val<uint32_t>(0);  // W0/W1 column groups of this core
    const uint32_t nq = get_arg_val<uint32_t>(1);  // W2 output groups of this core
    const uint32_t r = get_arg_val<uint32_t>(2);   // ring position of this core's bank (W2 row rotation)
    // the last W0/W1 group is a half block-column, the last W2 group the half-width last a2a iteration
    const bool half_col = get_arg_val<uint32_t>(3) == 1;
    const bool half_out = get_arg_val<uint32_t>(4) == 1;
    // tiles per K row (matmul ct_dim) of column group u and of output group q: 4, or 2 for a half group
    auto p1_width = [&](uint32_t u) -> uint32_t { return half_col && u + 1 == ng ? 2 : 4; };
    auto p2_width = [&](uint32_t q) -> uint32_t { return half_out && q + 1 == nq ? 2 : 4; };

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_a);
    if constexpr (has_bias) {
        // ones tile: matmul(ones, bias row) adds the bias to every row
        pack_reconfig_data_format(cb_ones);
        copy_init(cb_ones);
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(0, 1.f);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_ones, 1);
        pack_tile(0, cb_ones);
        tile_regs_release();
        cb_push_back(cb_ones, 1);
        cb_wait_front(cb_ones, 1);
    }
    cb_wait_front(cb_ctl, 1);
    const uint32_t nj = read_tile_value(cb_ctl, 0, 0);  // this group's jobs
    if (nj == 0) {
        return;
    }
    pack_reconfig_data_format(cb_a);
    reconfig_data_format_srcb(cb_x);
    reconfig_data_format_srca(cb_w);
    MATH((ckernel::zeroacc()));
    WeightStream ws;

    constexpr uint32_t C1 = (Ht + KC - 1) / KC;
    constexpr uint32_t C2 = (Nt + KC2 - 1) / KC2;
    constexpr uint32_t UNIT = SB * BT;  // weight pages of one chunk unit (rows_reader.cpp)
    // a one-row-tile job holds nothing for other row tiles: x chunks as large as its slot, its weights streamed in
    // whole runs, its pass padded to whole units when held units exist (M > 1, rows_reader.cpp)
    constexpr uint32_t KC1 = KC * M < Ht ? KC * M : Ht;
    constexpr uint32_t C1S = (Ht + KC1 - 1) / KC1;
    constexpr uint32_t UNIT_BLOCKS = M > 1 ? SB : 1;
    // packer target: the partials, or the bf16 outputs (cb_a / cb_rows share a format)
    bool packing_part = false;
    auto pack_target = [&](bool part) {
        if (part != packing_part) {
            if (part) {
                pack_reconfig_data_format(cb_a, cb_part);
            } else {
                pack_reconfig_data_format(cb_part, cb_a);
            }
            packing_part = part;
        }
    };
    // cb_part holds one 4-tile slot per (group, row tile) whatever the group's width (a half group fills 2 of them):
    // every reserve then starts at a multiple of 4 tiles of the CB (4 M groups tiles) and none runs past its end
    constexpr uint32_t PART_SLOT = 4;
    // DEST 0..w-1 <- this (group, row tile)'s partials of the previous chunk, in the order they were packed; then the
    // matmul of a group w tiles wide
    auto reload = [&](uint32_t cb_in0, uint32_t w) {
        reconfig_data_format_srca(cb_w, cb_part);
        copy_init(cb_part);
        cb_wait_front(cb_part, PART_SLOT);
        for (uint32_t t = 0; t < w; ++t) {
            copy_tile(cb_part, t, t);
        }
        cb_pop_front(cb_part, PART_SLOT);
        reconfig_data_format_srca(cb_part, cb_w);
        matmul_block_init(cb_in0, cb_w, /*transpose=*/false, /*ct_dim=*/w, /*rt_dim=*/1, /*kt_dim=*/1);
    };
    auto pack_part = [&](uint32_t w) {
        tile_regs_wait();
        cb_reserve_back(cb_part, PART_SLOT);
        for (uint32_t t = 0; t < w; ++t) {
            pack_tile<true>(t, cb_part, t);
        }
        cb_push_back(cb_part, PART_SLOT);
    };
    // the activation of a group w tiles wide on the packer's SFPU (DEST 0..w-1), its a tiles -> cb_a [at, at + w / 2)
    // (as compute.cpp's compute_w0_w1_block_column: wait for math plus a CFG stall before the SFPU's DEST offset)
    auto pack_activation = [&](uint32_t w, uint32_t at) {
        PACK(TTI_SEMWAIT(
            p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
        if (w == 4) {
            ::moe_activation::PackActivation<activation, /*kPairs=*/2>::compute();
        } else {
            ::moe_activation::PackActivation<activation, /*kPairs=*/1>::compute();
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        pack_tile<true>(0, cb_a, at);
        if (w == 4) {
            pack_tile<true>(2, cb_a, at + 1);
        }
    };
    auto rows_of = [&](uint32_t e) { return read_tile_value(cb_ctl, 0, 1 + e); };
    constexpr auto col_lut = [] {
        moe_ring::ShardLUT<ring> first{};
        uint32_t col = 0;
        for (uint32_t s = 0; s < ring; ++s) {
            first.data[s] = col;
            col += moe_ring::shard_tiles(Nt, s, ring);
        }
        return first;
    }();
    auto pop_padding = [&](uint32_t blocks) {
        const uint32_t pad = (UNIT_BLOCKS - blocks % UNIT_BLOCKS) % UNIT_BLOCKS;
        if (pad) {
            cb_wait_front(cb_w, pad * BT);
            cb_pop_front(cb_w, pad * BT);
        }
    };
    auto phase1_single = [&]() {
        if constexpr (activation != ttnn::experimental::prim::detail::MoEActivationFunction::GELU) {
            ::moe_activation::pack_init_activation<activation>();
        }
        matmul_block_init(cb_x, cb_w, /*transpose=*/false, /*ct_dim=*/4, /*rt_dim=*/1, /*kt_dim=*/1);
        uint32_t blocks = 0;
        for (uint32_t c = 0; c < C1S; ++c) {
            const bool last = c + 1 == C1S;
            const uint32_t kc = last ? Ht - c * KC1 : KC1;
            pack_target(!last);
            cb_wait_front(cb_x, KC * M);
            if (last) {
                cb_reserve_back(cb_a, M * a_tiles);
            }
            for (uint32_t u = 0; u < ng; ++u) {
                const uint32_t w = p1_width(u);
                tile_regs_acquire();
                if (c > 0) {
                    reload(cb_x, w);
                } else if (w != 4) {
                    matmul_block_init(cb_x, cb_w, /*transpose=*/false, /*ct_dim=*/w, /*rt_dim=*/1, /*kt_dim=*/1);
                }
                for (uint32_t kt = 0; kt < kc; ++kt) {
                    matmul_block(cb_x, cb_w, kt, ws.take(w), 0, false, w, 1, 1);
                }
                if constexpr (has_bias) {
                    if (last) {
                        matmul_block(cb_ones, cb_w, 0, ws.take(w), 0, false, w, 1, 1);
                    }
                }
                ws.end_run();
                blocks += (w * (kc + (has_bias && last ? 1 : 0)) + BT - 1) / BT;
                tile_regs_commit();
                if (last) {
                    pack_activation(w, 2 * u);
                } else {
                    pack_part(w);
                }
                tile_regs_release();
            }
            cb_pop_front(cb_x, KC * M);
        }
        cb_push_back(cb_a, M * a_tiles);
        pop_padding(blocks);
    };
    auto phase2_single = [&]() {
        pack_target(false);
        cb_wait_front(cb_a2, M * Nt);
        matmul_block_init(cb_a2, cb_w, /*transpose=*/false, /*ct_dim=*/4, /*rt_dim=*/1, /*kt_dim=*/1);
        uint32_t blocks = 0;
        for (uint32_t q = 0; q < nq; ++q) {
            const uint32_t w = p2_width(q);
            if (w != 4) {
                matmul_block_init(cb_a2, cb_w, /*transpose=*/false, /*ct_dim=*/w, /*rt_dim=*/1, /*kt_dim=*/1);
            }
            tile_regs_acquire();
            uint32_t src = r, col = col_lut[r], left = cols_lut[r];
            for (uint32_t i = 0; i < Nt; ++i) {
                while (left == 0) {
                    src = src == 0 ? ring - 1 : src - 1;
                    left = cols_lut[src];
                    col = col_lut[src];
                }
                matmul_block(cb_a2, cb_w, col, ws.take(w), 0, false, w, 1, 1);
                ++col;
                --left;
            }
            if constexpr (has_bias) {
                matmul_block(cb_ones, cb_w, 0, ws.take(w), 0, false, w, 1, 1);
            }
            ws.end_run();
            blocks += (w * (Nt + (has_bias ? 1 : 0)) + BT - 1) / BT;
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_rows, 4);
            for (uint32_t t = 0; t < 4; ++t) {
                pack_tile<true>(t, cb_rows, t);
            }
            cb_push_back(cb_rows, 4);
            tile_regs_release();
        }
        cb_pop_front(cb_a2, M * Nt);
        pop_padding(blocks);
    };
    // P1 of a job of m row tiles: x chunk slot = [hidden tile of the chunk][row tile]; a -> cb_a [row tile][a_tiles]
    auto phase1_chunked = [&](uint32_t m) {
        if constexpr (activation != ttnn::experimental::prim::detail::MoEActivationFunction::GELU) {
            ::moe_activation::pack_init_activation<activation>();
        }
        matmul_block_init(cb_x, cb_w, /*transpose=*/false, /*ct_dim=*/4, /*rt_dim=*/1, /*kt_dim=*/1);
        for (uint32_t c = 0; c < C1; ++c) {
            const bool last = c + 1 == C1;
            const uint32_t kc = last ? Ht - c * KC : KC;
            pack_target(!last);
            cb_wait_front(cb_x, KC * M);
            if (last) {
                cb_reserve_back(cb_a, M * a_tiles);
            }
            for (uint32_t u = 0; u < ng; ++u) {
                const uint32_t w = p1_width(u);
                if (c == 0 && w != 4) {
                    matmul_block_init(cb_x, cb_w, /*transpose=*/false, /*ct_dim=*/w, /*rt_dim=*/1, /*kt_dim=*/1);
                }
                cb_wait_front(cb_w, UNIT);
                for (uint32_t rt = 0; rt < m; ++rt) {
                    tile_regs_acquire();
                    if (c > 0) {
                        reload(cb_x, w);
                    }
                    for (uint32_t kt = 0; kt < kc; ++kt) {
                        matmul_block(cb_x, cb_w, kt * m + rt, w * kt, 0, false, w, 1, 1);
                    }
                    if constexpr (has_bias) {
                        if (last) {
                            matmul_block(cb_ones, cb_w, 0, w * kc, 0, false, w, 1, 1);
                        }
                    }
                    tile_regs_commit();
                    if (last) {
                        pack_activation(w, rt * a_tiles + 2 * u);
                    } else {
                        pack_part(w);
                    }
                    tile_regs_release();
                }
                cb_pop_front(cb_w, UNIT);
            }
            cb_pop_front(cb_x, KC * M);
        }
        cb_push_back(cb_a, M * a_tiles);
    };
    // P2 of a job of m row tiles: a2 slot = [row tile][intermediate column]; rows -> cb_rows per (group, row tile)
    auto phase2_chunked = [&](uint32_t m) {
        cb_wait_front(cb_a2, M * Nt);
        matmul_block_init(cb_a2, cb_w, /*transpose=*/false, /*ct_dim=*/4, /*rt_dim=*/1, /*kt_dim=*/1);
        for (uint32_t q = 0; q < nq; ++q) {
            const uint32_t w = p2_width(q);
            if (w != 4) {
                matmul_block_init(cb_a2, cb_w, /*transpose=*/false, /*ct_dim=*/w, /*rt_dim=*/1, /*kt_dim=*/1);
            }
            // W2 rows: this ring position's columns first, then the previous ring positions' (prepare_w2 order)
            uint32_t src0 = r, col0 = col_lut[r], left0 = cols_lut[r];
            for (uint32_t p = 0; p < C2; ++p) {
                const bool last = p + 1 == C2;
                const uint32_t rows = last ? Nt - p * KC2 : KC2;
                pack_target(!last);
                cb_wait_front(cb_w, UNIT);
                uint32_t src = src0, col = col0, left = left0;
                for (uint32_t rt = 0; rt < m; ++rt) {
                    tile_regs_acquire();
                    if (p > 0) {
                        reload(cb_a2, w);
                    }
                    src = src0;
                    col = col0;
                    left = left0;
                    for (uint32_t i = 0; i < rows; ++i) {
                        while (left == 0) {
                            src = src == 0 ? ring - 1 : src - 1;
                            left = cols_lut[src];
                            col = col_lut[src];
                        }
                        matmul_block(cb_a2, cb_w, rt * Nt + col, w * i, 0, false, w, 1, 1);
                        ++col;
                        --left;
                    }
                    if constexpr (has_bias) {
                        if (last) {
                            matmul_block(cb_ones, cb_w, 0, w * rows, 0, false, w, 1, 1);
                        }
                    }
                    tile_regs_commit();
                    if (last) {
                        tile_regs_wait();
                        cb_reserve_back(cb_rows, 4);
                        for (uint32_t t = 0; t < 4; ++t) {
                            pack_tile<true>(t, cb_rows, t);
                        }
                        cb_push_back(cb_rows, 4);
                    } else {
                        pack_part(w);
                    }
                    tile_regs_release();
                }
                src0 = src;
                col0 = col;
                left0 = left;
                cb_pop_front(cb_w, UNIT);
            }
        }
        cb_pop_front(cb_a2, M * Nt);
    };

    auto phase1_job = [&](uint32_t e) {
        const uint32_t m = M > 1 ? rows_of(e) : 1;
        if (m == 1) {
            phase1_single();
        } else {
            phase1_chunked(m);
        }
    };
    auto phase2_job = [&](uint32_t e) {
        const uint32_t m = M > 1 ? rows_of(e) : 1;
        if (m == 1) {
            phase2_single();
        } else {
            phase2_chunked(m);
        }
    };
    phase1_job(0);
    for (uint32_t j = 1; j < nj; ++j) {
        phase1_job(j);
        phase2_job(j - 1);
    }
    phase2_job(nj - 1);
}
