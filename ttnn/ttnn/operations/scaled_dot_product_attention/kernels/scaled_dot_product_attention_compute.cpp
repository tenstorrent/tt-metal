// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// scaled_dot_product_attention compute: NEO t of C. Each NEO sees its own slice of every STRIDED ring and
// every entry of the ALL rings (K, V), so this is the single-threaded kernel over this NEO's units: per
// round one unit (q_chunk_tiles query tile-rows x Dt), walking the round's key chunks with the online
// softmax (op_design.md, Block schedule):
//
//   score_block          S = scale * Q K^T (+ causal mask)              -> scores     (pops K)
//   row_max_block        m_new = max(m_old, rowmax S), alpha = exp(m_old - m_new) -> row_max, rescale
//   exp_block            P = exp(S - m_new)                             -> probs      (pops S)
//   row_sum_block        l = alpha * l + rowsum P                       -> row_sum
//   rescale_accum_block  O *= alpha  (chunks after the first)           -> out_accum  (pops alpha)
//   pv_block             O += P V                                       -> out_accum  (pops P, V)
//   normalize_block      out = O * (1 / l)                              -> out        (pops O, l, m)
//
// Helper substitutions (raw LLK):
//  - matmul (score_block, pv_block): written with LLK calls, not with matmul_block().
//  - reduce MAX / SUM over a held block (row_max_block, row_sum_block): the reduce helper's
//    WaitAndPopPerTile policy would consume S / P that the next block operation still needs.
//    Raw reduce_tile with every index multiplied by the ring's consumer stride.
//  - the max / alpha / l folds: the fold combines a reduce result already in DEST with a reloaded tile,
//    which the chain (ring -> DEST per element) cannot express.
//  - scale and mask in DEST after the matmul: a chain cannot continue a DEST window it did not open
//    (chain.inl opens its own tile_regs window per walk step), so it would need a pack + reload per tile.
// The kernel library's eltwise helpers realize exp_block, rescale_accum_block and normalize_block.
//
// Ring indexing: the tile indices of matmul_tiles and reduce_tile are multiplied by
// get_consumer_stride_tiles() (C on the STRIDED rings, 1 on the ALL rings); copy_tile, the eltwise helpers
// and pack_tile take local indices.
// pack_init before every pack into a ring other than the previous one (the pack target is baked at
// pack_init on this part).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"

#include "sdpa_geometry.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t FP32_NEG_INF = 0xFF800000u;

// The math thread latches the ALU source formats. This kernel mixes source formats (bfloat16 Q/K/V/constants,
// Float32 internal rings). Clearing the latch before each op init makes that init program the ALU with its
// own operands' formats (the reduce-SUM init does the same).
ALWI void invalidate_alu_formats() {
    MATH((data_format_config_set = ckernel::math::DataFormatConfigSet::UNCONFIGURED));
}
}  // namespace

void kernel_main() {
    const uint32_t first_round = get_arg(args::first_round);
    const uint32_t num_rounds = get_arg(args::num_rounds);
    const uint32_t scale_bits = get_arg(args::scale_bits);
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t Dt = get_arg(args::Dt);
    constexpr uint32_t qc = get_arg(args::q_chunk_tiles);
    constexpr uint32_t kc = get_arg(args::k_chunk_tiles);
    constexpr bool causal = get_arg(args::causal) != 0;
    constexpr uint32_t const_tiles = get_arg(args::const_tiles);
    constexpr bool tag_threads = get_arg(args::tag_threads) != 0;
    using G = SDPA_GEOMETRY_FROM_ARGS;

    constexpr uint32_t q_block_tiles = qc * Dt;
    constexpr uint32_t kv_block_tiles = kc * Dt;
    constexpr uint32_t s_block_tiles = qc * kc;
    constexpr uint32_t o_block_tiles = qc * Dt;
    // DEST sections: score sub-blocks keep one slot for the diagonal mask when causal.
    constexpr uint32_t DEST_TILES = ckl::DEST_AUTO_LIMIT;
    constexpr uint32_t SCORE_SUB = causal ? DEST_TILES - 1 : DEST_TILES;
    constexpr uint32_t MASK_SLOT = SCORE_SUB;
    constexpr uint32_t PV_SUB = DEST_TILES;
    static_assert(SCORE_SUB >= 1 && DEST_TILES >= 3, "DEST too small for the folds");

    const uint32_t t = get_my_thread_id();

    DataflowBuffer q_ring(dfb::query_block);
    DataflowBuffer k_ring(dfb::key_block);
    DataflowBuffer v_ring(dfb::value_block);
    DataflowBuffer constants(dfb::constants);
    DataflowBuffer scores(dfb::scores);
    DataflowBuffer probs(dfb::probs);
    DataflowBuffer row_max(dfb::row_max);
    DataflowBuffer rescale(dfb::rescale);
    DataflowBuffer row_sum(dfb::row_sum);
    DataflowBuffer out_accum(dfb::out_accum);
    DataflowBuffer out(dfb::out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::query_block, dfb::key_block, dfb::scores);
    constants.wait_front(const_tiles);  // held for the whole kernel

    // Index strides of the matmul and reduce unpacks.
    const uint32_t q_stride = q_ring.get_consumer_stride_tiles();
    const uint32_t k_stride = k_ring.get_consumer_stride_tiles();
    const uint32_t v_stride = v_ring.get_consumer_stride_tiles();
    const uint32_t s_stride = scores.get_consumer_stride_tiles();
    const uint32_t p_stride = probs.get_consumer_stride_tiles();

    // ------------------------------------------------------------------ score_block
    // S[i, j] = scale * sum_d Q[i, d] K[j, d]^T, then the causal mask: tiles on the diagonal get the
    // diagonal mask added, tiles past it are -inf. The scale is applied before the mask, so the mask is never
    // scaled. Only the first `keff` key tiles of the chunk are computed; the push stays the nominal block.
    auto score_block = [&](uint32_t u, uint32_t chunk, uint32_t keff) __attribute__((always_inline)) {
        pack_reconfig_data_format(dfb::scores);
        pack_init(dfb::scores);
        scores.reserve_back(s_block_tiles);
        for (uint32_t i = 0; i < qc; ++i) {
            const uint32_t qt = G::q_row(u, i);
            for (uint32_t j0 = 0; j0 < keff; j0 += SCORE_SUB) {
                const uint32_t n = keff - j0 < SCORE_SUB ? keff - j0 : SCORE_SUB;
                const uint32_t kt0 = chunk * kc + j0;
                const bool has_diag = causal && qt >= kt0 && qt < kt0 + n;
                const bool has_full = causal && kt0 + n - 1 > qt;
                tile_regs_acquire();
                if (has_diag) {
                    reconfig_data_format_srca(dfb::constants);
                    invalidate_alu_formats();
                    copy_init(dfb::constants);
                    copy_tile(dfb::constants, DIAG_MASK_TILE, MASK_SLOT);
                }
                reconfig_data_format<SrcOrder::Reverse>(dfb::query_block, dfb::key_block);
                invalidate_alu_formats();
                matmul_init(dfb::query_block, dfb::key_block, /*transpose=*/1);
                for (uint32_t jj = 0; jj < n; ++jj) {
                    if (causal && kt0 + jj > qt) {
                        continue;  // entirely masked: filled with -inf below
                    }
                    const uint32_t k_tile = j0 + jj;
                    for (uint32_t d = 0; d < Dt; ++d) {
                        matmul_tiles(
                            dfb::query_block, dfb::key_block, (i * Dt + d) * q_stride, (k_tile * Dt + d) * k_stride, jj);
                    }
                }
                binop_with_scalar_tile_init();
                for (uint32_t jj = 0; jj < n; ++jj) {
                    mul_unary_tile(jj, scale_bits);
                }
                if (has_full) {
                    fill_tile_init();
                    for (uint32_t jj = qt + 1 > kt0 ? qt + 1 - kt0 : 0; jj < n; ++jj) {
                        fill_tile_bitcast(jj, FP32_NEG_INF);
                    }
                }
                if (has_diag) {
                    add_binary_tile_init();
                    add_binary_tile(qt - kt0, MASK_SLOT, qt - kt0);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t jj = 0; jj < n; ++jj) {
                    pack_tile<true>(jj, dfb::scores, i * kc + j0 + jj);
                }
                tile_regs_release();
            }
        }
        scores.push_back(s_block_tiles);
    };

    // ------------------------------------------------------------------ row_max_block
    // Per query tile-row one DEST section: slot 0 = max over the row's keff score tiles (column 0 valid);
    // after the first chunk slot 1 = m_old, m_new = max(slot 0, slot 1) and alpha = exp(m_old - m_new).
    // The state ring is rotated in place: m_old is popped before m_new is reserved.
    auto row_max_block = [&](uint32_t chunk, uint32_t keff) __attribute__((always_inline)) {
        scores.wait_front(s_block_tiles);  // held through exp_block, which pops it
        for (uint32_t i = 0; i < qc; ++i) {
            tile_regs_acquire();
            if (chunk > 0) {
                row_max.wait_front(1);
                reconfig_data_format_srca(dfb::row_max);
                invalidate_alu_formats();
                copy_init(dfb::row_max);
                copy_tile(dfb::row_max, 0, 1);
                row_max.pop_front(1);
            }
            pack_reconfig_data_format(dfb::row_max);
            pack_init(dfb::row_max);
            reconfig_data_format(dfb::scores, dfb::constants);
            invalidate_alu_formats();
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb::scores, dfb::constants, dfb::row_max);
            for (uint32_t j = 0; j < keff; ++j) {
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                    dfb::scores, dfb::constants, (i * kc + j) * s_stride, SCALER_TILE, 0);
            }
            if (chunk > 0) {
                binary_max_tile_init();
                binary_max_tile(0, 1, 0);  // m_new
                sub_binary_tile_init();
                sub_binary_tile(1, 0, 1);  // m_old - m_new
                exp_tile_init<false>();
                exp_tile<false>(1);  // alpha
            }
            tile_regs_commit();
            row_max.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, dfb::row_max);
            if (chunk > 0) {
                pack_reconfig_data_format(dfb::rescale);
                pack_init(dfb::rescale);
                rescale.reserve_back(1);
                pack_tile(1, dfb::rescale);
            }
            tile_regs_release();
            row_max.push_back(1);
            if (chunk > 0) {
                rescale.push_back(1);
            }
            reduce_uninit(dfb::scores);
        }
    };

    // ------------------------------------------------------------------ exp_block
    // P = exp(S - m) over the [qc x keff] block, m broadcast along each row (column broadcast). S and m are
    // held (caller-managed), P is written at its nominal block index (row stride kc).
    auto exp_block = [&](uint32_t keff) __attribute__((always_inline)) {
        row_max.wait_front(qc);
        pack_reconfig_data_format(dfb::probs);
        pack_init(dfb::probs);
        probs.reserve_back(s_block_tiles);
        invalidate_alu_formats();
        ckl::eltwise_chain(
            ckl::IterationShape::grid(qc, keff),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Sub,
                ckl::input(
                    dfb::scores,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Block,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileAddressing::Strided),
                ckl::input(
                    dfb::row_max,
                    ckl::BroadcastDim::Col,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Col)>{scores, row_max, ckl::StridedTileRange{0, kc}},
            ckl::Exp<ckl::Approx::Exact>{},
            ckl::PackTile<ckl::output(
                dfb::probs,
                ckl::ReservePolicy::None,
                ckl::PushPolicy::None,
                ckl::DataFormatReconfig::Enabled,
                ckl::TileAddressing::Strided)>{probs, ckl::StridedTileRange{0, kc}});
        probs.push_back(s_block_tiles);
        scores.pop_front(s_block_tiles);
    };

    // ------------------------------------------------------------------ row_sum_block
    // Per query tile-row: slot 0 = sum over the row's keff probability tiles (column 0 valid); after the first
    // chunk l = alpha * l_old + slot 0. Rotated in place like row_max.
    auto row_sum_block = [&](uint32_t chunk, uint32_t keff) __attribute__((always_inline)) {
        probs.wait_front(s_block_tiles);
        for (uint32_t i = 0; i < qc; ++i) {
            tile_regs_acquire();
            if (chunk > 0) {
                row_sum.wait_front(1);
                reconfig_data_format_srca(dfb::row_sum);
                invalidate_alu_formats();
                copy_init(dfb::row_sum);
                copy_tile(dfb::row_sum, 0, 1);
                row_sum.pop_front(1);
                // The unpack buffer descriptor is baked at the op init on this part: re-init per ring.
                reconfig_data_format_srca(dfb::rescale);
                copy_init(dfb::rescale);
                copy_tile(dfb::rescale, i, 2);
            }
            pack_reconfig_data_format(dfb::row_sum);
            pack_init(dfb::row_sum);
            // REDUCE_ROW SUM swaps the operands: scaler -> SrcA, data -> SrcB.
            reconfig_data_format(dfb::constants, dfb::probs);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::probs, dfb::constants, dfb::row_sum);
            for (uint32_t j = 0; j < keff; ++j) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(
                    dfb::probs, dfb::constants, (i * kc + j) * p_stride, SCALER_TILE, 0);
            }
            if (chunk > 0) {
                mul_binary_tile_init();
                mul_binary_tile(1, 2, 1);  // alpha * l_old
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            tile_regs_commit();
            row_sum.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, dfb::row_sum);
            tile_regs_release();
            row_sum.push_back(1);
            reduce_uninit(dfb::probs);
        }
    };

    // ------------------------------------------------------------------ rescale_accum_block
    // O *= alpha (column broadcast), in place: each O tile is popped before its successor is reserved.
    auto rescale_accum_block = [&]() __attribute__((always_inline)) {
        pack_reconfig_data_format(dfb::out_accum);
        pack_init(dfb::out_accum);
        invalidate_alu_formats();
        ckl::mul<
            ckl::input(dfb::out_accum),
            ckl::input(
                dfb::rescale,
                ckl::BroadcastDim::Col,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::AtEnd,
                ckl::InputTileMapping::Col),
            ckl::output(dfb::out_accum)>(ckl::IterationShape::grid(qc, Dt), out_accum, rescale, out_accum);
    };

    // ------------------------------------------------------------------ pv_block
    // O[i, d] (+)= sum_j P[i, j] V[j, d]; per DEST section of up to PV_SUB output tiles: after the first chunk
    // the O tiles are reloaded from the front of out_accum (and popped), the products accumulate on top.
    auto pv_block = [&](uint32_t chunk, uint32_t keff) __attribute__((always_inline)) {
        pack_reconfig_data_format(dfb::out_accum);
        pack_init(dfb::out_accum);
        for (uint32_t i = 0; i < qc; ++i) {
            for (uint32_t d0 = 0; d0 < Dt; d0 += PV_SUB) {
                const uint32_t n = Dt - d0 < PV_SUB ? Dt - d0 : PV_SUB;
                tile_regs_acquire();
                if (chunk > 0) {
                    out_accum.wait_front(n);
                    reconfig_data_format_srca(dfb::out_accum);
                    invalidate_alu_formats();
                    copy_init(dfb::out_accum);
                    for (uint32_t dd = 0; dd < n; ++dd) {
                        copy_tile(dfb::out_accum, dd, dd);
                    }
                    out_accum.pop_front(n);
                }
                reconfig_data_format<SrcOrder::Reverse>(dfb::probs, dfb::value_block);
                invalidate_alu_formats();
                matmul_init(dfb::probs, dfb::value_block, /*transpose=*/0);
                for (uint32_t dd = 0; dd < n; ++dd) {
                    for (uint32_t j = 0; j < keff; ++j) {
                        matmul_tiles(
                            dfb::probs, dfb::value_block, (i * kc + j) * p_stride, (j * Dt + d0 + dd) * v_stride, dd);
                    }
                }
                tile_regs_commit();
                out_accum.reserve_back(n);
                tile_regs_wait();
                for (uint32_t dd = 0; dd < n; ++dd) {
                    pack_tile(dd, dfb::out_accum);
                }
                tile_regs_release();
                out_accum.push_back(n);
            }
        }
    };

    // ------------------------------------------------------------------ normalize_block
    // 1/l in place, then out = O * (1/l) (column broadcast) straight into the writer's ring.
    auto normalize_block = [&]() __attribute__((always_inline)) {
        if constexpr (tag_threads) {
            // Test probe: every output tile of this NEO is t + 1 (shows which NEO computed which unit).
            out_accum.wait_front(o_block_tiles);
            dummy_unpack(dfb::out_accum);
            out_accum.pop_front(o_block_tiles);
            row_sum.wait_front(qc);
            dummy_unpack(dfb::row_sum);
            row_sum.pop_front(qc);
            pack_reconfig_data_format(dfb::out);
            pack_init(dfb::out);
            for (uint32_t k = 0; k < o_block_tiles; ++k) {
                tile_regs_acquire();
                fill_tile_init();
                fill_tile(0, static_cast<float>(t + 1));
                tile_regs_commit();
                out.reserve_back(1);
                tile_regs_wait();
                pack_tile(0, dfb::out);
                tile_regs_release();
                out.push_back(1);
            }
        } else {
            pack_reconfig_data_format(dfb::row_sum);
            pack_init(dfb::row_sum);
            invalidate_alu_formats();
            ckl::unary<ckl::Recip<>, ckl::input(dfb::row_sum), ckl::output(dfb::row_sum)>(
                ckl::IterationShape::tiles(qc), row_sum, row_sum);
            pack_reconfig_data_format(dfb::out);
            pack_init(dfb::out);
            invalidate_alu_formats();
            ckl::mul<
                ckl::input(dfb::out_accum),
                ckl::input(
                    dfb::row_sum,
                    ckl::BroadcastDim::Col,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::AtEnd,
                    ckl::InputTileMapping::Col),
                ckl::output(dfb::out)>(ckl::IterationShape::grid(qc, Dt), out_accum, row_sum, out);
        }
        dummy_unpack(dfb::row_max);
        row_max.pop_front(qc);
    };

    // ------------------------------------------------------------------ the schedule
    uint32_t chunk_counter = 0;  // the cluster's key chunks over all its rounds (as in the reader)
    for (uint32_t r = 0; r < num_rounds; ++r) {
        const auto rc = G::round_coords(first_round + r);
        const uint32_t u = G::unit_of(rc.head_round, t);
        const uint32_t key_limit = G::unit_key_limit(u);
        const uint32_t chunks = G::round_chunks(rc.head_round);

        q_ring.wait_front(q_block_tiles);
        for (uint32_t chunk = 0; chunk < chunks; ++chunk, ++chunk_counter) {
            const uint32_t keff = G::chunk_extent(chunk, key_limit);
            k_ring.wait_front(kv_block_tiles);
            if (keff == 0) {
                // A chunk past this NEO's last query tile (causal): pop it unread.
                dummy_unpack(dfb::key_block);
                k_ring.pop_front(kv_block_tiles);
                v_ring.wait_front(kv_block_tiles);
                dummy_unpack(dfb::value_block);
                v_ring.pop_front(kv_block_tiles);
                continue;
            }
            score_block(u, chunk, keff);
            k_ring.pop_front(kv_block_tiles);
            row_max_block(chunk, keff);
            exp_block(keff);
            if (chunk > 0) {
                rescale.wait_front(qc);
            }
            row_sum_block(chunk, keff);
            if (chunk > 0) {
                rescale_accum_block();
            }
            v_ring.wait_front(kv_block_tiles);
            pv_block(chunk, keff);
            v_ring.pop_front(kv_block_tiles);
            probs.pop_front(s_block_tiles);
        }
        normalize_block();
        q_ring.pop_front(q_block_tiles);
    }
    // Credit-only chunks that round the cluster's total up to the reader harts.
    for (; chunk_counter % R != 0; ++chunk_counter) {
        k_ring.wait_front(kv_block_tiles);
        dummy_unpack(dfb::key_block);
        k_ring.pop_front(kv_block_tiles);
        v_ring.wait_front(kv_block_tiles);
        dummy_unpack(dfb::value_block);
        v_ring.pop_front(kv_block_tiles);
    }
    dummy_unpack(dfb::constants);
    constants.pop_front(const_tiles);
}
