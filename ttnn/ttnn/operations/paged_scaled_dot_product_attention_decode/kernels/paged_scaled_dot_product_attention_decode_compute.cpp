// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute of paged_scaled_dot_product_attention_decode: one NEO = one worker of the virtual key sequence
// (op_design.md, Blocking Model). The kernel is the single-threaded one over this NEO's share: every ring is
// STRIDED, so each NEO sees its own slice (capacity counted per NEO).
//
// Per user of this cluster:
//   load_position_block   the raw position from the reader (cb_cur_pos) -> the plan (decode_plan.hpp)
//   init_state            m = NEG_EMPTY, l = 0, O = 0 (the neutral partial: a worker with no unit emits it as is)
//   scale_q_block         q' = q * scale, tensor_d_tiles tiles, resident for the user
//   per chunk (n <= k_chunk_tiles position tiles of one head segment):
//     score_block         S = q' K^T (+ row-group mask, + position mask on the head's last tile), DEST passes
//     rowmax_block        m_chunk = rowmax(S)
//     stats_block         m_new = max(m, m_chunk), alpha = exp(m - m_new)
//     exp_block           P = exp(S - m_new)
//     pv_block            O_chunk = P V (DEST sub-blocks along D), l_chunk = rowsum(P)
//     merge_block         l = alpha l + l_chunk, O = alpha O + O_chunk
//   emit_partial_block    [O, m, l] to the writer
//   combiner (rank 0): init_state; per partial k: stats (m_new, alpha, beta), beta l_k, beta O_k[share], merge;
//                      normalize_block O / l -> output.
//
// Raw LLK, with the reason for each (the design's "Helpers considered and rejected" table):
//   score_block, pv_block: matmul_tiles, not the matmul_block helper: tiles are issued one at a time with the index
//     multiplied by the ring's consumer stride.
//   score_block mask: add_reuse_dest_tiles, in the same DEST section as the matmul accumulation (a chain owns its
//     own acquire, so a helper would need a second pass over S through another ring).
//   rowmax_block: reduce_tile<MAX, ROW> by index * stride. WaitAndPopPerTile would pop S, which exp needs.
//   stats_block: copy_tile / binary_max_tile / sub_binary_tile / exp_tile in one DEST section.
//   load_position_block: read_tile_value + dummy_unpack (no helper reads a scalar from a ring; TEN-4746).
// Every other step is a kernel-library helper. pack_init precedes each helper whose output ring differs from the
// previous pack's.

#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

#include "decode_plan.hpp"

namespace ckl = compute_kernel_lib;

namespace {

// The chain element of kernel_lib's MulUnary (eltwise/unary/scalar.inl), declared here: scalar.hpp is not used.
// mul_unary_tile itself exists on Quasar (binop_with_scalar.h:61-73).
template <ckl::Dst Slot>
struct ScaleByScalar : ckl::UnaryOp<ScaleByScalar<Slot>, Slot> {
    uint32_t scale_bits;
    constexpr explicit ScaleByScalar(uint32_t bits) noexcept : scale_bits(bits) {}
    static ALWI void init() { binop_with_scalar_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        mul_unary_tile(ckl::to_u32(Slot) + slot_offset, scale_bits);
    }
};

// Drops the ALU-format latch: called before the Float32 eltwise multiplies (merge, beta products, normalize), so
// their inits program their own operands' formats.
ALWI void relatch_alu_formats() { MATH((data_format_config_set = DataFormatConfigSet::UNCONFIGURED)); }

}  // namespace

void kernel_main() {
    const uint32_t user_start = get_arg(args::user_start);
    const uint32_t user_count = get_arg(args::user_count);
    const uint32_t cluster_rank = get_arg(args::cluster_rank);
    const uint32_t scale_bits = get_arg(args::scale_bits);

    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t kv_heads = get_arg(args::kv_heads);
    constexpr uint32_t Dt = get_arg(args::tensor_d_tiles);
    constexpr uint32_t d_share_tiles = get_arg(args::d_share_tiles);
    constexpr uint32_t tensor_pos_tiles = get_arg(args::tensor_pos_tiles);
    constexpr uint32_t clusters_per_user = get_arg(args::clusters_per_user);
    constexpr uint32_t k_chunk_tiles = get_arg(args::k_chunk_tiles);
    constexpr uint32_t merge_cost = get_arg(args::merge_cost_units);
    constexpr bool has_row_groups = get_arg(args::has_row_groups) != 0;
    constexpr uint32_t pv_subblock_tiles = get_arg(args::pv_subblock_tiles);
    constexpr ckl::Approx exp_mode = get_arg(args::exp_approx) != 0 ? ckl::Approx::Fast : ckl::Approx::Exact;
    constexpr bool exp_fast = exp_mode == ckl::Approx::Fast;
    constexpr uint32_t score_pass_tiles =
        k_chunk_tiles < ckl::DEST_AUTO_LIMIT ? k_chunk_tiles : ckl::DEST_AUTO_LIMIT;
    static_assert(pv_subblock_tiles >= 1 && pv_subblock_tiles <= ckl::DEST_AUTO_LIMIT, "PV sub-block exceeds DEST");
    static_assert(ckl::DEST_AUTO_LIMIT >= 3, "stats_block holds three DEST tiles");

    const uint32_t neo = get_my_thread_id();

    DataflowBuffer cur_pos(dfb::cur_pos);
    DataflowBuffer scaler(dfb::reduce_scaler);
    DataflowBuffer q_tiles(dfb::q_tiles);
    DataflowBuffer k_tiles(dfb::k_tiles);
    DataflowBuffer v_tiles(dfb::v_tiles);
    DataflowBuffer masks(dfb::score_masks);
    DataflowBuffer partials_in(dfb::partials_in);
    DataflowBuffer q_scaled(dfb::q_scaled);
    DataflowBuffer scores(dfb::scores);
    DataflowBuffer probs(dfb::probs);
    DataflowBuffer max_run(dfb::max_run);
    DataflowBuffer col_step(dfb::col_step);
    DataflowBuffer sum_run(dfb::sum_run);
    DataflowBuffer out_acc(dfb::out_acc);
    DataflowBuffer step(dfb::step);
    DataflowBuffer partial_out(dfb::partial_out);
    DataflowBuffer output_tiles(dfb::output_tiles);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::q_scaled, dfb::k_tiles, dfb::scores);

    // Indexed unpacks (matmul, reduce) multiply by the ring's consumer stride.
    const uint32_t q_stride = q_scaled.get_consumer_stride_tiles();
    const uint32_t score_stride = scores.get_consumer_stride_tiles();
    const uint32_t prob_stride = probs.get_consumer_stride_tiles();

    scaler.wait_front(1);  // pushed once by the reader; popped at the end

    // A held operand broadcast along columns (a column-vector tile: row max, alpha, beta, 1/l).
    constexpr auto held_col = [](auto token) {
        return ckl::input(token, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None);
    };

    // ---- init_state: the neutral partial m = NEG_EMPTY, l = 0, O = 0 (o_tiles tiles).
    // On Quasar the SFPU takes DEST's format from the last FPU write to that DEST bank, which a fill alone never
    // makes. So both DEST banks (half sync) are primed once, at kernel start, with a datacopy of the resident
    // scaler tile.
    for (uint32_t bank = 0; bank < 2; ++bank) {
        tile_regs_acquire();
        reconfig_data_format_srca(dfb::reduce_scaler);
        copy_init(dfb::reduce_scaler);
        copy_tile(dfb::reduce_scaler, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        tile_regs_release();
    }
    auto init_state = [&](uint32_t o_tiles) {
        pack_init(dfb::max_run);
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::FillScalar<ckl::Dst::D0>{__builtin_bit_cast(float, decode_plan::NEG_EMPTY_FP32_BITS)},
            ckl::PackTile<ckl::output(dfb::max_run)>{max_run});
        pack_init(dfb::sum_run);
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::FillScalar<ckl::Dst::D0>{0.0f},
            ckl::PackTile<ckl::output(dfb::sum_run)>{sum_run});
        pack_init(dfb::out_acc);
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(o_tiles),
            ckl::FillScalar<ckl::Dst::D0>{0.0f},
            ckl::PackTile<ckl::output(dfb::out_acc)>{out_acc});
    };

    // ---- stats_block: m_new = max(m, m_step); alpha = exp(m - m_new); with_beta: beta = exp(m_step - m_new).
    // m_step is the front tile of `step_dfb` (popped). Writes max_run (m_new) and col_step ([beta,] alpha).
    auto stats_block = [&](uint32_t step_dfb, auto& step_ring, bool with_beta) {
        tile_regs_acquire();
        max_run.wait_front(1);
        step_ring.wait_front(1);
        reconfig_data_format(dfb::max_run, dfb::max_run);
        copy_init(dfb::max_run);
        copy_tile(dfb::max_run, 0, 0);
        reconfig_data_format(step_dfb, step_dfb);
        copy_init(step_dfb);
        copy_tile(step_dfb, 0, 1);
        binary_max_tile_init();
        binary_max_tile(0, 1, 2);
        sub_binary_tile_init();
        sub_binary_tile(0, 2, 0);
        if (with_beta) {
            sub_binary_tile(1, 2, 1);
        }
        exp_tile_init<exp_fast>();
        exp_tile<exp_fast>(0);
        if (with_beta) {
            exp_tile<exp_fast>(1);
        }
        tile_regs_commit();
        max_run.pop_front(1);
        step_ring.pop_front(1);
        const uint16_t col_tiles = with_beta ? 2 : 1;
        max_run.reserve_back(1);
        col_step.reserve_back(col_tiles);
        tile_regs_wait();
        pack_reconfig_data_format(dfb::max_run);
        pack_init(dfb::max_run);
        pack_tile(2, dfb::max_run);
        pack_init(dfb::col_step);
        if (with_beta) {
            pack_tile(1, dfb::col_step);  // beta first: the beta products read it at the front, then pop it
        }
        pack_tile(0, dfb::col_step);
        tile_regs_release();
        max_run.push_back(1);
        col_step.push_back(col_tiles);
    };

    // ---- merge_block: O = alpha O + O_step (o_tiles tiles); l = alpha l + l_step. `step` holds [O_step, l_step];
    // alpha is the front of col_step (popped).
    auto merge_block = [&](uint32_t o_tiles) {
        relatch_alu_formats();
        pack_init(dfb::out_acc);
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(o_tiles),
            ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, ckl::input(dfb::out_acc), held_col(dfb::col_step)>{out_acc, col_step},
            ckl::DestReuseBinary<ckl::BinaryFpuOp::Add, ckl::input(dfb::step), ckl::DestReuseType::DEST_TO_SRCA>{step},
            ckl::PackTile<ckl::output(dfb::out_acc)>{out_acc});
        pack_init(dfb::sum_run);
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, ckl::input(dfb::sum_run), held_col(dfb::col_step)>{sum_run, col_step},
            ckl::DestReuseBinary<ckl::BinaryFpuOp::Add, ckl::input(dfb::step), ckl::DestReuseType::DEST_TO_SRCA>{step},
            ckl::PackTile<ckl::output(dfb::sum_run)>{sum_run});
        col_step.pop_front(1);
    };

    for (uint32_t u = user_start; u < user_start + user_count; ++u) {
        // ---- load_position_block
        cur_pos.wait_front(1);
        const uint32_t raw_pos = cur_pos.read_tile_value<uint32_t>(0, 0);
        dummy_unpack(dfb::cur_pos);
        cur_pos.pop_front(1);
        const auto plan = decode_plan::make_plan<tensor_pos_tiles, kv_heads, C, clusters_per_user, merge_cost>(
            raw_pos, cluster_rank, neo);
        if (!plan.cluster_active) {
            continue;
        }

        init_state(Dt);
        if (plan.unit_count > 0) {
            // ---- scale_q_block: q' = q * scale (any sign: applied before the mask, rule R4).
            pack_init(dfb::q_scaled);
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(Dt),
                ckl::CopyTile<ckl::input(dfb::q_tiles)>{q_tiles},
                ScaleByScalar<ckl::Dst::D0>{scale_bits},
                ckl::PackTile<ckl::output(dfb::q_scaled)>{q_scaled});
            q_scaled.wait_front(Dt);

            decode_plan::for_each_chunk<k_chunk_tiles>(
                plan,
                [&](uint32_t, uint32_t, uint32_t) { masks.wait_front(2); },
                [&](uint32_t, uint32_t, uint32_t n, bool ends_head) {
                    // ---- score_block
                    scores.reserve_back(k_chunk_tiles);
                    for (uint32_t p0 = 0; p0 < n; p0 += score_pass_tiles) {
                        const uint32_t np = (n - p0) < score_pass_tiles ? (n - p0) : score_pass_tiles;
                        const bool pass_has_last = ends_head && (p0 + np == n);
                        tile_regs_acquire();
                        reconfig_data_format(dfb::k_tiles, dfb::q_scaled);
                        matmul_init(dfb::q_scaled, dfb::k_tiles, /*transpose=*/1);
                        for (uint32_t i = 0; i < np; ++i) {
                            for (uint32_t j = 0; j < Dt; ++j) {
                                k_tiles.wait_front(1);
                                matmul_tiles(dfb::q_scaled, dfb::k_tiles, j * q_stride, 0, i);
                                k_tiles.pop_front(1);
                            }
                        }
                        if (has_row_groups || pass_has_last) {
                            reconfig_data_format(dfb::score_masks, dfb::score_masks);
                            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::score_masks);
                            for (uint32_t i = 0; i < np; ++i) {
                                const bool is_last = pass_has_last && (i + 1 == np);
                                if (has_row_groups || is_last) {
                                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                                        dfb::score_masks, is_last ? 1 : 0, i);
                                }
                            }
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_reconfig_data_format(dfb::scores);
                        pack_init(dfb::scores);
                        for (uint32_t i = 0; i < np; ++i) {
                            pack_tile<true>(i, dfb::scores, p0 + i);
                        }
                        tile_regs_release();
                    }
                    scores.push_back(k_chunk_tiles);
                    scores.wait_front(k_chunk_tiles);

                    // ---- rowmax_block
                    tile_regs_acquire();
                    col_step.reserve_back(1);
                    reconfig_data_format(dfb::scores, dfb::reduce_scaler);
                    pack_reconfig_data_format(dfb::col_step);
                    pack_init(dfb::col_step);
                    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb::scores, dfb::reduce_scaler, dfb::col_step);
                    for (uint32_t t = 0; t < n; ++t) {
                        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                            dfb::scores, dfb::reduce_scaler, t * score_stride, 0, 0);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, dfb::col_step);
                    reduce_uninit(dfb::scores);
                    tile_regs_release();
                    col_step.push_back(1);

                    // ---- stats_block (m_chunk is the front of col_step; alpha replaces it)
                    stats_block(dfb::col_step, col_step, false);

                    // ---- exp_block: P = exp(S - m_new), the nominal block reserved and pushed whole.
                    probs.reserve_back(k_chunk_tiles);
                    pack_init(dfb::probs);
                    ckl::eltwise_chain(
                        ckl::IterationShape::tiles(n),
                        ckl::BinaryFpu<
                            ckl::BinaryFpuOp::Sub,
                            ckl::input(
                                dfb::scores, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
                            held_col(dfb::max_run)>{scores, max_run},
                        ckl::Exp<exp_mode>{},
                        ckl::PackTile<ckl::output(dfb::probs, ckl::ReservePolicy::None, ckl::PushPolicy::None)>{probs});
                    scores.pop_front(k_chunk_tiles);
                    probs.push_back(k_chunk_tiles);
                    probs.wait_front(k_chunk_tiles);

                    // ---- pv_block: O_chunk = P V, DEST sub-blocks of pv_subblock_tiles along D.
                    step.reserve_back(Dt);
                    for (uint32_t b0 = 0; b0 < Dt; b0 += pv_subblock_tiles) {
                        const uint32_t nb = (Dt - b0) < pv_subblock_tiles ? (Dt - b0) : pv_subblock_tiles;
                        tile_regs_acquire();
                        reconfig_data_format(dfb::v_tiles, dfb::probs);
                        matmul_init(dfb::probs, dfb::v_tiles, /*transpose=*/0);
                        for (uint32_t t = 0; t < n; ++t) {
                            for (uint32_t jj = 0; jj < nb; ++jj) {
                                v_tiles.wait_front(1);
                                matmul_tiles(dfb::probs, dfb::v_tiles, t * prob_stride, 0, jj);
                                v_tiles.pop_front(1);
                            }
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_reconfig_data_format(dfb::step);
                        pack_init(dfb::step);
                        for (uint32_t jj = 0; jj < nb; ++jj) {
                            pack_tile<true>(jj, dfb::step, b0 + jj);
                        }
                        tile_regs_release();
                    }
                    step.push_back(Dt);

                    // ---- rowsum: l_chunk = rowsum(P) over the n valid tiles, then the nominal pad is released.
                    ckl::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, dfb::probs, dfb::reduce_scaler, dfb::step>(
                        probs, scaler, step, ckl::ReduceInputBlockShape::row(n));
                    if (n < k_chunk_tiles) {
                        dummy_unpack(dfb::probs);
                        probs.pop_front(k_chunk_tiles - n);
                    }

                    // ---- merge_block
                    merge_block(Dt);
                },
                [&]() {
                    dummy_unpack(dfb::score_masks);
                    masks.pop_front(2);
                });
            q_scaled.pop_front(Dt);
        }

        // ---- emit_partial_block: [O tiles, m, l] in the order the writer stores them.
        pack_init(dfb::partial_out);
        ckl::copy<ckl::input(dfb::out_acc), ckl::output(dfb::partial_out)>(
            ckl::IterationShape::tiles(Dt), out_acc, partial_out);
        ckl::copy<ckl::input(dfb::max_run), ckl::output(dfb::partial_out)>(
            ckl::IterationShape::one_tile(), max_run, partial_out);
        ckl::copy<ckl::input(dfb::sum_run), ckl::output(dfb::partial_out)>(
            ckl::IterationShape::one_tile(), sum_run, partial_out);

        if (cluster_rank == 0) {
            // ---- combine: merge every partial of the user into this NEO's share of D.
            init_state(d_share_tiles);
            for (uint32_t k = 0; k < plan.workers_active; ++k) {
                // partials_in streams [m_k, O_k share, l_k]; stats pops m_k and leaves [beta, alpha] in col_step.
                stats_block(dfb::partials_in, partials_in, true);
                relatch_alu_formats();
                pack_init(dfb::step);
                ckl::mul<ckl::input(dfb::partials_in), held_col(dfb::col_step), ckl::output(dfb::step)>(
                    ckl::IterationShape::tiles(d_share_tiles + 1), partials_in, col_step, step);
                col_step.pop_front(1);  // beta
                merge_block(d_share_tiles);
            }

            // ---- normalize_block: out = O / l.
            pack_init(dfb::col_step);
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::CopyTile<ckl::input(dfb::sum_run)>{sum_run},
                ckl::Recip<ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(dfb::col_step)>{col_step});
            relatch_alu_formats();
            pack_init(dfb::output_tiles);
            ckl::mul<ckl::input(dfb::out_acc), held_col(dfb::col_step), ckl::output(dfb::output_tiles)>(
                ckl::IterationShape::tiles(d_share_tiles), out_acc, col_step, output_tiles);
            col_step.pop_front(1);
            max_run.wait_front(1);
            dummy_unpack(dfb::max_run);
            max_run.pop_front(1);
        }
    }

    dummy_unpack(dfb::reduce_scaler);
    scaler.pop_front(1);
}
