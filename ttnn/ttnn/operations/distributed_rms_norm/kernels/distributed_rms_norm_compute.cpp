// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm — worker compute.
//
// Block operations (op_design.md "Block Operation Realization"), one pass per block (= one exchange round; a core
// runs num_rounds blocks of block_h tile rows, see the schedule at the end of kernel_main):
//   2a sumsq   : S[r] = sum_w x[r,w]^2 (eltwise_chain, Mul with per-row DEST accumulation) -> cb_sumsq
//   2b         : transpose S in place (cb_sumsq -> cb_sumsq_t)
//   2c         : REDUCE_COL (Accurate fp32) -> partial row vector per tile row -> cb_partial_row (wire format)
//   5  scale   : xg = x * gamma (Row broadcast) -> cb_xg (stall shadow, before the stats arrive)
//   6  combine : leader only, REDUCE_COL over the G * num_w_chunks gathered vectors, post-op x(1/W), +eps, rsqrt
//   8a         : transpose inv row -> col in place (cb_inv_row -> cb_inv_col)
//   8b normalize: out = xg (or x) * inv (Col broadcast) -> cb_out_tiles
//   pre-combine (leader, WIDTH shards): REDUCE_COL over the row-group's num_w_chunks partials -> cb_partial_row (a
//      second push: the one vector the leader publishes), before the exchange
//   streamed gamma (wide shards, no stall shadow): 8a, then inv col -> inv broadcast over columns (UnaryBcast Col,
//      in place) and one fused pass out = (x * gamma[Row bcast, streamed]) * inv_full (DestReuseBinary)
// A sharded x / out is the core's own shard behind cb_x_block / cb_out_tiles; the walk is the same, only the reader
// pushes exactly h * w_act pages per round (no pad pages).
//
// Raw-LLK substitutions (justified; measured in perf_experiments/, changelog "Perf 1" helper-bypass table):
//   * 2a-2c fused (32-bit DEST): x*x is accumulated per tile row in DEST (FPU mul_tiles, acc_to_dest), transposed in
//     DEST (transpose_dest) and column-folded on the SFPU in DEST (sfpu_reduce REDUCE_COL, fp32) -> one pack of the
//     partial row vector. Bit-identical to the 3-pass helper path (eltwise_chain + transpose + ckl::reduce Accurate)
//     and ~400 ns faster per tile row (1442 -> 1040 ns kernel at h=1, w=7; perf_experiments/fused_sumsq_dest).
//     Capability gap: eltwise_chain has no in-DEST transpose / SFPU-reduce element, and ckl::reduce cannot take a tile
//     already in DEST. Carve-out: a 16-bit DEST keeps the 3-pass path (the fused SFPU fold measured a -3e-3 relative
//     bias on the sums vs the FPU (Fast) reduce there).
//   * 6 post-op scoped to row 0 (VectorMode::R, 2 iterations = row-group 0 of faces 0/1, where REDUCE_COL leaves the
//     32 sums): mul_unary_tile / add_unary_tile / rsqrt_tile hard-code VectorMode::RC x 8 iterations, so the stock
//     ckernel::sfpu bodies are called through SFPU_UNARY_CALL. Row 0 bit-identical; combine 1670 -> 785 ns at h=1
//     (perf_experiments/combine_sfpu_scope). Ergonomics gap: the compute API exposes neither knob.
//   * 2b / 8a transpose: kernel_lib has no transpose element (no Transpose<> chain element; reduce's internal
//     transpose is not exposed). Closing gap: a `Transpose<InputSpec, Dst>` chain element.
//   * 6 post-op SFPU calls: reduce()'s post_reduce_op is a void(uint32_t dst) callback, not a chain, so the bodies of
//     the MulUnary / AddUnary / Rsqrt elements are called directly inside it.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "api/compute/transpose_dest.h"
#include "api/compute/eltwise_binary.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_binop_with_unary.h"
#endif

namespace ckl = compute_kernel_lib;

// fp32 partials are reduced on the SFPU (Accurate) only with a 32-bit DEST: the helper's SFPU fold reads DEST in the
// input CB's format (Float32), which a 16-bit DEST does not hold. With a 16-bit DEST the sums are 16-bit anyway, so
// the FPU path (Fast; Float32 pages unpack to Float16_b in srcA) loses nothing.
constexpr ReduceFp32Mode kReduceFp32Mode = DST_ACCUM_MODE ? ReduceFp32Mode::Accurate : ReduceFp32Mode::Fast;

constexpr uint32_t cb_x_block = 0;
constexpr uint32_t cb_gamma = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_sumsq = 3;
constexpr uint32_t cb_sumsq_t = 4;
constexpr uint32_t cb_inv_row_out = 5;
constexpr uint32_t cb_partial_row = 6;
constexpr uint32_t cb_gather = 7;
constexpr uint32_t cb_inv_row = 8;
constexpr uint32_t cb_inv_col = 9;
constexpr uint32_t cb_xg = 10;
constexpr uint32_t cb_out_tiles = 11;
constexpr uint32_t cb_pregather = 12;
constexpr uint32_t cb_inv_full = 13;

// Row-0-scoped SFPU epilogue bodies (stock ckernel::sfpu bodies; the compute API hard-codes VectorMode::RC x 8)
template <int IT>
ALWI void mul_unary_scoped(uint32_t idst, uint32_t param, ckernel::VectorMode vm) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, ckernel::MUL_UNARY, IT, DST_ACCUM_MODE),
        idst,
        vm,
        param));
}
template <int IT>
ALWI void add_unary_scoped(uint32_t idst, uint32_t param, ckernel::VectorMode vm) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, ckernel::ADD_UNARY, IT, DST_ACCUM_MODE),
        idst,
        vm,
        param));
}
template <int IT>
ALWI void rsqrt_scoped(uint32_t idst, ckernel::VectorMode vm) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt, (APPROX, IT, DST_ACCUM_MODE, false /*FAST_APPROX*/), idst, vm));
}

// Transpose n fp32 tiles of icb into ocb, where ocb aliases icb's bytes (in place: each DEST group is unpacked
// before it is packed back to the same positions).
ALWI void transpose_block_in_place(uint32_t icb, uint32_t ocb, uint32_t n) {
    reconfig_data_format_srca(icb);
    pack_reconfig_data_format(ocb);
    transpose_init(icb);
    cb_wait_front(icb, n);
    cb_reserve_back(ocb, n);
    for (uint32_t i = 0; i < n; i += ckl::DEST_AUTO_LIMIT) {
        const uint32_t m = (n - i) < ckl::DEST_AUTO_LIMIT ? (n - i) : ckl::DEST_AUTO_LIMIT;
        tile_regs_acquire();
        for (uint32_t j = 0; j < m; ++j) {
            transpose_tile(icb, i + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < m; ++j) {
            pack_tile(j, ocb);
        }
        tile_regs_release();
    }
    cb_pop_front(icb, n);
    cb_push_back(ocb, n);
}

void kernel_main() {
    constexpr uint32_t block_h = get_compile_time_arg_val(0);
    constexpr uint32_t block_w = get_compile_time_arg_val(1);
    constexpr bool has_gamma = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t n_gather_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t num_rounds = get_compile_time_arg_val(4);
    constexpr uint32_t round_stride = get_compile_time_arg_val(5);
    constexpr bool x_sharded = get_compile_time_arg_val(6) != 0;
    constexpr bool pre_combine = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t n_pre_tiles = get_compile_time_arg_val(8);
    constexpr bool gamma_stream = get_compile_time_arg_val(9) != 0;
    constexpr bool stall_shadow = has_gamma && !gamma_stream;
    // leader combines round r+1 before normalizing round r (cb_inv_row_out then has its own 2-round buffer)
    constexpr bool combine_ahead = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t x_block_pages =
        block_h * block_w;  // interleaved: the reader pushes the whole capacity per round

    const uint32_t row0 = get_arg_val<uint32_t>(0);
    const uint32_t row_end = get_arg_val<uint32_t>(1);
    const uint32_t w_act = get_arg_val<uint32_t>(2);
    const bool is_leader = get_arg_val<uint32_t>(3) != 0;
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(4);
    const uint32_t eps_bits = get_arg_val<uint32_t>(5);

    auto rows_of = [&](uint32_t r) -> uint32_t {
        const uint32_t first = row0 + r * round_stride;
        if (first >= row_end) {
            return 0;
        }
        return (row_end - first) < block_h ? (row_end - first) : block_h;
    };

    compute_kernel_hw_startup(cb_x_block, cb_x_block, cb_partial_row);

    // pages of the round's x push that carry no data (ragged chunk / short last round): popped unread
    auto pop_x_pad = [&](uint32_t h) {
        if constexpr (x_sharded) {
            return;  // a shard round pushes exactly h * w_act pages
        }
        const uint32_t pad = x_block_pages - h * w_act;
        if (pad > 0) {
            cb_wait_front(cb_x_block, pad);
            cb_pop_front(cb_x_block, pad);
        }
    };

    // Pre-combine: the leader's cb_partial_row stream interleaves its own partials with the combined vectors, so a
    // short last round would leave the next (full) block straddling the FIFO end, and the no-pop reduce packs its
    // outputs linearly from the write pointer. Every push is padded to block_h pages (the writer pops block_h).
    auto pad_partial = [&](uint32_t h) {
        if constexpr (pre_combine) {
            if (h < block_h) {
                cb_reserve_back(cb_partial_row, block_h - h);
                cb_push_back(cb_partial_row, block_h - h);
            }
        }
    };

    // ---- sumsq_block(r): 2a S = sum_w x^2 (x stays resident: no pop), 2b transpose, 2c REDUCE_COL -> partial row
    auto sumsq = [&](uint32_t h) {
        {
            MaybeDeviceZoneScope("C_xwait");  // starved on the reader's x(r) (unpack thread)
            cb_wait_front(cb_x_block, h * w_act);
        }
        MaybeDeviceZoneScope("C_sumsq");
        if constexpr (DST_ACCUM_MODE) {
            // fused (raw LLK, see the header): S = sum_w x*x stays in DEST; transpose + SFPU column fold DEST -> DEST
            reconfig_data_format(cb_x_block, cb_x_block);
            pack_reconfig_data_format(cb_partial_row);
            for (uint32_t r0 = 0; r0 < h; r0 += ckl::DEST_AUTO_LIMIT) {
                const uint32_t m = (h - r0) < ckl::DEST_AUTO_LIMIT ? (h - r0) : ckl::DEST_AUTO_LIMIT;
                cb_reserve_back(cb_partial_row, m);
                tile_regs_acquire();
                mul_init(cb_x_block, cb_x_block, /*acc_to_dest=*/true);
                for (uint32_t j = 0; j < m; ++j) {
                    const uint32_t base = (r0 + j) * w_act;
                    for (uint32_t c = 0; c < w_act; ++c) {
                        mul_tiles(cb_x_block, cb_x_block, base + c, base + c, j);
                    }
                }
                transpose_dest_init</*is_32bit=*/true, true>(cb_x_block);
                for (uint32_t j = 0; j < m; ++j) {
                    transpose_dest<true, true>(j);
                }
                sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
                for (uint32_t j = 0; j < m; ++j) {
                    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < m; ++j) {
                    pack_tile(j, cb_partial_row);
                }
                tile_regs_release();
                cb_push_back(cb_partial_row, m);
            }
        } else {
            // 16-bit DEST carve-out: 2a/2b/2c through the helpers (FPU Fast reduce; see the header)
            constexpr auto x_resident =
                ckl::input(cb_x_block, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::InputTileMapping::Block);
            ckl::eltwise_chain(
                ckl::IterationShape::grid(h, w_act),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    x_resident,
                    x_resident,
                    ckl::Dst::D0,
                    ckl::DestAccumulation::PerRow>{},
                ckl::PackTile<ckl::output(
                    cb_sumsq,
                    ckl::ReservePolicy::PerOuter,
                    ckl::PushPolicy::PerOuter,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileAddressing::Direct,
                    ckl::DestAccumulation::PerRow)>{});

            // 2b: transpose S (row sums become a column reduce)
            transpose_block_in_place(cb_sumsq, cb_sumsq_t, h);

            // 2c: REDUCE_COL -> partial row vector (row 0 = [v0..v31])
            ckl::reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_COL,
                cb_sumsq_t,
                cb_scaler,
                cb_partial_row,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                kReduceFp32Mode>(ckl::ReduceInputBlockShape::of(1, h));
            // WaitUpfrontNoPop hands the pop to the caller (BulkWaitBulkPop would need a DEST-chunk-sized CB)
            cb_pop_front(cb_sumsq_t, h);
        }
        pad_partial(h);
    };

    // ---- scale_by_gamma_block(r): stall shadow, xg = x * gamma (fp32); consumes x(r). Gamma stays for every round.
    auto scale_by_gamma = [&](uint32_t h) {
        if constexpr (stall_shadow) {
            MaybeDeviceZoneScope("C_xg");
            ckl::eltwise_chain(
                ckl::IterationShape::grid(h, w_act),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    ckl::input(cb_x_block),
                    ckl::input(
                        cb_gamma,
                        ckl::BroadcastDim::Row,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::None,
                        ckl::InputTileMapping::Row)>{},
                ckl::PackTile<ckl::output(cb_xg)>{});
            pop_x_pad(h);
        }
    };

    // ---- pre_combine_block(r): leader only, sum the row-group's num_w_chunks partials -> the vector it publishes
    auto pre_combine_block = [&](uint32_t h) {
        if constexpr (pre_combine) {
            if (!is_leader) {
                return;
            }
            MaybeDeviceZoneScope("C_precombine");
            ckl::reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_COL,
                cb_pregather,
                cb_scaler,
                cb_partial_row,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                kReduceFp32Mode>(ckl::ReduceInputBlockShape::of(n_pre_tiles, h));
            // WaitUpfrontNoPop hands the pop to the caller (pregather tiles are row-major: tile = t * h + rt)
            cb_pop_front(cb_pregather, n_pre_tiles * h);
            pad_partial(h);
        }
    };

    // ---- combine_block(r): leader only, sum G * num_w_chunks partials, inv_rms = rsqrt(ss / W + eps)
    auto combine = [&](uint32_t h) {
        if (!is_leader) {
            return;
        }
        {
            MaybeDeviceZoneScope("C_gwait");  // starved on the leader reader's gather (unpack thread)
            cb_wait_front(cb_gather, n_gather_tiles * h);
        }
        MaybeDeviceZoneScope("C_combine");
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            cb_gather,
            cb_scaler,
            cb_inv_row_out,
            ckl::ReduceInputPolicy::WaitUpfrontNoPop,
            ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            kReduceFp32Mode>(
            ckl::ReduceInputBlockShape::of(n_gather_tiles, h),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [inv_w_bits, eps_bits](uint32_t dst) {
                // row 0 only (VectorMode::R x 2 iterations = row-group 0 of faces 0/1): REDUCE_COL leaves the 32 sums
                // there and nothing downstream reads the rest of the tile
                binop_with_scalar_tile_init();
                mul_unary_scoped<2>(dst, inv_w_bits, ckernel::VectorMode::R);
                add_unary_scoped<2>(dst, eps_bits, ckernel::VectorMode::R);
                rsqrt_tile_init();
                rsqrt_scoped<2>(dst, ckernel::VectorMode::R);
            });
        // WaitUpfrontNoPop hands the pop to the caller (gather tiles are row-major: tile = t * h + rt)
        cb_pop_front(cb_gather, n_gather_tiles * h);
    };

    // ---- normalize_block(r): 8a inv row -> column vector, 8b out = xg (or x) * inv_rms, Col broadcast
    auto normalize = [&](uint32_t h) {
        {
            MaybeDeviceZoneScope("C_invwait");  // starved on inv_rms (exchange chain + leader fan-out)
            cb_wait_front(cb_inv_row, h);
        }
        MaybeDeviceZoneScope("C_norm");
        transpose_block_in_place(cb_inv_row, cb_inv_col, h);
        if constexpr (gamma_stream) {
            // inv col vector -> every column of the tile (in place: cb_inv_full aliases cb_inv_col's bytes), so the
            // fused pass can apply it as a plain DEST-reuse multiply after the Row-broadcast gamma multiply
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(h),
                ckl::UnaryBcast<
                    ckl::BroadcastDim::Col,
                    ckl::input(
                        cb_inv_col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block)>{},
                ckl::PackTile<ckl::output(cb_inv_full)>{});
            ckl::eltwise_chain(
                ckl::IterationShape::grid(h, w_act),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    ckl::input(cb_x_block),
                    ckl::input(cb_gamma, ckl::BroadcastDim::Row)>{},  // streamed: one sliver tile per step
                ckl::DestReuseBinary<
                    ckl::BinaryFpuOp::Mul,
                    ckl::input(
                        cb_inv_full, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Col),
                    ckl::DestReuseType::DEST_TO_SRCA>{},
                ckl::PackTile<ckl::output(cb_out_tiles)>{});
            pop_x_pad(h);
            return;
        }
        constexpr uint32_t cb_scaled = has_gamma ? cb_xg : cb_x_block;
        ckl::eltwise_chain(
            ckl::IterationShape::grid(h, w_act),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(cb_scaled),
                ckl::input(
                    cb_inv_col,
                    ckl::BroadcastDim::Col,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::AtEnd,
                    ckl::InputTileMapping::Col)>{},
            ckl::PackTile<ckl::output(cb_out_tiles)>{});
        if constexpr (!has_gamma) {
            pop_x_pad(h);
        }
    };

    // Per-core schedule over this row-slot's rounds (one row-block of block_h tile rows each; 0 rows = an empty last
    // round). With the stall shadow x(r) is consumed by scale_by_gamma, so round r+1's sumsq (and with it its exchange)
    // runs before round r's combine/normalize; otherwise x(r) is held until normalize(r) and rounds run back to back.
    if constexpr (combine_ahead) {
        // round r+1's stats are complete on the leader as soon as its exchange lands, which does not depend on round
        // r's inv arrival; combining it before normalize(r) takes round r's fan-out + normalize off its critical path
        if (rows_of(0) > 0) {
            sumsq(rows_of(0));
        }
        for (uint32_t r = 0; r < num_rounds; ++r) {
            const uint32_t h = rows_of(r);
            const uint32_t h_next = r + 1 < num_rounds ? rows_of(r + 1) : 0;
            if (h > 0) {
                scale_by_gamma(h);
            }
            if (h_next > 0) {
                sumsq(h_next);
            }
            if (r == 0 && h > 0) {
                combine(h);
            }
            if (h_next > 0) {
                combine(h_next);
            }
            if (h > 0) {
                normalize(h);
            }
        }
        cb_pop_front(cb_gamma, w_act);  // PopPolicy::None above hands gamma's single pop to the caller
    } else if constexpr (stall_shadow) {
        if (rows_of(0) > 0) {
            sumsq(rows_of(0));
        }
        for (uint32_t r = 0; r < num_rounds; ++r) {
            const uint32_t h = rows_of(r);
            if (h > 0) {
                scale_by_gamma(h);
            }
            if (r + 1 < num_rounds && rows_of(r + 1) > 0) {
                sumsq(rows_of(r + 1));
            }
            if (h > 0) {
                pre_combine_block(h);
                combine(h);
                normalize(h);
            }
        }
        cb_pop_front(cb_gamma, w_act);  // PopPolicy::None above hands gamma's single pop to the caller
    } else {
        for (uint32_t r = 0; r < num_rounds; ++r) {
            const uint32_t h = rows_of(r);
            if (h > 0) {
                sumsq(h);
                pre_combine_block(h);
                combine(h);
                normalize(h);
            }
        }
    }
}
