// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for rms_norm.
//
//   out[..., h, :] = x[..., h, :] * rsqrt(mean(x^2) + eps) * gamma[:]
//
// Row-parallel: this core loops its num_rows tile-rows; each tile-row is a
// bounded two-pass streaming reduce over NUM_BLOCKS blocks of BLOCK_SIZE tiles.
//
//   Pass 1 (per block): [RM tilize x] -> square -> accumulate_reduce_block(SUM,1/W)
//           -> cb_rstd = mean(x^2).  Then transform_in_place: +eps, rsqrt -> 1/RMS.
//   Pass 2 (per block): [RM tilize x] -> mul<Col>(x, rstd) -> cb_norm
//           -> mul<Row>(norm, gamma) / copy -> cb_out  [-> RM untilize].
//
// All compute goes through the kernel-lib helpers. The one raw-LLK block is the
// +eps/rsqrt finalize inside transform_in_place's lambda: eps must be added
// (runtime scalar) BEFORE rsqrt on the same DST tile, which is exactly the
// 1-tile in-DST finalize transform_in_place exists for (streaming_reduce_helpers
// .hpp:104-105) — its inner ops are raw SFPU calls by that helper's design.
//
// Refinement 3 (resident single-read fast-path, op_design.md §1 lamp 1): a TILE
// input that fits L1 holds its whole tile-row (and gamma) resident so BOTH passes
// read x from L1 — no 2nd DRAM read. The resident square / x*rstd / *gamma steps
// use `eltwise_chain` DIRECTLY (BinaryFpu + PackTile) rather than the square<>/mul<>
// convenience wrappers, ONLY because those wrappers do not expose the TileOffset
// template parameter this path needs to index block b of the held row at absolute
// front offset b*BLOCK_SIZE. This is the SAME eltwise helper, just its lower-level
// composable form — not a raw-LLK substitution. The streaming path (fallback) keeps
// the square<>/mul<>/copy<> convenience wrappers unchanged.
//
// Deviation (forced, advisory): the streaming-reduce wrapper
// accumulate_reduce_block is stale in this kernel_lib checkout — it calls
// reduce<>() with the CBs as RUNTIME args, but reduce<>()'s current signature
// takes the CB ids as TEMPLATE params, so the wrapper does not compile. We call
// the underlying reduce<> helper directly (still the reduce helper, not raw LLK)
// with Accumulate::at(cb_rstd, b) for cross-block accumulation and route the
// partial scaler to the last block — exactly what accumulate_reduce_block would
// have done (streaming_reduce_helpers.inl:30-51). transform_in_place from the
// same header is unaffected (it never calls reduce<>).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t cb_x_sticks = 0;
constexpr uint32_t cb_x_in = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_gamma = 3;
constexpr uint32_t cb_gamma_sticks = 4;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_out_sticks = 17;
constexpr uint32_t cb_xsq = 24;
constexpr uint32_t cb_rstd = 25;
constexpr uint32_t cb_norm = 26;
}  // namespace

void kernel_main() {
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(1);
    constexpr bool IS_RM = get_compile_time_arg_val(2) != 0;
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(3) != 0;
    constexpr bool HAS_PARTIAL_W = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t eps_bits = get_compile_time_arg_val(5);
    // RM gamma -> compute tilizes cb_gamma_sticks -> cb_gamma. TILE gamma ->
    // the reader already filled cb_gamma with tiles, so the tilize is skipped
    // (op_design.md §5 tiled-gamma knob-turn, Refinement 2).
    constexpr bool GAMMA_IS_RM = get_compile_time_arg_val(6) != 0;
    // Resident single-read fast-path (op_design.md §1 lamp 1, Refinement 3): the
    // whole tile-row (Wt tiles) is held in cb_x_in so BOTH passes read it from L1
    // (no 2nd DRAM read), and gamma is held resident across the core's rows. Only
    // the input CB policy + CB sizing change vs the streaming path — the compute
    // phase sequence is identical (square -> reduce -> rsqrt -> x*rstd -> *gamma).
    constexpr bool USE_RESIDENT = get_compile_time_arg_val(7) != 0;
    // HEIGHT_SHARDED local per-core reduction (op_design.md §1 lamp 3, Refinement 5):
    // this core's full-W row-shard is resident in L1, so cb_x_in is a zero-copy CB
    // backed on the sharded buffer and cb_out is zero-copy on the sharded output (no
    // writer drains it). Compute SELF-ARMS cb_x_in (there is no reader push) and packs
    // cb_out in place. Knob-turn on the USE_RESIDENT path: the two-pass indexed compute
    // is identical — only the input arming + the (whole-shard) cb_out sizing change.
    constexpr bool X_RESIDENT = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t Wt = BLOCK_SIZE * NUM_BLOCKS;  // whole tile-row width

    const uint32_t num_rows = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_x_in, cb_scaler, cb_out);

    // One tile-row per pass (ROWS_PER_CALL=1): reduce shape rows=1.
    constexpr auto reduce_shape = ckl::ReduceInputBlockShape::of(1, BLOCK_SIZE, 1);
    // 2D shape (Ht=1) so the Col/Row broadcast operands index correctly.
    constexpr auto block_shape = ckl::EltwiseShape::of(1, BLOCK_SIZE);
    // Partial scaler tile (idx 1) on the last W-tile of the last block.
    constexpr auto partial_scaler =
        HAS_PARTIAL_W ? ckl::ReducePartialScaler::last_tile_at(1) : ckl::ReducePartialScaler::none();

    // ---- Resident single-read fast-path (TILE input, TILE/no gamma) ----
    // Block-offset resident: the whole tile-row (Wt tiles) is held in cb_x_in —
    // waited ONCE per row and popped ONCE per row — and each of the NUM_BLOCKS
    // blocks reads its slice at the absolute front offset b*BLOCK_SIZE via
    // OperandKind::Block + TileOffset::Set + InputLifecycle::CallerManaged (the
    // chain neither waits nor pops; the row-level wait/pop bracket own that). So
    // BOTH passes read the same L1 tiles — no 2nd DRAM read — while cb_xsq/cb_norm
    // stay 2*BLOCK_SIZE, so this fits EVERY prefill width (unlike a whole-row
    // block). gamma is likewise held resident: waited once at core entry, read per
    // block at offset b*BLOCK_SIZE (CallerManaged), popped once at core exit. The
    // compute phase sequence is otherwise identical to the streaming path below.
    if constexpr (USE_RESIDENT) {
        // HEIGHT_SHARDED (Refinement 5): the whole resident row-shard (num_rows tile-rows
        // x Wt tiles) is backed zero-copy on cb_x_in with no external producer, so
        // self-arm it once. The per-row cb_wait_front/pop below then walks the shard
        // tile-row by tile-row, keeping the front at the current row — so the block
        // offset indexing (b*BLOCK_SIZE, relative to the front) is IDENTICAL to the R3
        // resident path. cb_out is likewise zero-copy on the output shard (sized to the
        // whole shard, no writer drains it): the Streaming PackTile below fills it
        // exactly, so no pop is needed.
        if constexpr (X_RESIDENT) {
            cb_reserve_back(cb_x_in, num_rows * Wt);
            cb_push_back(cb_x_in, num_rows * Wt);
        }
        // Interleaved resident path holds the whole gamma row resident (read once by the
        // reader). HEIGHT (X_RESIDENT) instead STREAMS gamma per block (small cb_gamma):
        // a full-W gamma held resident (Wt tiles) would blow L1 on top of the resident
        // input+output shards for wide W, so gamma is re-read per block per row (phase-1
        // gamma behavior; resident gamma is the R6 perf lamp). No pre-loop wait/pop then.
        if constexpr (HAS_GAMMA && !X_RESIDENT) {
            cb_wait_front(cb_gamma, Wt);  // resident gamma: read once by the reader, held
        }
        for (uint32_t row = 0; row < num_rows; ++row) {
            cb_wait_front(cb_x_in, Wt);  // whole tile-row resident (read once from DRAM)

            // ---------- Pass 1: mean(x^2) ----------
            for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                // x^2 for block b: read cb_x_in tiles [b*BS, b*BS+BS) (no wait/pop).
                ckl::eltwise_chain(
                    block_shape,
                    ckl::BinaryFpu<
                        cb_x_in,
                        cb_x_in,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::None,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block,
                        ckl::OperandKind::Block,
                        ckl::TileOffset::Set,
                        ckl::TileOffset::Set>{b * BLOCK_SIZE, b * BLOCK_SIZE},
                    ckl::PackTile<cb_xsq, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
                // Cross-block accumulating reduce: partial scaler only on the last block.
                const ckl::ReducePartialScaler ps =
                    (b + 1 == NUM_BLOCKS) ? partial_scaler : ckl::ReducePartialScaler::none();
                ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, cb_xsq, cb_scaler, cb_rstd>(
                    reduce_shape,
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::Accumulate::at(cb_rstd, b),
                    ckl::NoOp{},
                    ps);
            }

            // +eps, rsqrt -> 1/RMS (held across pass 2).
            ckl::transform_in_place(cb_rstd, [](uint32_t dst) {
                binop_with_scalar_tile_init();
                add_unary_tile(dst, eps_bits);
                rsqrt_tile_init();
                rsqrt_tile(dst);
            });

            // ---------- Pass 2: x * rstd * gamma (same resident tiles) ----------
            for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                // x * rstd for block b: read cb_x_in at offset b*BS (no wait/pop);
                // rstd held (1 tile), popped at row end.
                ckl::eltwise_chain(
                    block_shape,
                    ckl::BinaryFpu<
                        cb_x_in,
                        cb_rstd,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Col,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::InputLifecycle::HeldBulk,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block,
                        ckl::OperandKind::Col,
                        ckl::TileOffset::Set,
                        ckl::TileOffset::Unset>{b * BLOCK_SIZE, 0},
                    ckl::PackTile<cb_norm, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});

                if constexpr (HAS_GAMMA) {
                    if constexpr (X_RESIDENT) {
                        // HEIGHT: gamma streamed per block (reader pushes BLOCK_SIZE/block);
                        // Bulk = wait+pop the block's gamma tiles. Same streaming ·gamma mul
                        // as the interleaved streaming path below (small cb_gamma, any W).
                        if constexpr (GAMMA_IS_RM) {
                            // RM gamma: tilize the block's sticks the reader pushed to
                            // cb_gamma_sticks -> cb_gamma (mirror of the interleaved RM-gamma).
                            ckl::tilize<BLOCK_SIZE, cb_gamma_sticks, cb_gamma>(1);
                        }
                        ckl::mul<
                            cb_norm,
                            cb_gamma,
                            cb_out,
                            ckl::BroadcastDim::Row,
                            ckl::InputLifecycle::Streaming,
                            ckl::InputLifecycle::Bulk,
                            ckl::OutputLifecycle::Streaming,
                            ckl::BinaryDataFormatReconfig::Input,
                            ckl::PackTileReconfig::Output,
                            ckl::OperandKind::Scalar,
                            ckl::OperandKind::Row>(block_shape);
                    } else {
                        // interleaved resident: gamma held, read at offset b*BS.
                        ckl::eltwise_chain(
                            block_shape,
                            ckl::BinaryFpu<
                                cb_norm,
                                cb_gamma,
                                ckl::BinaryFpuOp::Mul,
                                ckl::BroadcastDim::Row,
                                ckl::InputLifecycle::Streaming,
                                ckl::InputLifecycle::CallerManaged,
                                ckl::BinaryDataFormatReconfig::Input,
                                ckl::Dst::D0,
                                ckl::OperandKind::Scalar,
                                ckl::OperandKind::Row,
                                ckl::TileOffset::Unset,
                                ckl::TileOffset::Set>{0, b * BLOCK_SIZE},
                            ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
                    }
                } else {
                    ckl::copy<cb_norm, cb_out>(block_shape);
                }
            }

            cb_pop_front(cb_x_in, Wt);  // release the resident tile-row
            cb_pop_front(cb_rstd, 1);   // release the held 1/RMS tile
        }
        if constexpr (HAS_GAMMA && !X_RESIDENT) {
            cb_pop_front(cb_gamma, Wt);  // release the resident gamma row (interleaved only)
        }
        cb_pop_front(cb_scaler, HAS_PARTIAL_W ? 2 : 1);
        return;
    }

    for (uint32_t row = 0; row < num_rows; ++row) {
        // ---------- Pass 1: mean(x^2) ----------
        for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
            if constexpr (IS_RM) {
                ckl::tilize<BLOCK_SIZE, cb_x_sticks, cb_x_in>(1);
            }
            ckl::square<cb_x_in, cb_xsq>(block_shape);
            // Cross-block accumulating reduce: partial scaler only on the last block.
            const ckl::ReducePartialScaler ps =
                (b + 1 == NUM_BLOCKS) ? partial_scaler : ckl::ReducePartialScaler::none();
            ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, cb_xsq, cb_scaler, cb_rstd>(
                reduce_shape,
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::Accumulate::at(cb_rstd, b),
                ckl::NoOp{},
                ps);
        }

        // ---------- +eps, rsqrt -> 1/RMS (held across pass 2) ----------
        ckl::transform_in_place(cb_rstd, [](uint32_t dst) {
            binop_with_scalar_tile_init();
            add_unary_tile(dst, eps_bits);  // eps encoded as fp32 bits
            rsqrt_tile_init();
            rsqrt_tile(dst);
        });

        // ---------- Pass 2: x * rstd * gamma ----------
        for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
            if constexpr (IS_RM) {
                ckl::tilize<BLOCK_SIZE, cb_x_sticks, cb_x_in>(1);
            }
            // x * rstd (rstd is REDUCE_ROW result -> column-shaped -> BroadcastDim::Col).
            // A = cb_x_in streamed/popped; B = cb_rstd held (1 tile, popped at row end).
            ckl::mul<
                cb_x_in,
                cb_rstd,
                cb_norm,
                ckl::BroadcastDim::Col,
                ckl::InputLifecycle::Streaming,
                ckl::InputLifecycle::HeldBulk,
                ckl::OutputLifecycle::Streaming,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::PackTileReconfig::Output,
                ckl::OperandKind::Scalar,
                ckl::OperandKind::Col>(block_shape);

            if constexpr (HAS_GAMMA) {
                if constexpr (GAMMA_IS_RM) {
                    // RM gamma: tilize the sticks the reader pushed to cb_gamma_sticks.
                    ckl::tilize<BLOCK_SIZE, cb_gamma_sticks, cb_gamma>(1);
                }
                // else TILE gamma: reader already pushed BLOCK_SIZE tiles to cb_gamma.
                // norm * gamma (gamma is [1,W] -> row-shaped -> BroadcastDim::Row).
                // B = cb_gamma held-bulk (Wt tiles per block, popped at chain end).
                ckl::mul<
                    cb_norm,
                    cb_gamma,
                    cb_out,
                    ckl::BroadcastDim::Row,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::Bulk,
                    ckl::OutputLifecycle::Streaming,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::PackTileReconfig::Output,
                    ckl::OperandKind::Scalar,
                    ckl::OperandKind::Row>(block_shape);
            } else {
                ckl::copy<cb_norm, cb_out>(block_shape);
            }

            if constexpr (IS_RM) {
                ckl::untilize<BLOCK_SIZE, cb_out, cb_out_sticks>(1);
            }
        }

        cb_pop_front(cb_rstd, 1);  // release the held 1/RMS tile
    }

    cb_pop_front(cb_scaler, HAS_PARTIAL_W ? 2 : 1);
}
