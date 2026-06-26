// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax compute kernel — V2 streaming path (constant-bounded CBs).
//
// Dispatched when the V1 (full-slab) CB footprint exceeds the L1 budget.
// Processes one tile-row (dim=-1) or tile-column (dim=-2) at a time,
// chunking along the reduce dimension in blocks of BLOCK_SIZE tiles.
//
// Three DRAM passes per tile-row/column:
//   Pass 1 (max):  chunk-wise reduce<MAX> → BinaryMax with running_max
//   Pass 2 (sum):  re-read, sub(x-max)+exp, reduce<SUM> → add to running_sum
//                  After all chunks: recip(running_sum) → cb_recip_sum
//   Pass 3 (apply): re-read, sub(x-max)+exp, mul(recip_sum) → output
//
// All CBs are bounded by BLOCK_SIZE (a host-chosen constant), not by Wt/Ht.
//
// Layout dispatch (CT arg is_rm):
//   TILE: reader feeds cb_input_tiles directly
//   RM:   reader feeds cb_rm_in, compute tilizes to cb_input_tiles per chunk,
//         untilize cb_output_tiles to cb_rm_out per chunk (pass 3 only)

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/recip.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace {
// CB indices — must match program descriptor
constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_scaler_max = 1;
constexpr uint32_t cb_scaler_sum = 2;
constexpr uint32_t cb_rm_in = 3;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_rm_out = 17;
constexpr uint32_t cb_max = 24;  // V2b (non-reduce chunking): per-chunk max
constexpr uint32_t cb_exp = 25;
constexpr uint32_t cb_recip_sum_v1 = 26;  // V2b: per-chunk recip_sum
// V2 persistent stats (1 tile each, persist across chunks within a tile-row/col)
constexpr uint32_t cb_running_max = 27;
constexpr uint32_t cb_running_sum = 28;
constexpr uint32_t cb_recip_sum = 29;
// V2 per-chunk intermediates (1 tile each)
constexpr uint32_t cb_chunk_max = 30;
constexpr uint32_t cb_chunk_sum = 31;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr int32_t dim = static_cast<int32_t>(get_compile_time_arg_val(2));
    constexpr uint32_t is_rm = get_compile_time_arg_val(3);
    constexpr uint32_t origin_W = get_compile_time_arg_val(4);
    constexpr uint32_t origin_H = get_compile_time_arg_val(5);
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(6);
    constexpr bool chunk_along_reduce = get_compile_time_arg_val(7);
    constexpr bool chunk_along_non_reduce = !chunk_along_reduce;

    // Partial scaler: needed when the REDUCTION axis is non-tile-aligned.
    constexpr uint32_t partial_W = origin_W % 32;
    constexpr uint32_t partial_H = origin_H % 32;
    constexpr bool has_partial = (dim == -1) ? (partial_W > 0) : (partial_H > 0);
    constexpr auto partial_scaler =
        has_partial ? ckl::ReducePartialScaler::last_tile_at(1) : ckl::ReducePartialScaler::none();

    uint32_t num_slabs = get_arg_val<uint32_t>(0);

    constexpr uint32_t reduce_dim_tiles = (dim == -1) ? Wt : Ht;
    constexpr uint32_t non_reduce_dim = (dim == -1) ? Ht : Wt;
    // For chunk_along_reduce: chunk the reduce dim (3-pass)
    // For chunk_along_non_reduce: chunk the non-reduce dim (1-pass V1-style per chunk)
    constexpr uint32_t chunked_dim = chunk_along_reduce ? reduce_dim_tiles : non_reduce_dim;
    constexpr uint32_t num_chunks = chunked_dim / BLOCK_SIZE;
    // Per-chunk tile grid: (chunk_h, chunk_w)
    // chunk_along_reduce: (non_reduce_dim, BLOCK_SIZE) for dim=-1, (BLOCK_SIZE, non_reduce_dim) for dim=-2
    // chunk_along_non_reduce: (BLOCK_SIZE, reduce_dim_tiles) for dim=-1, (reduce_dim_tiles, BLOCK_SIZE) for dim=-2
    constexpr uint32_t chunk_h =
        chunk_along_reduce ? ((dim == -1) ? 1 : BLOCK_SIZE) : ((dim == -1) ? BLOCK_SIZE : reduce_dim_tiles);
    constexpr uint32_t chunk_w =
        chunk_along_reduce ? ((dim == -1) ? BLOCK_SIZE : 1) : ((dim == -1) ? reduce_dim_tiles : BLOCK_SIZE);
    constexpr uint32_t tiles_per_chunk = chunk_h * chunk_w;

    if constexpr (is_rm) {
        compute_kernel_hw_startup(cb_rm_in, cb_input_tiles);
    } else {
        compute_kernel_hw_startup(cb_input_tiles, cb_scaler_max, cb_output_tiles);
    }

    for (uint32_t slab = 0; slab < num_slabs; ++slab) {
        if constexpr (chunk_along_non_reduce) {
            // ===== chunk_along_non_reduce: V1-style 4-phase per chunk =====
            // Each chunk processes chunk_h × chunk_w tiles (BLOCK_SIZE along
            // the non-reduce dim × full reduce dim).
            // cb_max (24) and cb_recip_sum (26) are sized to BLOCK_SIZE tiles.
            constexpr auto chunk_reduce_shape = (dim == -1) ? ckl::ReduceInputBlockShape::of(chunk_h, chunk_w, 1)
                                                            : ckl::ReduceInputBlockShape::of(chunk_h, chunk_w, 1);
            constexpr auto chunk_eltwise_shape = ckl::EltwiseShape::grid(chunk_h, chunk_w);

            for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                // RM path: tilize this chunk
                if constexpr (is_rm) {
                    ckl::tilize<
                        chunk_w,
                        cb_rm_in,
                        cb_input_tiles,
                        ckl::tilize_config::InitUninitMode::InitAndUninit,
                        ckl::tilize_config::WaitMode::WaitBlock,
                        ckl::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(chunk_h);
                }

                // Phase 1: max reduce (WaitUpfrontNoPop — input retained for Phase 2)
                if constexpr (dim == -1) {
                    ckl::reduce<
                        ckernel::PoolType::MAX,
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_input_tiles,
                        cb_scaler_max,
                        cb_max,
                        ckl::ReduceInputPolicy::WaitUpfrontNoPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::NoAccumulation,
                        ckl::NoOp>(
                        chunk_reduce_shape,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::NoAccumulation{},
                        ckl::NoOp{},
                        partial_scaler);
                } else {
                    ckl::reduce<
                        ckernel::PoolType::MAX,
                        ckernel::ReduceDim::REDUCE_COL,
                        cb_input_tiles,
                        cb_scaler_max,
                        cb_max,
                        ckl::ReduceInputPolicy::WaitUpfrontNoPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::NoAccumulation,
                        ckl::NoOp>(
                        chunk_reduce_shape,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::NoAccumulation{},
                        ckl::NoOp{},
                        partial_scaler);
                }

                // Phase 2: sub + exp (fused chain)
                if constexpr (dim == -1) {
                    ckl::eltwise_chain(
                        chunk_eltwise_shape,
                        ckl::BinaryFpu<
                            cb_input_tiles,
                            cb_max,
                            ckl::BinaryFpuOp::Sub,
                            ckl::BroadcastDim::Col,
                            ckl::InputLifecycle::Bulk,
                            ckl::InputLifecycle::HeldBulk,
                            ckl::BinaryDataFormatReconfig::Input,
                            ckl::Dst::D0,
                            ckl::OperandKind::Block,
                            ckl::OperandKind::Col>{},
                        ckl::Exp<>{},
                        ckl::PackTile<cb_exp, ckl::OutputLifecycle::Streaming>{});
                    cb_pop_front(cb_max, chunk_h);
                } else {
                    ckl::eltwise_chain(
                        chunk_eltwise_shape,
                        ckl::BinaryFpu<
                            cb_input_tiles,
                            cb_max,
                            ckl::BinaryFpuOp::Sub,
                            ckl::BroadcastDim::Row,
                            ckl::InputLifecycle::Bulk,
                            ckl::InputLifecycle::HeldBulk,
                            ckl::BinaryDataFormatReconfig::Input,
                            ckl::Dst::D0,
                            ckl::OperandKind::Block,
                            ckl::OperandKind::Row>{},
                        ckl::Exp<>{},
                        ckl::PackTile<cb_exp, ckl::OutputLifecycle::Streaming>{});
                    cb_pop_front(cb_max, chunk_w);
                }

                // Phase 3: sum + recip
                auto recip_op = [](uint32_t dst_idx) {
                    recip_tile_init();
                    recip_tile(dst_idx);
                };
                if constexpr (dim == -1) {
                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_exp,
                        cb_scaler_sum,
                        cb_recip_sum_v1,
                        ckl::ReduceInputPolicy::WaitUpfrontNoPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::NoAccumulation,
                        decltype(recip_op)>(
                        chunk_reduce_shape,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::NoAccumulation{},
                        recip_op,
                        partial_scaler);
                } else {
                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_COL,
                        cb_exp,
                        cb_scaler_sum,
                        cb_recip_sum_v1,
                        ckl::ReduceInputPolicy::WaitUpfrontNoPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::NoAccumulation,
                        decltype(recip_op)>(
                        chunk_reduce_shape,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::NoAccumulation{},
                        recip_op,
                        partial_scaler);
                }

                // Phase 4: mul (broadcast recip_sum)
                if constexpr (dim == -1) {
                    ckl::mul<
                        cb_exp,
                        cb_recip_sum_v1,
                        cb_output_tiles,
                        ckl::BroadcastDim::Col,
                        ckl::InputLifecycle::Bulk,
                        ckl::InputLifecycle::HeldBulk,
                        ckl::OutputLifecycle::Streaming,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::PackTileReconfig::Output,
                        ckl::OperandKind::Block,
                        ckl::OperandKind::Col>(chunk_eltwise_shape);
                    cb_pop_front(cb_recip_sum_v1, chunk_h);
                } else {
                    ckl::mul<
                        cb_exp,
                        cb_recip_sum_v1,
                        cb_output_tiles,
                        ckl::BroadcastDim::Row,
                        ckl::InputLifecycle::Bulk,
                        ckl::InputLifecycle::HeldBulk,
                        ckl::OutputLifecycle::Streaming,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::PackTileReconfig::Output,
                        ckl::OperandKind::Block,
                        ckl::OperandKind::Row>(chunk_eltwise_shape);
                    cb_pop_front(cb_recip_sum_v1, chunk_w);
                }

                // RM path: untilize this chunk
                if constexpr (is_rm) {
                    ckl::untilize<
                        chunk_w,
                        cb_output_tiles,
                        cb_rm_out,
                        ckl::untilize_config::InitUninitMode::InitAndUninit,
                        ckl::untilize_config::WaitMode::WaitBlock,
                        ckl::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(chunk_h);
                }
            }
        } else if constexpr (dim == -1) {
            // ===== chunk_along_reduce, dim=-1 (W reduction): one tile-row at a time =====
            constexpr auto reduce_block_shape = ckl::ReduceInputBlockShape::row(BLOCK_SIZE);
            constexpr auto eltwise_shape = ckl::EltwiseShape::grid(1, BLOCK_SIZE);

            for (uint32_t ht = 0; ht < Ht; ++ht) {
                // ===== PASS 1: compute global max for this tile-row =====
                for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                    if constexpr (is_rm) {
                        ckl::tilize<
                            BLOCK_SIZE,
                            cb_rm_in,
                            cb_input_tiles,
                            ckl::tilize_config::InitUninitMode::InitAndUninit,
                            ckl::tilize_config::WaitMode::WaitBlock,
                            ckl::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
                    }

                    // Partial scaler only on the last chunk's last tile
                    auto chunk_partial = (chunk == num_chunks - 1) ? partial_scaler : ckl::ReducePartialScaler::none();

                    // reduce<MAX, REDUCE_ROW, WaitUpfrontNoPop> → cb_chunk_max (1 tile)
                    // Input NOT popped — retained for pass 2
                    ckl::reduce<
                        ckernel::PoolType::MAX,
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_input_tiles,
                        cb_scaler_max,
                        cb_chunk_max,
                        ckl::ReduceInputPolicy::WaitUpfrontNoPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::NoAccumulation,
                        ckl::NoOp>(
                        reduce_block_shape,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::NoAccumulation{},
                        ckl::NoOp{},
                        chunk_partial);

                    if (chunk == 0) {
                        // First chunk: running_max = chunk_max
                        ckl::copy<cb_chunk_max, cb_running_max>(ckl::EltwiseShape::single());
                    } else {
                        // Subsequent: running_max = max(running_max, chunk_max)
                        ckl::binary_sfpu<ckl::BinaryMax<>, cb_running_max, cb_chunk_max, cb_running_max>(
                            ckl::EltwiseShape::single());
                    }

                    // Pop input tiles (reduce used WaitUpfrontNoPop — didn't pop)
                    cb_pop_front(cb_input_tiles, BLOCK_SIZE);
                }

                // ===== PASS 2: compute sum(exp(x - max)) =====
                for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                    if constexpr (is_rm) {
                        ckl::tilize<
                            BLOCK_SIZE,
                            cb_rm_in,
                            cb_input_tiles,
                            ckl::tilize_config::InitUninitMode::InitOnly,
                            ckl::tilize_config::WaitMode::WaitBlock,
                            ckl::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
                    }

                    // sub(x - running_max) + exp → cb_exp (BLOCK_SIZE tiles)
                    // cb_input_tiles: Bulk (pops at end). cb_running_max: HeldBulk (no pop).
                    ckl::eltwise_chain(
                        eltwise_shape,
                        ckl::BinaryFpu<
                            cb_input_tiles,
                            cb_running_max,
                            ckl::BinaryFpuOp::Sub,
                            ckl::BroadcastDim::Col,
                            ckl::InputLifecycle::Bulk,
                            ckl::InputLifecycle::HeldBulk,
                            ckl::BinaryDataFormatReconfig::Input,
                            ckl::Dst::D0,
                            ckl::OperandKind::Block,
                            ckl::OperandKind::Col>{},
                        ckl::Exp<>{},
                        ckl::PackTile<cb_exp, ckl::OutputLifecycle::Streaming>{});

                    // Pop running_max (HeldBulk left it unpopped — needed for pass 3)
                    // Actually, HeldBulk doesn't pop, but we need running_max to persist
                    // for pass 3. So we DON'T pop it here. We pop it after pass 3.

                    auto chunk_partial = (chunk == num_chunks - 1) ? partial_scaler : ckl::ReducePartialScaler::none();

                    // reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop> → cb_chunk_sum (1 tile)
                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_exp,
                        cb_scaler_sum,
                        cb_chunk_sum,
                        ckl::ReduceInputPolicy::WaitUpfrontNoPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::NoAccumulation,
                        ckl::NoOp>(
                        reduce_block_shape,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::NoAccumulation{},
                        ckl::NoOp{},
                        chunk_partial);

                    if (chunk == 0) {
                        ckl::copy<cb_chunk_sum, cb_running_sum>(ckl::EltwiseShape::single());
                    } else {
                        ckl::add<cb_running_sum, cb_chunk_sum, cb_running_sum>(ckl::EltwiseShape::single());
                    }

                    // Pop exp tiles (reduce didn't pop them)
                    cb_pop_front(cb_exp, BLOCK_SIZE);
                }

                // Compute recip(running_sum) → cb_recip_sum
                ckl::unary<ckl::Recip<>, cb_running_sum, cb_recip_sum>(ckl::EltwiseShape::single());

                // ===== PASS 3: apply exp(x - max) * recip_sum → output =====
                for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                    if constexpr (is_rm) {
                        ckl::tilize<
                            BLOCK_SIZE,
                            cb_rm_in,
                            cb_input_tiles,
                            ckl::tilize_config::InitUninitMode::InitOnly,
                            ckl::tilize_config::WaitMode::WaitBlock,
                            ckl::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
                    }

                    // sub(x - running_max) + exp → cb_exp
                    ckl::eltwise_chain(
                        eltwise_shape,
                        ckl::BinaryFpu<
                            cb_input_tiles,
                            cb_running_max,
                            ckl::BinaryFpuOp::Sub,
                            ckl::BroadcastDim::Col,
                            ckl::InputLifecycle::Bulk,
                            ckl::InputLifecycle::HeldBulk,
                            ckl::BinaryDataFormatReconfig::Input,
                            ckl::Dst::D0,
                            ckl::OperandKind::Block,
                            ckl::OperandKind::Col>{},
                        ckl::Exp<>{},
                        ckl::PackTile<cb_exp, ckl::OutputLifecycle::Streaming>{});

                    // mul(exp, recip_sum) → cb_output_tiles
                    ckl::mul<
                        cb_exp,
                        cb_recip_sum,
                        cb_output_tiles,
                        ckl::BroadcastDim::Col,
                        ckl::InputLifecycle::Bulk,
                        ckl::InputLifecycle::HeldBulk,
                        ckl::OutputLifecycle::Streaming,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::PackTileReconfig::Output,
                        ckl::OperandKind::Block,
                        ckl::OperandKind::Col>(eltwise_shape);

                    // Pop exp tiles (mul used Bulk on cb_exp, which pops at end)
                    // Actually, mul with InputLifecycle::Bulk on cb_exp pops at end.
                    // So cb_exp is already empty. No explicit pop needed.
                    // But wait — does mul's Bulk pop cb_exp? Let me check.
                    // mul<cb_exp, cb_recip_sum, cb_output_tiles, ..., Bulk, HeldBulk, ...>
                    // cb_exp is operand A with InputLifecycle::Bulk → waited upfront, popped at end. ✓
                    // cb_recip_sum is operand B with InputLifecycle::HeldBulk → waited upfront, NOT popped.
                    // So cb_recip_sum needs to be popped after the last chunk.
                    // Actually, HeldBulk doesn't pop. I need to pop it after all chunks.

                    // RM path: untilize cb_output_tiles → cb_rm_out
                    if constexpr (is_rm) {
                        ckl::untilize<
                            BLOCK_SIZE,
                            cb_output_tiles,
                            cb_rm_out,
                            ckl::untilize_config::InitUninitMode::InitAndUninit,
                            ckl::untilize_config::WaitMode::WaitBlock,
                            ckl::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
                    }
                }

                // Pop persistent stats for the next tile-row
                cb_pop_front(cb_running_max, 1);
                cb_pop_front(cb_recip_sum, 1);
                // cb_running_sum was already popped by unary<Recip> (Streaming lifecycle)
            }
        } else if constexpr (dim == -2) {
            // ===== chunk_along_reduce, dim=-2 (H reduction): one tile-column at a time =====
            constexpr auto reduce_block_shape = ckl::ReduceInputBlockShape::col(BLOCK_SIZE);
            constexpr auto eltwise_shape = ckl::EltwiseShape::grid(BLOCK_SIZE, 1);

            for (uint32_t wt = 0; wt < Wt; ++wt) {
                // ===== PASS 1: compute global max for this tile-column =====
                for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                    if constexpr (is_rm) {
                        // dim=-2 RM: tilize BLOCK_SIZE blocks of 1 tile each
                        ckl::tilize<
                            1,
                            cb_rm_in,
                            cb_input_tiles,
                            ckl::tilize_config::InitUninitMode::InitAndUninit,
                            ckl::tilize_config::WaitMode::WaitBlock,
                            ckl::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(BLOCK_SIZE);
                    }

                    auto chunk_partial = (chunk == num_chunks - 1) ? partial_scaler : ckl::ReducePartialScaler::none();

                    ckl::reduce<
                        ckernel::PoolType::MAX,
                        ckernel::ReduceDim::REDUCE_COL,
                        cb_input_tiles,
                        cb_scaler_max,
                        cb_chunk_max,
                        ckl::ReduceInputPolicy::WaitUpfrontNoPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::NoAccumulation,
                        ckl::NoOp>(
                        reduce_block_shape,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::NoAccumulation{},
                        ckl::NoOp{},
                        chunk_partial);

                    if (chunk == 0) {
                        ckl::copy<cb_chunk_max, cb_running_max>(ckl::EltwiseShape::single());
                    } else {
                        ckl::binary_sfpu<ckl::BinaryMax<>, cb_running_max, cb_chunk_max, cb_running_max>(
                            ckl::EltwiseShape::single());
                    }

                    cb_pop_front(cb_input_tiles, BLOCK_SIZE);
                }

                // ===== PASS 2: compute sum(exp(x - max)) =====
                for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                    if constexpr (is_rm) {
                        ckl::tilize<
                            1,
                            cb_rm_in,
                            cb_input_tiles,
                            ckl::tilize_config::InitUninitMode::InitOnly,
                            ckl::tilize_config::WaitMode::WaitBlock,
                            ckl::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(BLOCK_SIZE);
                    }

                    // sub(x - running_max) + exp → cb_exp
                    // For dim=-2: BroadcastDim::Row (max is row-shaped, broadcast across H rows)
                    ckl::eltwise_chain(
                        eltwise_shape,
                        ckl::BinaryFpu<
                            cb_input_tiles,
                            cb_running_max,
                            ckl::BinaryFpuOp::Sub,
                            ckl::BroadcastDim::Row,
                            ckl::InputLifecycle::Bulk,
                            ckl::InputLifecycle::HeldBulk,
                            ckl::BinaryDataFormatReconfig::Input,
                            ckl::Dst::D0,
                            ckl::OperandKind::Block,
                            ckl::OperandKind::Row>{},
                        ckl::Exp<>{},
                        ckl::PackTile<cb_exp, ckl::OutputLifecycle::Streaming>{});

                    auto chunk_partial = (chunk == num_chunks - 1) ? partial_scaler : ckl::ReducePartialScaler::none();

                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_COL,
                        cb_exp,
                        cb_scaler_sum,
                        cb_chunk_sum,
                        ckl::ReduceInputPolicy::WaitUpfrontNoPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::NoAccumulation,
                        ckl::NoOp>(
                        reduce_block_shape,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::NoAccumulation{},
                        ckl::NoOp{},
                        chunk_partial);

                    if (chunk == 0) {
                        ckl::copy<cb_chunk_sum, cb_running_sum>(ckl::EltwiseShape::single());
                    } else {
                        ckl::add<cb_running_sum, cb_chunk_sum, cb_running_sum>(ckl::EltwiseShape::single());
                    }

                    cb_pop_front(cb_exp, BLOCK_SIZE);
                }

                // recip(running_sum) → cb_recip_sum
                ckl::unary<ckl::Recip<>, cb_running_sum, cb_recip_sum>(ckl::EltwiseShape::single());

                // ===== PASS 3: apply exp(x - max) * recip_sum → output =====
                for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                    if constexpr (is_rm) {
                        ckl::tilize<
                            1,
                            cb_rm_in,
                            cb_input_tiles,
                            ckl::tilize_config::InitUninitMode::InitOnly,
                            ckl::tilize_config::WaitMode::WaitBlock,
                            ckl::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(BLOCK_SIZE);
                    }

                    // sub(x - running_max) + exp → cb_exp
                    ckl::eltwise_chain(
                        eltwise_shape,
                        ckl::BinaryFpu<
                            cb_input_tiles,
                            cb_running_max,
                            ckl::BinaryFpuOp::Sub,
                            ckl::BroadcastDim::Row,
                            ckl::InputLifecycle::Bulk,
                            ckl::InputLifecycle::HeldBulk,
                            ckl::BinaryDataFormatReconfig::Input,
                            ckl::Dst::D0,
                            ckl::OperandKind::Block,
                            ckl::OperandKind::Row>{},
                        ckl::Exp<>{},
                        ckl::PackTile<cb_exp, ckl::OutputLifecycle::Streaming>{});

                    // mul(exp, recip_sum) → cb_output_tiles
                    ckl::mul<
                        cb_exp,
                        cb_recip_sum,
                        cb_output_tiles,
                        ckl::BroadcastDim::Row,
                        ckl::InputLifecycle::Bulk,
                        ckl::InputLifecycle::HeldBulk,
                        ckl::OutputLifecycle::Streaming,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::PackTileReconfig::Output,
                        ckl::OperandKind::Block,
                        ckl::OperandKind::Row>(eltwise_shape);

                    // RM path: untilize cb_output_tiles → cb_rm_out
                    if constexpr (is_rm) {
                        ckl::untilize<
                            1,
                            cb_output_tiles,
                            cb_rm_out,
                            ckl::untilize_config::InitUninitMode::InitAndUninit,
                            ckl::untilize_config::WaitMode::WaitBlock,
                            ckl::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(
                            BLOCK_SIZE);
                    }
                }

                // Pop persistent stats for the next tile-column
                cb_pop_front(cb_running_max, 1);
                cb_pop_front(cb_recip_sum, 1);
            }
        }
    }
}
