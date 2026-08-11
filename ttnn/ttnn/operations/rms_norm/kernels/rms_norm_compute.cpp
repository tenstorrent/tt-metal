// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm compute (3 TRISCs).
//
// Per block (block_row_tiles x core_w_tiles tiles, resident in cb_input_tiles
// from sumsq_block through scale_block — which is what keeps the input to ONE
// DRAM crossing):
//
//   tilize_in_block   (ROW_MAJOR input only)
//   sumsq_block       Sum_c x[r,c]^2 accumulated ELEMENTWISE across the block's
//                     hidden tiles inside a persistent fp32 DEST accumulator
//                     (DestAccumulation::PerRow) -> one tile per tile-row.
//                     No block-sized x^2 scratch buffer exists.
//   mask_tail_block   only when this core owns the tensor's last hidden tile and
//                     W % 32 != 0: (x_tail * wmask)^2 -> the second stat column.
//                     mask BEFORE squaring, so a finite poison value is
//                     annihilated instead of overflowing.
//   reduce_stat_block reduce<SUM, REDUCE_ROW> folds the within-tile columns of
//                     the 1..2 stat columns into a column-0-valid tile.
//   combine (root)    the G per-core partials gathered by the writer are summed
//                     in a second DEST accumulation, then finalized
//                     x -> rsqrt(x/W_true + eps) into the multicast source.
//   scale_block       x * rstd            (BroadcastDim::Col)
//   gamma_block       normed * gamma      (BroadcastDim::Row), elided w/o gamma
//   untilize_out_block (ROW_MAJOR output only)
//
// UNIFORM BLOCK WIDTH. Every CB moves CB_W_TILES pages per tile-row on every
// core of a reduction group (required both by the peer-addressed cb_rstd /
// cb_stat_gather L1 map and by the CB "capacity is a multiple of the quantum"
// rule, dataflow_api.h:216-221). A core's VALID hidden slice is the runtime
// `core_w <= CB_W_TILES`; the statistics phases walk `core_w` columns at row
// stride CB_W_TILES, so the trailing pad tiles never enter the reduction. The
// apply phases DO cover the pad columns (uniform, contiguous, no stride) — their
// output is simply never written to DRAM by the writer.
//
// Helper substitutions: none. Every phase is a kernel_lib helper call
// (eltwise_chain / eltwise_convenience / reduce / tilize / untilize). The
// caller-managed (None, None) CB policies on sumsq_block and mask_tail_block are
// NOT a substitution — TileOffset::Strided *requires* them
// (eltwise_chain.inl:1169-1172), and strided addressing is what lets both
// phases index the block's hidden tiles at row stride core_w_tiles without
// copying them into a contiguous scratch buffer.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace {
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_wmask = 3;
constexpr uint32_t cb_zero_tile = 4;
constexpr uint32_t cb_stat_sq = 5;
constexpr uint32_t cb_stat_partial = 7;
constexpr uint32_t cb_stat_gather = 8;
constexpr uint32_t cb_stat_sum = 9;
constexpr uint32_t cb_rstd_send = 10;
constexpr uint32_t cb_rstd = 11;
constexpr uint32_t cb_gamma_rm = 12;
constexpr uint32_t cb_gamma_tiles = 13;
constexpr uint32_t cb_normed = 14;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_output_rm = 17;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t CB_W_TILES = get_compile_time_arg_val(0);
    constexpr uint32_t W_GROUP_SIZE = get_compile_time_arg_val(1);
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(2) != 0;
    constexpr bool IS_RM_IN = get_compile_time_arg_val(3) != 0;
    constexpr bool IS_RM_OUT = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(5);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(6);
    constexpr bool IS_RM_GAMMA = get_compile_time_arg_val(7) != 0;

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t block_row_tiles = get_arg_val<uint32_t>(1);
    const uint32_t last_block_row_tiles = get_arg_val<uint32_t>(2);
    const uint32_t core_w = get_arg_val<uint32_t>(3);
    const uint32_t has_tail = get_arg_val<uint32_t>(4);
    const uint32_t is_root = get_arg_val<uint32_t>(5);

    compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_stat_sq);

    // A mcast-box FILLER core: inside a reduction group's broadcast rectangle (a
    // physical shard grid is not always a rectangle) but owning no shard, so it
    // carries no work. It stays a program core only so the group's L1 map stays
    // uniform for the peer-addressed cb_rstd / cb_stat_gather.
    if (num_blocks == 0) {
        return;
    }

    // Hidden tiles handled by the bulk accumulation, and the stat-column layout.
    const uint32_t c_full = core_w - has_tail;
    const uint32_t bulk_cols = (c_full > 0) ? 1u : 0u;
    const uint32_t nc = bulk_cols + has_tail;  // stat columns per tile-row
    const uint32_t tail_col = bulk_cols;

    // ---- gamma is resident for the whole kernel: tilize it once (RM gamma) ----
    if constexpr (HAS_GAMMA && IS_RM_GAMMA) {
        // Converts the single stick to the row-0-valid tile form; TILE gamma
        // already arrives in that form from the reader.
        ckl::tilize<CB_W_TILES, cb_gamma_rm, cb_gamma_tiles>(1);
    }

    for (uint32_t b = 0; b < num_blocks; ++b) {
        const uint32_t rows_t = (b + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;

        // ---------------- tilize_in_block (RM input only) ----------------
        if constexpr (IS_RM_IN) {
            ckl::tilize<CB_W_TILES, cb_input_rm, cb_input_tiles>(rows_t);
        }

        // The block stays resident: waited here, waited again by scale_block,
        // popped exactly once (by scale_block) at the end.
        cb_wait_front(cb_input_tiles, rows_t * CB_W_TILES);

        // ---------------- sumsq_block + mask_tail_block ----------------
        cb_reserve_back(cb_stat_sq, rows_t * nc);
        if (c_full > 0) {
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(rows_t, c_full),
                ckl::BinaryFpu<
                    ckl::input(
                        cb_input_tiles,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::OperandKind::Block,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileOffset::Strided),
                    ckl::input(
                        cb_input_tiles,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::OperandKind::Block,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileOffset::Strided),
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::None,
                    ckl::Dst::D0,
                    ckl::DestAccumulation::PerRow>{
                    ckl::StridedTileRange{0, CB_W_TILES}, ckl::StridedTileRange{0, CB_W_TILES}},
                ckl::PackTile<ckl::output(
                    cb_stat_sq,
                    ckl::ReservePolicy::None,
                    ckl::PushPolicy::None,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::PackRelu::Disabled,
                    ckl::L1Accumulation::Disabled,
                    ckl::DestAccumulation::PerRow,
                    ckl::TileOffset::Strided)>{ckl::StridedTileRange{0, nc}});
        }
        if (has_tail) {
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(rows_t, 1),
                ckl::BinaryFpu<
                    ckl::input(
                        cb_input_tiles,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::OperandKind::Block,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileOffset::Strided),
                    ckl::input(cb_wmask, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Scalar),
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::Row,
                    ckl::Dst::D0,
                    ckl::DestAccumulation::Disabled>{ckl::StridedTileRange{core_w - 1, CB_W_TILES}},
                ckl::Square<>{},
                ckl::PackTile<ckl::output(
                    cb_stat_sq,
                    ckl::ReservePolicy::None,
                    ckl::PushPolicy::None,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::PackRelu::Disabled,
                    ckl::L1Accumulation::Disabled,
                    ckl::DestAccumulation::Disabled,
                    ckl::TileOffset::Strided)>{ckl::StridedTileRange{tail_col, nc}});
        }
        cb_push_back(cb_stat_sq, rows_t * nc);

        // ---------------- reduce_stat_block ----------------
        // Scaler is a plain 1.0: the hidden padding was already zeroed above, so
        // no partial scaler is needed here.
        ckl::reduce<
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            cb_stat_sq,
            cb_scaler,
            cb_stat_partial,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(ckl::ReduceInputBlockShape::of(rows_t, nc, 1));

        // ---------------- combine_stat_block (root side) ----------------
        // The writer has gathered the group's W_GROUP_SIZE partials into
        // cb_stat_gather at slot r * G + member. Summing across slots is one
        // more DEST accumulation over grid(rows_t, G); cb_zero_tile is the
        // identity B operand (BinaryFpu needs two CB inputs).
        if (is_root) {
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(rows_t, W_GROUP_SIZE),
                ckl::BinaryFpu<
                    ckl::input(
                        cb_stat_gather, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                    ckl::input(cb_zero_tile, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Scalar),
                    ckl::BinaryFpuOp::Add,
                    ckl::BroadcastDim::None,
                    ckl::Dst::D0,
                    ckl::DestAccumulation::PerRow>{},
                ckl::PackTile<ckl::output(
                    cb_stat_sum,
                    ckl::ReservePolicy::PerOuter,
                    ckl::PushPolicy::PerOuter,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::PackRelu::Disabled,
                    ckl::L1Accumulation::Disabled,
                    ckl::DestAccumulation::PerRow)>{});

            // finalize: rsqrt(sum * (1/W_true) + eps) -> the multicast source.
            // The 1/W uses the TRUE, unpadded W.
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t),
                ckl::CopyTile<ckl::input(cb_stat_sum)>{},
                ckl::MulUnary<>{INV_W_BITS},
                ckl::AddUnary<>{EPS_BITS},
                ckl::Rsqrt<>{},
                ckl::PackTile<ckl::output(cb_rstd_send)>{});
        }

        // ---------------- scale_block (+ gamma_block) ----------------
        // cb_rstd is filled by THIS core's writer (multicast landing buffer on
        // every group member, root included via INCLUDE_SRC loopback).
        if constexpr (HAS_GAMMA) {
            ckl::mul<
                ckl::input(cb_input_tiles, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                ckl::input(cb_rstd, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Col),
                ckl::output(cb_normed),
                ckl::BroadcastDim::Col>(ckl::EltwiseShape::grid(rows_t, CB_W_TILES));

            ckl::mul<
                ckl::input(cb_normed, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                ckl::input(cb_gamma_tiles, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Row),
                ckl::output(cb_output_tiles),
                ckl::BroadcastDim::Row>(ckl::EltwiseShape::grid(rows_t, CB_W_TILES));
        } else {
            ckl::mul<
                ckl::input(cb_input_tiles, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                ckl::input(cb_rstd, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Col),
                ckl::output(cb_output_tiles),
                ckl::BroadcastDim::Col>(ckl::EltwiseShape::grid(rows_t, CB_W_TILES));
        }

        // ---------------- untilize_out_block (RM output only) ----------------
        if constexpr (IS_RM_OUT) {
            ckl::untilize<CB_W_TILES, cb_output_tiles, cb_output_rm>(rows_t);
        }
    }
}
