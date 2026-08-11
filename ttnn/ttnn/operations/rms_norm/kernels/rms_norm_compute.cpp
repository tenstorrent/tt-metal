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
// HIDDEN-AXIS CHUNKING (op_design.md regime R3, Refinement 2b). CB_CHUNK_TILES
// (WC) is the block extent along `hidden` of every buffer that only STREAMS over
// it — cb_gamma_tiles, cb_normed, cb_output_*, cb_input_rm. At WC == CB_W_TILES
// (NUM_CHUNKS == 1, the default for every interleaved geometry) the loops below
// run exactly once and this file is the unchunked schedule. When a resident
// shard PINS both G and C — a HEIGHT shard hands one core the tensor's whole
// hidden slice — those buffers are what overruns L1, so the schedule walks the
// hidden axis in NUM_CHUNKS = ceil(C / WC) chunks instead:
//   * sumsq_block runs once per chunk and packs that chunk's partial Sum x^2
//     into its OWN stat column, so reduce_stat_block (which already sums the
//     `nc` columns of a tile-row) folds them together for free — no L1
//     read-modify-write accumulator, and no change to the combine.
//   * the apply pass runs once per chunk against the chunk's gamma slice.
// The input block stays whole-resident either way (that is what keeps the input
// to one crossing), so chunking never re-reads it.
//
// BLOCK LAYOUT of cb_input_tiles. A TILE input — interleaved (reader-written) or
// a pinned shard — is row-major at row stride CB_W_TILES. A ROW_MAJOR input is
// staged by `tilize<CB_CHUNK_TILES>` per chunk, which emits chunks
// back-to-back, so it is CHUNK-major: tile (r, g) sits at
// (g / WC) * rows * WC + r * WC + g % WC. `in_ref()` is the one place that knows
// which; every phase addresses the block through it. The two coincide at
// NUM_CHUNKS == 1.
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
    // See the reader: the ROW_MAJOR gamma stick is staged in cb_input_rm when the
    // two CBs share a page format (disjoint lifetimes, same producer + consumer).
    constexpr bool ALIAS_GAMMA_RM = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t cb_gamma_stage = ALIAS_GAMMA_RM ? cb_input_rm : cb_gamma_rm;
    // Block extent along `hidden` of the buffers that only STREAM over it. Equal
    // to CB_W_TILES (one chunk) unless a resident shard pinned C too wide to fit.
    constexpr uint32_t CB_CHUNK_TILES = get_compile_time_arg_val(9);
    constexpr uint32_t NUM_CHUNKS = (CB_W_TILES + CB_CHUNK_TILES - 1) / CB_CHUNK_TILES;
    constexpr bool CHUNKED = NUM_CHUNKS > 1;
    // The only chunked configuration with a TILE output is a PINNED output shard
    // (the host gates chunking on a resident, uniform-width shard), whose layout
    // is the shard's own row-major-C one. The apply therefore packs into it at a
    // strided offset under a caller-managed reserve/push, instead of streaming
    // chunk-major pages the writer would then have to un-permute.
    constexpr bool OUT_STRIDED = CHUNKED && !IS_RM_OUT;

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
    // ONE stat column per bulk chunk (plus the tail's): reduce_stat_block already
    // sums a tile-row's `nc` columns, so the per-chunk partials fold there instead
    // of through an L1 read-modify-write accumulator. Unchunked this is the
    // Phase 0 expression (bulk_cols = c_full > 0).
    const uint32_t c_full = core_w - has_tail;
    const uint32_t bulk_cols = (c_full + CB_CHUNK_TILES - 1) / CB_CHUNK_TILES;
    const uint32_t nc = bulk_cols + has_tail;  // stat columns per tile-row
    const uint32_t tail_col = bulk_cols;

    // Tile (r, g) of the resident input block, as (base, row_stride) — the ONE
    // place that knows whether the block is row-major (TILE input, stride
    // CB_W_TILES) or chunk-major (ROW_MAJOR input, staged chunk by chunk by
    // tilize<CB_CHUNK_TILES>). Identical at NUM_CHUNKS == 1.
    auto in_ref = [](uint32_t g, uint32_t rows_t) -> ckl::StridedTileRange {
        if constexpr (IS_RM_IN) {
            const uint32_t k = g / CB_CHUNK_TILES;
            return ckl::StridedTileRange{k * rows_t * CB_CHUNK_TILES + (g - k * CB_CHUNK_TILES), CB_CHUNK_TILES};
        } else {
            return ckl::StridedTileRange{g, CB_W_TILES};
        }
    };

    // ---- gamma (RM): tilize the stick into the row-0-valid tile form ----
    // Unchunked, gamma is read ONCE for the whole kernel and never popped. Chunked,
    // one chunk's slice is resident at a time, so the reader re-feeds it per chunk
    // inside the block loop and `apply_chunk` pops it there.
    if constexpr (HAS_GAMMA && IS_RM_GAMMA && !CHUNKED) {
        ckl::tilize<CB_W_TILES, cb_gamma_stage, cb_gamma_tiles>(1);
    }

    for (uint32_t b = 0; b < num_blocks; ++b) {
        const uint32_t rows_t = (b + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;

        // ---------------- tilize_in_block (RM input only) ----------------
        // One tilize per chunk: cb_input_rm holds one chunk's width, and the
        // chunks land back-to-back in cb_input_tiles (the chunk-major layout
        // `in_ref` decodes). NUM_CHUNKS == 1 is the single unchunked call.
        if constexpr (IS_RM_IN) {
            for (uint32_t k = 0; k < NUM_CHUNKS; ++k) {
                ckl::tilize<CB_CHUNK_TILES, cb_input_rm, cb_input_tiles>(rows_t);
            }
        }

        // The whole block stays resident: waited here, read again by every apply
        // chunk, popped exactly once at the end of the block.
        const uint32_t in_block_pages = rows_t * (IS_RM_IN ? (NUM_CHUNKS * CB_CHUNK_TILES) : CB_W_TILES);
        cb_wait_front(cb_input_tiles, in_block_pages);

        // ---------------- sumsq_block + mask_tail_block ----------------
        cb_reserve_back(cb_stat_sq, rows_t * nc);
        for (uint32_t k = 0; k < bulk_cols; ++k) {
            const uint32_t chunk_base = k * CB_CHUNK_TILES;
            const uint32_t cols = (c_full - chunk_base < CB_CHUNK_TILES) ? (c_full - chunk_base) : CB_CHUNK_TILES;
            const ckl::StridedTileRange src = in_ref(chunk_base, rows_t);
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(rows_t, cols),
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
                    ckl::DestAccumulation::PerRow>{src, src},
                ckl::PackTile<ckl::output(
                    cb_stat_sq,
                    ckl::ReservePolicy::None,
                    ckl::PushPolicy::None,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::PackRelu::Disabled,
                    ckl::L1Accumulation::Disabled,
                    ckl::DestAccumulation::PerRow,
                    ckl::TileOffset::Strided)>{ckl::StridedTileRange{k, nc}});
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
                    ckl::DestAccumulation::Disabled>{in_ref(core_w - 1, rows_t)},
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

        // ---------------- scale_block (+ gamma_block), once per chunk ---------
        // cb_rstd is filled by THIS core's writer (multicast landing buffer on
        // every group member, root included via INCLUDE_SRC loopback). It and the
        // resident input block serve EVERY chunk, so both are caller-managed here:
        // waited once, popped once, after the last chunk.
        cb_wait_front(cb_rstd, rows_t);
        if constexpr (OUT_STRIDED) {
            cb_reserve_back(cb_output_tiles, rows_t * CB_W_TILES);
        }
        for (uint32_t k = 0; k < NUM_CHUNKS; ++k) {
            const uint32_t chunk_base = k * CB_CHUNK_TILES;
            // A strided pack into the row-major output block must stop at column
            // CB_W_TILES or it would spill into the next tile-row; a chunk-major
            // streaming output carries the pad columns of the last chunk, exactly
            // as the unchunked apply carries `CB_W_TILES - core_w`.
            const uint32_t cols =
                OUT_STRIDED ? ((CB_W_TILES - chunk_base < CB_CHUNK_TILES) ? (CB_W_TILES - chunk_base) : CB_CHUNK_TILES)
                            : CB_CHUNK_TILES;
            const ckl::StridedTileRange src = in_ref(chunk_base, rows_t);
            constexpr auto in_spec = ckl::input(
                cb_input_tiles,
                ckl::WaitPolicy::None,
                ckl::PopPolicy::None,
                ckl::OperandKind::Block,
                ckl::DataFormatReconfig::Enabled,
                ckl::TileOffset::Strided);
            constexpr auto rstd_spec =
                ckl::input(cb_rstd, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Col);
            constexpr auto gamma_spec =
                ckl::input(cb_gamma_tiles, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Row);
            constexpr auto out_strided = ckl::output(
                cb_output_tiles,
                ckl::ReservePolicy::None,
                ckl::PushPolicy::None,
                ckl::DataFormatReconfig::Enabled,
                ckl::PackRelu::Disabled,
                ckl::L1Accumulation::Disabled,
                ckl::DestAccumulation::Disabled,
                ckl::TileOffset::Strided);

            if constexpr (HAS_GAMMA) {
                if constexpr (IS_RM_GAMMA && CHUNKED) {
                    // This chunk's gamma slice, re-fed by the reader per chunk.
                    ckl::tilize<CB_CHUNK_TILES, cb_gamma_stage, cb_gamma_tiles>(1);
                }
                cb_wait_front(cb_gamma_tiles, CB_CHUNK_TILES);
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(rows_t, cols),
                    ckl::BinaryFpu<in_spec, rstd_spec, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col>{src},
                    ckl::PackTile<ckl::output(cb_normed)>{});

                if constexpr (OUT_STRIDED) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::grid(rows_t, cols),
                        ckl::BinaryFpu<
                            ckl::input(
                                cb_normed, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                            gamma_spec,
                            ckl::BinaryFpuOp::Mul,
                            ckl::BroadcastDim::Row>{},
                        ckl::PackTile<out_strided>{ckl::StridedTileRange{chunk_base, CB_W_TILES}});
                } else {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::grid(rows_t, cols),
                        ckl::BinaryFpu<
                            ckl::input(
                                cb_normed, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                            gamma_spec,
                            ckl::BinaryFpuOp::Mul,
                            ckl::BroadcastDim::Row>{},
                        ckl::PackTile<ckl::output(cb_output_tiles)>{});
                }
                if constexpr (CHUNKED) {
                    cb_pop_front(cb_gamma_tiles, CB_CHUNK_TILES);
                }
            } else if constexpr (OUT_STRIDED) {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(rows_t, cols),
                    ckl::BinaryFpu<in_spec, rstd_spec, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col>{src},
                    ckl::PackTile<out_strided>{ckl::StridedTileRange{chunk_base, CB_W_TILES}});
            } else {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(rows_t, cols),
                    ckl::BinaryFpu<in_spec, rstd_spec, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col>{src},
                    ckl::PackTile<ckl::output(cb_output_tiles)>{});
            }

            // ---------------- untilize_out_block (RM output only) ------------
            if constexpr (IS_RM_OUT) {
                ckl::untilize<CB_CHUNK_TILES, cb_output_tiles, cb_output_rm>(rows_t);
            }
        }
        if constexpr (OUT_STRIDED) {
            cb_push_back(cb_output_tiles, rows_t * CB_W_TILES);
        }
        cb_pop_front(cb_rstd, rows_t);
        cb_pop_front(cb_input_tiles, in_block_pages);
    }
}
