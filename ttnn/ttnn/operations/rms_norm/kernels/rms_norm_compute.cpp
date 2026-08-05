// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for rms_norm (UNPACK / MATH / PACK).
//
//   out = x * rsqrt( (1/W) * sum_w x^2 + eps ) * gamma
//
// One loop nest covers ALL THREE regimes (op_design.md section 7).  Per row-block:
//
//   pass A   [RM] tilize            cb_input_sticks -> cb_input_tiles
//            square                 cb_input_tiles  -> cb_x_squared
//            accumulate_reduce_block cb_x_squared   -> cb_row_stat   (sum x^2)
//   finalize transform_in_place     cb_row_stat     -> cb_row_stat   (1/rms)
//   pass B   mul<Col>               cb_input_tiles, cb_row_stat -> NORM_OUT
//            mul<Row>               cb_normalized, cb_gamma_tiles -> cb_output_tiles
//            [RM] untilize          cb_output_tiles -> cb_output_sticks
//
// The regimes differ ONLY in whether cb_input_tiles / cb_gamma_tiles are held
// across both passes, and how wide they are (X_RESIDENT / X_HOLD_WT, from the
// descriptor -- deviation D14):
//   RESIDENT      X_RESIDENT, NUM_W_CHUNKS == 1.  The whole row is one chunk and
//                 is held, so x is read from DRAM once.
//   ROW_RESIDENT  X_RESIDENT, NUM_W_CHUNKS >  1.  One whole tile-row of x and the
//                 whole row of gamma are held while the DERIVED CBs stay chunked,
//                 so x is STILL read once: each helper call indexes the held CBs
//                 at a TILE OFFSET (TileOffset::Set, base = c * WT_CHUNK) and they
//                 are popped once per row-block rather than per chunk.
//   STREAM        !X_RESIDENT.  Each pass pops its chunk and the reader re-reads x
//                 (and gamma) for pass B -- the L1 fallback, ~2x the DRAM bytes.
//
// Every phase is a kernel_lib helper.  The only raw LLK is inside the
// transform_in_place lambda (x1/W, +eps, rsqrt) — that helper's documented
// calling convention, and the family explicitly routes multi-instruction
// finalizers like rsqrt-with-eps here rather than to a chain
// (streaming_reduce_helpers.hpp:75-78).  Refinement 4 adds ONE raw-LLK function
// inside that same lambda, `rsqrt_tile_col`; its justification is at its
// definition below (the rsqrt API exposes no VectorMode seam).
//
// Pass A's square and pass B's two multiplies are spelled as `eltwise_chain`
// rather than the `square` / `mul` convenience one-liners for one reason: the
// convenience wrappers take no element constructor arguments, and the
// ROW_RESIDENT regime needs to pass a runtime TILE OFFSET to the chain element.
// Each is exactly what the corresponding convenience call expands to.
//
// Explicit cb_pop_front calls on cb_input_tiles / cb_row_stat / cb_gamma_tiles /
// cb_scaler are the sanctioned pattern for operands whose lifetime spans more
// calls than any single PopPolicy can express (op_design.md section 6.1).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace ckl = compute_kernel_lib;

// Lamp L6b: the finalize's rsqrt, scoped to the tile's COLUMN faces.
//
// RAW-LLK SUBSTITUTION, with the reason.  `rsqrt_tile` hard-codes
// `VectorMode::RC` (api/compute/eltwise_unary/rsqrt.h:45) and exposes no
// VectorMode seam -- neither a template parameter nor a runtime argument -- so
// there is no helper that can express "rsqrt only the half of the tile that is
// read".  cb_row_stat is a REDUCE_ROW result whose only consumer is pass B's
// mul<BroadcastDim::Col>, i.e. column 0; in the 2x2-faces-of-16x16 tile layout
// that column lives in faces 0 and 2, which is exactly `VectorMode::C`
// (llk_math_eltwise_sfpu_common.h walks faces 0 and 2 for C).  Halving the
// datums the SFPU's 8-iteration rsqrt walks is the whole lever.
//
// This is the same macro `rsqrt_tile` itself expands to, with one argument
// changed, and it follows an established in-tree pattern for exactly this
// (sdpa/.../compute_common.hpp:251-256 `recip_tile_first_column`).  It stays
// inside the transform_in_place lambda, which is already this kernel's one
// sanctioned raw-LLK site.
#ifdef TRISC_MATH
template <bool legacy_compat = false, bool FAST_APPROX = false>
ALWI void rsqrt_tile_col(uint32_t idst) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_rsqrt,
        (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE, FAST_APPROX, legacy_compat),
        idst,
        VectorMode::C);
}
#endif

namespace {
constexpr uint32_t cb_input_sticks = 0;
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_x_squared = 2;
constexpr uint32_t cb_scaler = 3;
constexpr uint32_t cb_row_stat = 4;
constexpr uint32_t cb_gamma_sticks = 5;
constexpr uint32_t cb_gamma_tiles = 6;
constexpr uint32_t cb_normalized = 7;
constexpr uint32_t cb_output_tiles = 8;
constexpr uint32_t cb_output_sticks = 9;
// Cross-core width combine (op_design.md section 3.4) -- allocated only when the
// plan says COMBINE.
constexpr uint32_t cb_sum_handoff = 10;
constexpr uint32_t cb_partials_gathered = 11;
constexpr uint32_t cb_stat_handoff = 12;
constexpr uint32_t cb_row_final = 13;
}  // namespace

void kernel_main() {
    // ---- compile-time knobs (all from rms_norm_program_descriptor.py) -----
    constexpr uint32_t IS_TILE = get_compile_time_arg_val(0);
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(1);
    constexpr uint32_t NUM_W_CHUNKS = get_compile_time_arg_val(2);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(3);
    constexpr uint32_t PARTIAL_W = get_compile_time_arg_val(4);
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(5);
    constexpr uint32_t GAMMA_IS_RM = get_compile_time_arg_val(6);
    constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(7);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(8);
    constexpr uint32_t REDUCE_BULK = get_compile_time_arg_val(9);
    constexpr uint32_t REDUCE_ACC_VIA_ADD = get_compile_time_arg_val(10);
    constexpr uint32_t SCALER_TILES = get_compile_time_arg_val(11);
    // Refinement 2: the cross-core width combine.  COMBINE == 1 means this core
    // owns only a width SLICE of its rows, so pass A yields a PARTIAL sum(x^2)
    // that the group root sums, finalizes and multicasts back (the writer owns
    // the dataflow; see op_design.md section 3.4).
    constexpr uint32_t COMBINE = get_compile_time_arg_val(12);
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(13);
    // Refinement 4 / Lamp L6d (descriptor D12): cb_x_squared's width tiles per
    // tile-row, and hence the reduce's per-call reduce-dim width.  1 means pass A's
    // `square` folds the chunk's width tiles straight into DEST
    // (DestAccumulation::PerRow) rather than packing WT_CHUNK x^2 tiles out to L1
    // for the reduce to read back; WT_CHUNK is the unfolded (Phase-0) path.
    constexpr uint32_t X_SQUARED_WT = get_compile_time_arg_val(14);
    // Refinement 4 / Lamp L5 (descriptor D14).  X_RESIDENT is now an EXPLICIT flag
    // rather than `NUM_W_CHUNKS == 1`, which is what decouples "x is held across
    // both passes" from "the width is one chunk" and gives the op its third regime:
    //   RESIDENT      X_RESIDENT=1, NUM_W_CHUNKS==1   whole row in one chunk
    //   ROW_RESIDENT  X_RESIDENT=1, NUM_W_CHUNKS>1    x + gamma held for the whole
    //                                                 tile-row, only the DERIVED CBs
    //                                                 chunked -> x read ONCE
    //   STREAM        X_RESIDENT=0, NUM_W_CHUNKS>1    x re-read in pass B
    constexpr uint32_t X_RES = get_compile_time_arg_val(15);
    // Refinement 4 / Lamp L6b (descriptor D15): scope the finalize's rsqrt to the
    // tile's column faces (see rsqrt_tile_col above).  1 = scoped, 0 = whole tile.
    constexpr uint32_t RSQRT_COL = get_compile_time_arg_val(16);

    const uint32_t num_rows = get_arg_val<uint32_t>(0);  // tile-rows owned by this core
    // Only the core holding the row's LAST width tile applies the partial-W
    // scaler/mask; 1 on the whole-row schemes.
    const uint32_t owns_last_w = get_arg_val<uint32_t>(1);
    const uint32_t is_root = get_arg_val<uint32_t>(2);  // group root: sums + finalizes

    // An INACTIVE core (see the reader): no shard, no work, and its reader pushed
    // nothing -- return before any CB or LLK state is touched.
    if (num_rows == 0) {
        return;
    }

    constexpr bool RM = (IS_TILE == 0);
    constexpr bool HAS_G = (HAS_GAMMA != 0);
    constexpr bool G_RM = (GAMMA_IS_RM != 0);
    // X_RESIDENT == GAMMA_RESIDENT, from the descriptor's regime decision (D14).
    constexpr bool X_RESIDENT = (X_RES != 0);
    static_assert(NUM_W_CHUNKS > 1 || X_RESIDENT, "rms_norm: a one-chunk width is resident by definition");
    // Lamp L5's regime: resident x/gamma, chunked derived CBs.  The two held CBs
    // then span the WHOLE tile-row while every helper call still works on one
    // WT_CHUNK, so each call indexes them at a TILE OFFSET (TileOffset::Set) and
    // neither is popped until the row-block is done.
    constexpr bool ROW_RESIDENT = X_RESIDENT && (NUM_W_CHUNKS > 1);
    static_assert(!ROW_RESIDENT || BLOCK_ROWS == 1, "rms_norm: ROW_RESIDENT holds ONE tile-row of x");
    // Width tiles the HELD CBs (cb_input_tiles, cb_gamma_tiles) span.  Equals
    // WT_CHUNK in both Phase-0 regimes, so this is byte-identical off the L5 path.
    constexpr uint32_t X_HOLD_WT = X_RESIDENT ? (WT_CHUNK * NUM_W_CHUNKS) : WT_CHUNK;
    constexpr auto XOFF = ROW_RESIDENT ? ckl::TileOffset::Set : ckl::TileOffset::Unset;

    // srcA at boot is whichever CB the first helper unpacks from.
    constexpr uint32_t CB_A = RM ? cb_input_sticks : cb_input_tiles;
    compute_kernel_hw_startup(CB_A, cb_scaler, cb_output_tiles);

    // ---- policy / shape knobs --------------------------------------------
    constexpr auto REDUCE_POLICY =
        (REDUCE_BULK != 0) ? ckl::ReduceInputPolicy::BulkWaitBulkPop : ckl::ReduceInputPolicy::WaitAndPopPerTile;

    // Reduce datapath, chosen host-side from WT_CHUNK (the reduce-dim tiles per
    // reduce() call) -- see REDUCE_ACC_VIA_ADD_MIN_WT in the descriptor.
    //
    //   AccumulateViaAdd sums the width tiles ELEMENTWISE into DST with pairwise
    //   add_tiles and finishes the within-tile 32-column sum on the SFPU (fp32
    //   LREGs).  ReduceTile (the FPU matmul-with-ones) instead accumulates all
    //   WT_CHUNK*32 all-positive addends of sum(x^2) into a single DEST word --
    //   which at fp32_dest_acc_en=False is 16-bit, and is exactly the wide-W
    //   error Refinement 1 diagnosed.  AccumulateViaAdd cuts the DEST-resident
    //   accumulation depth by 32x (WT_CHUNK/2 pairwise adds instead of
    //   WT_CHUNK*32 serial ones) and is also the faster path once the reduce dim
    //   spans >= 4 tiles (examples/reduce_block/report_reduced_sweep.md).
    constexpr bool ACC_VIA_ADD = (REDUCE_ACC_VIA_ADD != 0);
    constexpr auto REDUCE_ALGO = ACC_VIA_ADD ? ckl::ReduceAlgorithm::AccumulateViaAdd : ckl::ReduceAlgorithm::Auto;
    // AccumulateViaAdd's cross-chunk Accumulate indexes a resident block, so it
    // is BulkWaitBulkPop-only (reduce_helpers_compute.inl static_assert). The
    // descriptor already couples the two knobs; assert it so a future flip of
    // REDUCE_BULK fails here instead of deep inside the library.
    static_assert(
        !ACC_VIA_ADD || REDUCE_BULK != 0,
        "rms_norm: ReduceAlgorithm::AccumulateViaAdd + Accumulate requires BulkWaitBulkPop (REDUCE_BULK == 1)");

    // Non-tile-aligned W: the two datapaths take DIFFERENT partial mechanisms.
    //   ReduceTile       : reader emitted [full scaler, partial scaler]; route the
    //                      partial one to the last width tile (pad lanes * 0).
    //   AccumulateViaAdd : reader emitted a single 0/1 MASK tile at index 0; the
    //                      last width tile folds in through a masked accumulating
    //                      broadcast-mul, PARTIAL_W valid lanes.
    // Both zero the pad lanes by multiplying them with an exact 0, so the reader's
    // pad-lane invariant (no inf/NaN in padding) is what it always was.
    //
    // Under a cross-core width split only ONE core in the group holds the row's
    // last width tile, so the choice is per-core (runtime `owns_last_w`).  A
    // non-owning core takes none(), which is exactly right on BOTH datapaths:
    // ReduceTile then uses cb_scaler tile 0 (the FULL 1.0 scaler) everywhere, and
    // AccumulateViaAdd ignores cb_scaler entirely when there is no partial.
    const auto PARTIAL_SCALER = (PARTIAL_W == 0 || owns_last_w == 0)
                                    ? ckl::ReducePartialScaler::none()
                                    : (ACC_VIA_ADD ? ckl::ReducePartialScaler::partial_mask(PARTIAL_W, /*mask_idx=*/0)
                                                   : ckl::ReducePartialScaler::last_tile_at(1));
    // RESIDENT holds x across both passes -> pass A must not pop it.
    constexpr auto PASS_A_POP = X_RESIDENT ? ckl::PopPolicy::None : ckl::PopPolicy::AtEnd;

    // Lamp L6d / D12: fold pass A's square straight into DEST.  The chunk's width
    // tiles are multiplied and ACCUMULATED in one DEST slot, so cb_x_squared takes
    // one tile per tile-row (X_SQUARED_WT == 1) and the reduce's per-call width is 1
    // -- deleting WT_CHUNK-1 packs and the matching unpacks per tile-row.  The
    // cross-chunk carry still runs through the fp32 cb_row_stat, so the accumulation
    // DEST sees is bounded by WT_CHUNK, which is exactly what the descriptor's
    // DEST_ACC_SQUARE_MAX_WT ceiling bounds.
    constexpr bool SQ_FOLD = (X_SQUARED_WT == 1) && (WT_CHUNK > 1);
    static_assert(
        X_SQUARED_WT == 1 || X_SQUARED_WT == WT_CHUNK,
        "rms_norm: X_SQUARED_WT must be 1 (the DEST fold) or WT_CHUNK (the packed path)");
    static_assert(
        !SQ_FOLD || PARTIAL_W == 0,
        "rms_norm: the DEST fold folds the last width tile's pad lanes in BEFORE the "
        "reduce, so the reduce's partial scaler / mask can no longer reach them");
    // The fold's pack is per-OUTER (one tile per tile-row of the grid), which is the
    // policy pair DestAccumulation::PerRow requires.
    constexpr auto SQ_OUT_FOLDED = ckl::output(
        cb_x_squared,
        ckl::ReservePolicy::PerOuter,
        ckl::PushPolicy::PerOuter,
        ckl::DataFormatReconfig::Enabled,
        ckl::PackRelu::Disabled,
        ckl::L1Accumulation::Disabled,
        ckl::DestAccumulation::PerRow);
    constexpr auto SQ_OUT = SQ_FOLD ? SQ_OUT_FOLDED : ckl::output(cb_x_squared);

    // ---- the operand specs, ONE definition each ---------------------------
    // Every one carries XOFF, so the L5 regime differs from Phase 0 only in the
    // (compile-time-elided) `+ base` on the tile index -- there is no second
    // code path.  base is 0 whenever XOFF is Unset, and `tile_base_value<Unset>`
    // folds the whole term away.
    constexpr auto X_IN_A = ckl::input(
        cb_input_tiles,
        ckl::WaitPolicy::Upfront,
        PASS_A_POP,
        ckl::OperandKind::Block,
        ckl::DataFormatReconfig::Enabled,
        XOFF);
    // Pass B's x: held CBs are popped ONCE per row-block below (an explicit pop is
    // the sanctioned pattern for a lifetime no single PopPolicy can express), so
    // that a chunk's `AtEnd` cannot pop the base tiles the next chunk still needs.
    constexpr auto PASS_B_X_POP = ROW_RESIDENT ? ckl::PopPolicy::None : ckl::PopPolicy::AtEnd;
    constexpr auto X_IN_B = ckl::input(
        cb_input_tiles,
        ckl::WaitPolicy::Upfront,
        PASS_B_X_POP,
        ckl::OperandKind::Block,
        ckl::DataFormatReconfig::Enabled,
        XOFF);
    constexpr auto G_IN = ckl::input(
        cb_gamma_tiles,
        ckl::WaitPolicy::Upfront,
        ckl::PopPolicy::None,
        ckl::OperandKind::Row,
        ckl::DataFormatReconfig::Enabled,
        XOFF);
    constexpr uint32_t NORM_OUT = HAS_G ? cb_normalized : cb_output_tiles;
    constexpr bool CROSS_CORE = (COMBINE != 0);
    // Pass B's Col operand: the multicast landing CB when the stat was combined
    // across cores, the local accumulator otherwise.
    constexpr uint32_t CB_STAT_B = CROSS_CORE ? cb_row_final : cb_row_stat;

    // 1/rms = rsqrt(sum/W + eps).  ONE definition, used by the local path and by
    // the root's post-combine finalize -- INV_W is the LOGICAL width either way.
    auto finalize = [](uint32_t dst) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, INV_W_BITS);  // x (1/W): the LOGICAL width (R1)
        add_unary_tile(dst, EPS_BITS);
        rsqrt_tile_init();
        // The scaled/eps-shifted value is now in every lane; only the lanes pass B
        // reads back need the rsqrt (L6b).  The init is vector-mode-agnostic.
        if constexpr (RSQRT_COL != 0) {
            MATH((rsqrt_tile_col(dst)));
        } else {
            rsqrt_tile(dst);
        }
    };

    // ---- gamma: resident for the whole core's assignment (RESIDENT) -------
    // ROW_RESIDENT holds the whole tile-row of gamma too, so it tilizes every
    // chunk the reader staged; NUM_W_CHUNKS == 1 makes this the Phase-0 single call.
    if constexpr (HAS_G && X_RESIDENT && G_RM) {
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            ckl::tilize<WT_CHUNK, cb_gamma_sticks, cb_gamma_tiles>(1);
        }
    }

    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t r0 = blk * BLOCK_ROWS;
        const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;

        // ================= pass A: sum(x^2) over the whole width ===========
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            // Tile offset of this chunk inside the HELD CBs.  0 (and elided) unless
            // ROW_RESIDENT, where cb_input_tiles / cb_gamma_tiles span the whole row.
            const uint32_t hold_base = ROW_RESIDENT ? (c * WT_CHUNK) : 0;
            if constexpr (RM) {
                ckl::tilize<WT_CHUNK, cb_input_sticks, cb_input_tiles>(rows);
            }
            // x^2, either packed to cb_x_squared per width tile or folded into DEST
            // (D12).  `square` cannot carry the tile base, so the chain is spelled
            // out; it is exactly what square<> expands to.
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(rows, WT_CHUNK),
                ckl::BinaryFpu<
                    X_IN_A,
                    X_IN_A,
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::None,
                    ckl::Dst::D0,
                    SQ_OUT.dest_accumulation>{hold_base, hold_base},
                ckl::PackTile<SQ_OUT>{});

            ckl::accumulate_reduce_block<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_x_squared,
                cb_scaler,
                cb_row_stat,
                REDUCE_POLICY,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                REDUCE_ALGO>(ckl::ReduceInputBlockShape::of(rows, X_SQUARED_WT), c, NUM_W_CHUNKS, PARTIAL_SCALER);
        }

        // ================= finalize: 1/rms = rsqrt(sum/W + eps) ============
        // Pops before reserving, so the `rows`-page accumulator CB suffices.
        if constexpr (!CROSS_CORE) {
            for (uint32_t i = 0; i < rows; ++i) {
                ckl::transform_in_place(cb_row_stat, finalize);
            }
        } else {
            // Hand this core's RAW partial to the writer, which ships it to the
            // group root.  cb_sum_handoff is DEDICATED to that handoff so the
            // writer never becomes a second consumer of cb_row_stat.
            ckl::copy<ckl::input(cb_row_stat), ckl::output(cb_sum_handoff)>(ckl::EltwiseShape::tiles(rows));
            if (is_root != 0) {
                // Sum the group's GROUP_SIZE partials ELEMENTWISE: each is a
                // column-shaped REDUCE_ROW result, so the row total is their
                // elementwise sum regardless of which lanes the datapath filled.
                // Slot 0 seeds the accumulator (this core's own partial), then the
                // remaining slots fold in-place -- per-tile streaming throughout,
                // so no slot has to be contiguous with any other.
                ckl::copy<ckl::input(cb_partials_gathered), ckl::output(cb_row_stat)>(ckl::EltwiseShape::tiles(rows));
                for (uint32_t g = 1; g < GROUP_SIZE; ++g) {
                    ckl::add<ckl::input(cb_row_stat), ckl::input(cb_partials_gathered), ckl::output(cb_row_stat)>(
                        ckl::EltwiseShape::tiles(rows));
                }
                for (uint32_t i = 0; i < rows; ++i) {
                    ckl::transform_in_place(cb_row_stat, finalize);
                }
                ckl::copy<ckl::input(cb_row_stat), ckl::output(cb_stat_handoff)>(ckl::EltwiseShape::tiles(rows));
            }
        }

        // ================= pass B: scale ===================================
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            const uint32_t hold_base = ROW_RESIDENT ? (c * WT_CHUNK) : 0;
            // ROW_RESIDENT never re-stages either held operand: pass A already put
            // the whole tile-row of x tiles (and of gamma) in L1.  This is the
            // pass-B re-read that Lamp L5 exists to delete.
            if constexpr (RM && !X_RESIDENT) {
                ckl::tilize<WT_CHUNK, cb_input_sticks, cb_input_tiles>(rows);
            }
            if constexpr (HAS_G && !X_RESIDENT && G_RM) {
                ckl::tilize<WT_CHUNK, cb_gamma_sticks, cb_gamma_tiles>(1);
            }

            // x * (1/rms). The stat is a REDUCE_ROW result: column-shaped, so it
            // broadcasts back ACROSS columns (BroadcastDim::Col) and must be
            // operand B. OperandKind::Col indexes it by row only, and it is not
            // popped -- every width chunk of this block re-reads it.
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(rows, WT_CHUNK),
                ckl::BinaryFpu<
                    X_IN_B,
                    ckl::input(CB_STAT_B, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Col),
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::Col>{hold_base},
                ckl::PackTile<ckl::output(NORM_OUT)>{});

            if constexpr (HAS_G) {
                // gamma is row-shaped (1 x W, valid in row 0) -> broadcasts DOWN
                // rows (BroadcastDim::Row), indexed by column (OperandKind::Row).
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(rows, WT_CHUNK),
                    ckl::BinaryFpu<
                        ckl::input(
                            cb_normalized, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                        G_IN,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Row>{0u, hold_base},
                    ckl::PackTile<ckl::output(cb_output_tiles)>{});
            }

            if constexpr (RM) {
                ckl::untilize<WT_CHUNK, cb_output_tiles, cb_output_sticks>(rows);
            }

            if constexpr (HAS_G && !X_RESIDENT) {
                cb_pop_front(cb_gamma_tiles, WT_CHUNK);
            }
        }

        // The held CBs' lifetime is the whole row-block, which no PopPolicy on a
        // per-chunk call can express (an `AtEnd` would drop the base tiles the next
        // chunk still indexes), so ROW_RESIDENT pops x here -- the same sanctioned
        // pattern as cb_row_stat / cb_gamma_tiles / cb_scaler.
        if constexpr (ROW_RESIDENT) {
            cb_pop_front(cb_input_tiles, rows * X_HOLD_WT);
        }
        cb_pop_front(CB_STAT_B, rows);
    }

    // SCALER_TILES is the descriptor's single source of truth for how many tiles
    // the reader pushed into cb_scaler (datapath- and PARTIAL_W-dependent).
    cb_pop_front(cb_scaler, SCALER_TILES);
    if constexpr (HAS_G && X_RESIDENT) {
        cb_pop_front(cb_gamma_tiles, X_HOLD_WT);
    }
}
