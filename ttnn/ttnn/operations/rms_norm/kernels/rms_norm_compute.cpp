// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm UNIFIED compute kernel — the single compute for all paths
// (TILE Regime A, TILE Regime B mcast, ROW_MAJOR tilize-wrapped, and ROW_MAJOR
// routed through the mcast all-gather). Refinement 4 merged the former
// rms_norm_compute.cpp (TILE) and rms_norm_compute_rm.cpp (RM) into this one
// kernel: the TILE-vs-RM difference is purely a data-access boundary (sticks vs
// tiles), expressed here as a constexpr tilize prologue + untilize epilogue. The
// core square -> reduce -> rsqrt -> normalize arithmetic is shared verbatim, and
// for the bf16/TILE Phase-0 corner this kernel is numerically byte-identical to
// the pre-R4 compute.
//
// Per resident shard (one tile-row, Wt tiles wide; RM: Wt = num_chunks*reduce_block,
// the padded shard width so each chunk is a full reduce_block):
//   [RM]     tilize cb_rm_in chunks -> cb_input_resident (held resident)
//   PASS 1   sum of squares over the WHOLE resident shard in one shot
//            (single square -> single reduce<SUM,REDUCE_ROW>) -> cb_partial_sumsq
//            (Refinement 5: the resident shard is already L1-bounded by the host
//             A/B W-split, so the former DEST-level reduce_block chunking +
//             reduce-Accumulate loop was redundant and has been removed.)
//   [B]      combine the K gathered partials in ONE reduce<SUM,REDUCE_ROW> over the
//            K-tile block (Refinement 9 Part B; replaced the copy + K-1 eltwise adds) -> cb_partial_sumsq
//   FINALIZE rsqrt(sum * inv_W + eps) -> cb_recip_rms
//   PASS 2   normalize x * recip (Col bcast) [* gamma (Row bcast)] -> cb_pass2_out
//   [RM]     untilize cb_pass2_out chunk -> cb_rm_out (writer drains sticks)
//
// Gamma is unified to the resident model regardless of layout: TILE gamma is fed
// as resident column tiles by the reader; ROW_MAJOR gamma is staged as sticks in
// cb_gamma_rm and tilized ONCE into cb_gamma here. PASS-2 always reads cb_gamma at
// an explicit per-chunk TileOffset — identical for every layout/regime.
//
// Helpers own all CB and DST ops; manual CB ops only appear between helper calls.
// Caller (this kernel) owns the single compute_kernel_hw_startup at boot.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    // ---- compile-time args ----
    constexpr uint32_t cb_input_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_partials_gathered = get_compile_time_arg_val(3);
    constexpr uint32_t cb_pass2_out = get_compile_time_arg_val(4);  // TILE: cb_output; RM: cb_out_tiled
    constexpr uint32_t cb_squared = get_compile_time_arg_val(5);
    constexpr uint32_t cb_partial_sumsq = get_compile_time_arg_val(6);
    constexpr uint32_t cb_recip_rms = get_compile_time_arg_val(7);
    constexpr uint32_t cb_normalized = get_compile_time_arg_val(8);
    constexpr uint32_t Wt = get_compile_time_arg_val(9);  // tiles per shard along W (RM: padded)
    constexpr uint32_t reduce_block = get_compile_time_arg_val(10);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(11);
    constexpr uint32_t inv_W_bits = get_compile_time_arg_val(12);    // fp32 bits of 1/W
    constexpr uint32_t eps_bits = get_compile_time_arg_val(13);      // fp32 bits of epsilon
    constexpr uint32_t num_partials = get_compile_time_arg_val(14);  // K (1 in Regime A)
    // Regime B only: dedicated single-push CB that hands the fully-accumulated local
    // Sum(x^2) to the reader's mcast all-gather (Refinement-1 correctness fix). Unused
    // when num_partials == 1.
    constexpr uint32_t cb_local_sumsq = get_compile_time_arg_val(15);
    // gamma supplied ROW_MAJOR (1,1,1,W): reader fills cb_gamma_rm with gamma sticks
    // and this kernel tilizes them ONCE into cb_gamma. For TILE gamma (gamma_is_rm==0)
    // the reader fills cb_gamma directly and this block is elided.
    constexpr uint32_t gamma_is_rm = get_compile_time_arg_val(16);
    constexpr uint32_t cb_gamma_rm = get_compile_time_arg_val(17);
    // Refinement 4: input supplied ROW_MAJOR. Reader fills cb_rm_in with input sticks;
    // this kernel tilizes them per chunk into cb_input_resident and untilizes the
    // per-chunk pass-2 result from cb_pass2_out into cb_rm_out. For TILE (==0) the
    // reader fills cb_input_resident directly and the writer drains cb_pass2_out.
    constexpr uint32_t layout_is_rm = get_compile_time_arg_val(18);
    constexpr uint32_t cb_rm_in = get_compile_time_arg_val(19);
    constexpr uint32_t cb_rm_out = get_compile_time_arg_val(20);
    // Refinement 7: row-blocking. Process BLOCK_HEIGHT tile-rows per work-unit so the
    // PASS-1 reduce / FINALIZE per-row helper init amortizes over a taller block. Only
    // set > 1 by the host for TILE Regime A no-gamma (layout_is_rm==0, has_gamma==0,
    // num_partials==1) with a grid already saturated (Ht_total > total_cores) — every
    // other path keeps BLOCK_HEIGHT==1 and is byte-identical to the pre-R7 kernel.
    constexpr uint32_t BLOCK_HEIGHT = get_compile_time_arg_val(21);

    // ---- runtime args ----
    const uint32_t num_rows = get_arg_val<uint32_t>(0);  // RM: number of 32-stick blocks

    using ckl::BinaryDataFormatReconfig;
    using ckl::BinaryFpu;
    using ckl::BinaryFpuOp;
    using ckl::BroadcastDim;
    using ckl::CopyTile;
    using ckl::Dst;
    using ckl::EltwiseShape;
    using ckl::InputLifecycle;
    using ckl::OperandKind;
    using ckl::OutputLifecycle;
    using ckl::PackTile;
    using ckl::PackTileReconfig;
    using ckl::ReduceInputBlockShape;
    using ckl::ReduceInputPolicy;
    using ckl::TileOffset;

    if constexpr (layout_is_rm) {
        compute_kernel_hw_startup(cb_rm_in, cb_input_resident);
    } else {
        compute_kernel_hw_startup(cb_input_resident, cb_scaler, cb_pass2_out);
    }

    constexpr uint32_t num_chunks = (Wt + reduce_block - 1) / reduce_block;

    // ROW_MAJOR gamma: tilize the staged gamma sticks into cb_gamma once, resident
    // for all rows. cb_gamma is sized to num_chunks*reduce_block tiles (the descriptor
    // pads it); PASS-2 reads offsets [0, Wt).
    if constexpr (has_gamma && gamma_is_rm) {
        for (uint32_t c = 0; c < num_chunks; ++c) {
            ckl::tilize<reduce_block, cb_gamma_rm, cb_gamma>(1);
        }
    }

    // =====================================================================
    // Refinement 7: row-blocked path (BLOCK_HEIGHT > 1). TILE Regime A no-gamma
    // only — process BLOCK_HEIGHT rows per block so PASS-1 reduce + FINALIZE init
    // amortize over the taller block. The host guarantees num_rows % BLOCK_HEIGHT
    // == 0 (it only enables bh>1 when every core owns the same, bh-divisible row
    // count), so there is no remainder to handle.
    // =====================================================================
    if constexpr (BLOCK_HEIGHT > 1) {
        const uint32_t num_blocks = num_rows / BLOCK_HEIGHT;
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            // ---------- PASS 1: square the WHOLE BLOCK_HEIGHT*Wt resident block ----------
            // x*x over the bh-row block (no pop; reused in PASS-2). Block walk 0..bh*Wt.
            ckl::eltwise_chain(
                EltwiseShape::tiles(BLOCK_HEIGHT * Wt),
                BinaryFpu<
                    cb_input_resident,
                    cb_input_resident,
                    BinaryFpuOp::Mul,
                    BroadcastDim::None,
                    InputLifecycle::HeldBulk,
                    InputLifecycle::HeldBulk,
                    BinaryDataFormatReconfig::Input,
                    Dst::D0,
                    OperandKind::Block,
                    OperandKind::Block,
                    TileOffset::Set,
                    TileOffset::Set>{0, 0},
                PackTile<cb_squared, OutputLifecycle::Streaming>{});

            // reduce each of BLOCK_HEIGHT rows over Wt -> BLOCK_HEIGHT partial Σx² tiles
            // (REDUCE_ROW emits `rows` output tiles; BulkWaitBulkPop waits/pops Wt/row).
            ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, ReduceInputPolicy::BulkWaitBulkPop>(
                cb_squared,
                cb_scaler,
                cb_partial_sumsq,
                ReduceInputBlockShape::of(BLOCK_HEIGHT, Wt),
                ckl::ReduceInputMemoryLayout::contiguous());

            // ---------- FINALIZE: rsqrt(sum * inv_W + eps) for all BLOCK_HEIGHT rows ----------
            ckl::eltwise_chain(
                EltwiseShape::tiles(BLOCK_HEIGHT),
                CopyTile<cb_partial_sumsq, Dst::D0, InputLifecycle::Streaming>{},
                ckl::MulUnary<>{inv_W_bits},
                ckl::AddUnary<>{eps_bits},
                ckl::Rsqrt<>{},
                PackTile<cb_recip_rms, OutputLifecycle::Streaming>{});

            // ---------- PASS 2: per-row Col-bcast normalize x * recip[row] ----------
            // recip held resident across the bh rows, indexed by a per-row TileOffset
            // (row r uses recip tile r); input streams Wt tiles/row in order and pops.
            // The per-row chains all read the SAME two CBs (cb_input_resident SrcA,
            // cb_recip_rms SrcB), so only row 0 needs the unpack data-format reconfig;
            // rows 1..bh-1 use Reconfig::None (the format is already set). [static-analyzer F1]
            ckl::eltwise_chain(
                EltwiseShape::tiles(Wt),
                BinaryFpu<
                    cb_input_resident,
                    cb_recip_rms,
                    BinaryFpuOp::Mul,
                    BroadcastDim::Col,
                    InputLifecycle::Streaming,
                    InputLifecycle::HeldBulk,
                    BinaryDataFormatReconfig::Input,  // establish the format once
                    Dst::D0,
                    OperandKind::Scalar,
                    OperandKind::Scalar,
                    TileOffset::Unset,
                    TileOffset::Set>{0, 0},
                PackTile<cb_pass2_out, OutputLifecycle::Streaming>{});
            for (uint32_t r = 1; r < BLOCK_HEIGHT; ++r) {
                ckl::eltwise_chain(
                    EltwiseShape::tiles(Wt),
                    BinaryFpu<
                        cb_input_resident,
                        cb_recip_rms,
                        BinaryFpuOp::Mul,
                        BroadcastDim::Col,
                        InputLifecycle::Streaming,       // x streams Wt/row, pops
                        InputLifecycle::HeldBulk,        // recip held, indexed by TileOffset r
                        BinaryDataFormatReconfig::None,  // format already set by row 0
                        Dst::D0,
                        OperandKind::Scalar,
                        OperandKind::Scalar,
                        TileOffset::Unset,
                        TileOffset::Set>{0, r},
                    PackTile<cb_pass2_out, OutputLifecycle::Streaming>{});
            }
            // recip was held across the row loop; free this block's bh tiles.
            cb_pop_front(cb_recip_rms, BLOCK_HEIGHT);
        }
        return;
    }

    for (uint32_t row = 0; row < num_rows; ++row) {
        // ---------- [RM] tilize input sticks -> resident tiles ----------
        if constexpr (layout_is_rm) {
            for (uint32_t c = 0; c < num_chunks; ++c) {
                ckl::tilize<reduce_block, cb_rm_in, cb_input_resident>(1);
            }
        }

        // ---------- PASS 1: sum of squares over the WHOLE resident shard ----------
        // Refinement 5: a single square + single reduce over the full shard — the
        // former DEST-level reduce_block chunking + reduce-Accumulate loop is gone.
        // The per-core shard width (Wt for Regime A, Wt_s for Regime B) is already
        // L1-bounded by the host A/B heuristic (the distributed W-split bounds the
        // resident shard), so an in-kernel chunking-and-accumulate was redundant.
        // ckl::eltwise_chain (square) and ckl::reduce (sum) are L1->L1 helpers that
        // tile their own work through DEST internally, so a square / reduce over the
        // full shard is a single call regardless of how many tiles wide it is — the
        // kernel never sees DEST capacity. cb_squared is sized to the shard width by
        // the descriptor (≤32 tiles for every routed shape).
        //
        // square resident[0 .. Wt) -> cb_squared (no pop of resident; reused in PASS-2)
        ckl::eltwise_chain(
            EltwiseShape::tiles(Wt),
            BinaryFpu<
                cb_input_resident,
                cb_input_resident,
                BinaryFpuOp::Mul,
                BroadcastDim::None,
                InputLifecycle::HeldBulk,
                InputLifecycle::HeldBulk,
                BinaryDataFormatReconfig::Input,
                Dst::D0,
                OperandKind::Block,
                OperandKind::Block,
                TileOffset::Set,
                TileOffset::Set>{0, 0},
            PackTile<cb_squared, OutputLifecycle::Streaming>{});

        // reduce the full squared shard into the local Sum(x^2) (single output tile)
        ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, ReduceInputPolicy::BulkWaitBulkPop>(
            cb_squared,
            cb_scaler,
            cb_partial_sumsq,
            ReduceInputBlockShape::of(1, Wt),
            ckl::ReduceInputMemoryLayout::contiguous());

        // ---------- (Regime B) cross-core combine ----------
        if constexpr (num_partials > 1) {
            // (1) Hand the FULLY-accumulated local Sum(x^2) to the reader on a dedicated
            //     single-push CB (copy also pops cb_partial_sumsq, emptying it).
            ckl::copy<cb_partial_sumsq, cb_local_sumsq>(EltwiseShape::tiles(1));

            // (2) Refinement 9 (Part B — combine compute): the reader all-gathered the K peer
            //     partials CONTIGUOUSLY into cb_partials_gathered; their plain elementwise sum
            //     is the global Sum(x^2) over the full W. Each partial is a PASS-1 REDUCE_ROW
            //     output (per-stick sum in column 0, every other column zero), so a SINGLE
            //     reduce<SUM, REDUCE_ROW> over the K-tile block sums the K tiles' rows into one
            //     output tile (col 0 = Σ_k partial_k per stick) — expressing the whole combine
            //     in ONE helper call. This REPLACES the former `copy` + (K-1) eltwise `add`s,
            //     each of which round-tripped L1 (read two operands, pack the running sum back):
            //     K-1 unnecessary L1 round-trips to sum K single tiles. The reduce LLK
            //     accumulates all K gathered tiles in DEST and packs exactly once. It reuses the
            //     same 1.0 SUM scaler as PASS-1 (cb_scaler). The Refinement-1 correctness
            //     invariant is preserved: the reader still hands over only the FULLY-accumulated
            //     local Σx² (via cb_local_sumsq, step 1) before the all-gather.
            ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, ReduceInputPolicy::BulkWaitBulkPop>(
                cb_partials_gathered,
                cb_scaler,
                cb_partial_sumsq,
                ReduceInputBlockShape::of(1, num_partials),
                ckl::ReduceInputMemoryLayout::contiguous());
        }

        // ---------- FINALIZE: rsqrt(sum * inv_W + eps) ----------
        ckl::eltwise_chain(
            EltwiseShape::tiles(1),
            CopyTile<cb_partial_sumsq, Dst::D0, InputLifecycle::Streaming>{},
            ckl::MulUnary<>{inv_W_bits},
            ckl::AddUnary<>{eps_bits},
            ckl::Rsqrt<>{},
            PackTile<cb_recip_rms, OutputLifecycle::Streaming>{});

        // ---------- PASS 2: normalize ----------
        if constexpr (has_gamma) {
            // Streaming fused Col->Row multiply, chunked by reduce_block. cb_normalized is
            // sized to ONE block (not Wt), so per-core L1 does not scale with shard width.
            // Per chunk: x * recip (Col bcast) -> cb_normalized; * gamma (Row bcast) -> out.
            for (uint32_t c = 0; c < num_chunks; ++c) {
                const uint32_t base = c * reduce_block;
                const uint32_t cw = (base + reduce_block <= Wt) ? reduce_block : (Wt - base);

                ckl::eltwise_chain(
                    EltwiseShape::tiles(cw),
                    BinaryFpu<
                        cb_input_resident,
                        cb_recip_rms,
                        BinaryFpuOp::Mul,
                        BroadcastDim::Col,
                        InputLifecycle::Streaming,  // input streams in order (its PASS-2 pop)
                        InputLifecycle::HeldBulk,   // recip held across chunks, freed below
                        BinaryDataFormatReconfig::Input,
                        Dst::D0,
                        OperandKind::Scalar,
                        OperandKind::Scalar>{},
                    PackTile<cb_normalized, OutputLifecycle::Streaming>{});

                ckl::eltwise_chain(
                    EltwiseShape::tiles(cw),
                    BinaryFpu<
                        cb_normalized,
                        cb_gamma,
                        BinaryFpuOp::Mul,
                        BroadcastDim::Row,
                        InputLifecycle::Streaming,  // normalized block streams/pops
                        InputLifecycle::HeldBulk,   // gamma held resident across rows
                        BinaryDataFormatReconfig::Input,
                        Dst::D0,
                        OperandKind::Scalar,
                        OperandKind::Block,
                        TileOffset::Unset,
                        TileOffset::Set>{0, base},
                    PackTile<cb_pass2_out, OutputLifecycle::Streaming>{});

                if constexpr (layout_is_rm) {
                    ckl::untilize<reduce_block, cb_pass2_out, cb_rm_out>(1);
                }
            }
            // recip was held across the chunk loop; free this row's tile.
            cb_pop_front(cb_recip_rms, 1);
        } else if constexpr (layout_is_rm) {
            // No-gamma RM: chunked Col-bcast multiply so the per-chunk untilize can
            // interleave (recip held across chunks, freed once at the row's end).
            for (uint32_t c = 0; c < num_chunks; ++c) {
                ckl::eltwise_chain(
                    EltwiseShape::tiles(reduce_block),
                    BinaryFpu<
                        cb_input_resident,
                        cb_recip_rms,
                        BinaryFpuOp::Mul,
                        BroadcastDim::Col,
                        InputLifecycle::Streaming,
                        InputLifecycle::HeldBulk,
                        BinaryDataFormatReconfig::Input,
                        Dst::D0,
                        OperandKind::Scalar,
                        OperandKind::Scalar>{},
                    PackTile<cb_pass2_out, OutputLifecycle::Streaming>{});

                ckl::untilize<reduce_block, cb_pass2_out, cb_rm_out>(1);
            }
            cb_pop_front(cb_recip_rms, 1);
        } else {
            // No-gamma TILE: single Col-bcast multiply over the whole shard
            // (byte-identical to the pre-R4 compute).
            ckl::mul<
                cb_input_resident,
                cb_recip_rms,
                cb_pass2_out,
                BroadcastDim::Col,
                InputLifecycle::Streaming,
                InputLifecycle::Bulk>(EltwiseShape::tiles(Wt));
        }
    }
}
