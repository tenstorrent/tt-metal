// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// onorm compute (unpack / math / pack TRISCs) — the fused KDA s6 tail.
//
//   out = flatten_heads( RMSNorm_over_V(o) * weight ) * sigmoid(gate)
//
// Per token-block (= `tokens_per_block` tokens of one batch):
//
//   for each chunk of `norm_chunk_tokens` tokens:
//     P1  o^2, DEST-accumulated over v_tiles         -> cb_sumsq   (1 tile/token)
//     P2  reduce<SUM,REDUCE_ROW> with the 1/V scaler,
//         + eps + rsqrt fused in the SAME DEST window -> cb_rstd    (col-0 valid)
//     P4  o * rstd            (bcast Col)            -> cb_normed
//     P5  * weight            (bcast Row)            -> cb_onorm
//     P6  untilize<v_tiles>                          -> cb_rm_flat_rows (ROW-MAJOR)
//   P7a tilize<flat_tiles>                           -> cb_flat_tiles
//   for each chunk of `gate_chunk_tiles` output tiles:
//     P7b sigmoid(gate) (SFPU)                       -> cb_gate_sig
//     P7c flat * sigmoid(gate) (**FPU**)             -> cb_out_tiles
//
// EVERY mechanism is a kernel_lib helper.  The only raw compute-API calls in
// this file are the three LLK calls inside P2's `post_reduce_op` lambda
// (binop_with_scalar_tile_init / add_unary_tile / rsqrt_tile) — and that lambda
// *is* the reduce helper's documented epilogue hook: it runs inside the helper's
// own DEST window, which is precisely the fusion the helper exists to expose.
// There is no raw tile_regs_*, no raw reduce_tile, no raw mul_tiles, no raw
// pack_tile, and no CB op wrapped around any helper call anywhere below.
//
// The re-tile (P6 -> P7a) deviates from the task rules' suggested
// `tilize<..., StreamMode::PerTile>` + 2-tile cb_flat.  That combination is not
// implementable here and would deadlock: tilize and its consumer both run in
// THIS kernel, so all three TRISCs execute them in sequence — tilize's PACK
// thread would block in cb_reserve_back(cb_flat_tiles, 1) at the 3rd tile, while
// the consumer's cb_pop_front is only reached by UNPACK after UNPACK finishes
// every tilize unpack, and UNPACK is throttled by MATH which is throttled by
// PACK.  StreamMode::PerTile only pays when the consumer is a *different* RISC.
// Phase 1 therefore uses the default StreamMode::Atomic (bit-identical output
// bytes) with cb_flat_tiles sized to one block.  See op_design.md §6.1.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_o_tiles = 0;
    constexpr uint32_t cb_gate_tiles = 1;
    constexpr uint32_t cb_weight = 2;
    constexpr uint32_t cb_scaler = 8;
    constexpr uint32_t cb_out_tiles = 16;
    constexpr uint32_t cb_sumsq = 24;
    constexpr uint32_t cb_rstd = 25;
    constexpr uint32_t cb_normed = 27;
    constexpr uint32_t cb_onorm = 28;
    constexpr uint32_t cb_rm_flat_rows = 29;
    constexpr uint32_t cb_flat_tiles = 30;
    constexpr uint32_t cb_gate_sig = 31;

    // --- Blocking Model parameters (compile-time; one source of truth on host) ---
    constexpr uint32_t nb = get_compile_time_arg_val(0);                   // NORM_CHUNK_TOKENS
    constexpr uint32_t norm_chunks = get_compile_time_arg_val(1);          // TOKENS_PER_BLOCK / nb
    constexpr uint32_t v_tiles = get_compile_time_arg_val(2);              // V / TILE_W
    constexpr uint32_t flat_tiles = get_compile_time_arg_val(3);           // FLAT / TILE_W
    constexpr uint32_t tile_rows_per_block = get_compile_time_arg_val(4);  // TOKENS_PER_BLOCK / TILE_H
    constexpr uint32_t gate_chunk_tiles = get_compile_time_arg_val(5);     // GATE_CHUNK_TILES
    constexpr uint32_t gate_chunks = get_compile_time_arg_val(6);          // flat/block ÷ gate_chunk_tiles

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t eps_bits = get_arg_val<uint32_t>(1);  // fp32 bit pattern of epsilon

    // Exactly once, first statement of the kernel body.  One boot serves every
    // phase because all 12 CBs share Float16_b.
    compute_kernel_hw_startup(cb_o_tiles, cb_weight, cb_onorm);

    // P2's fused epilogue: `+ eps` then `rsqrt`, inside the reduce's own DEST
    // window.  epsilon is applied to the MEAN OF SQUARES, matching
    // torch: x * rsqrt(mean(x^2) + eps).
    const auto eps_then_rsqrt = [eps_bits](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        add_unary_tile(dst_idx, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    };

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        for (uint32_t chunk = 0; chunk < norm_chunks; ++chunk) {
            // ---- P1: sum of squares over V, DEST-accumulated ----
            // One outer row per token: D0 stays sticky across that row's
            // v_tiles inputs and is packed once.  `HeldBulk` on both operands
            // waits for the chunk's o tiles but does NOT pop them — P4 needs
            // them again.  With fp32_dest_acc_en the accumulate is fp32.
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(nb, v_tiles),
                ckl::BinaryFpu<
                    cb_o_tiles,
                    cb_o_tiles,
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::None,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::Dst::D0,
                    ckl::OperandKind::Block,
                    ckl::OperandKind::Block,
                    ckl::TileOffset::Unset,
                    ckl::TileOffset::Unset,
                    ckl::DestAccumulation::Enabled>{},
                ckl::PackTile<
                    cb_sumsq,
                    ckl::OutputLifecycle::DestAccumulation,
                    ckl::PackTileReconfig::Output,
                    ckl::Dst::D0>{});

            // ---- P2: mean over V (via the 1/V scaler tile) + eps + rsqrt ----
            // `o`'s tiled row axis is HV, so the reduction is over W (= V):
            // REDUCE_ROW with Ht = 1, Wt = 1 and `nb` batches.  A REDUCE_COL
            // here would silently reduce across heads.
            ckl::reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                cb_sumsq,
                cb_scaler,
                cb_rstd,
                ckl::ReduceInputPolicy::BulkWaitBulkPop,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ckl::ReduceAlgorithm::Auto>(
                ckl::ReduceInputBlockShape::of(1, 1, nb),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::NoAccumulation{},
                eps_then_rsqrt);

            // ---- P4: normalize.  cb_rstd is a REDUCE_ROW result (col-0 valid)
            // so it broadcasts back across columns => BroadcastDim::Col, indexed
            // by row (OperandKind::Col).  `Bulk` on cb_o_tiles performs the
            // deferred pop of P1's held window — that release is what lets
            // O_DEPTH = 2 prefetch the next chunk.
            ckl::mul<
                cb_o_tiles,
                cb_rstd,
                cb_normed,
                ckl::BroadcastDim::Col,
                ckl::InputLifecycle::Bulk,
                ckl::InputLifecycle::Bulk,
                ckl::OutputLifecycle::Streaming,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::PackTileReconfig::Output,
                ckl::OperandKind::Block,
                ckl::OperandKind::Col>(ckl::EltwiseShape::grid(nb, v_tiles));

            // ---- P5: * weight.  weight is [1, V] (row-0 valid) so it
            // broadcasts down the rows => BroadcastDim::Row, indexed by column
            // tile (OperandKind::Row).  Row/Col kinds require a non-draining
            // lifecycle, hence HeldBulk: the weight is re-waited every chunk and
            // never popped.
            ckl::mul<
                cb_normed,
                cb_weight,
                cb_onorm,
                ckl::BroadcastDim::Row,
                ckl::InputLifecycle::Bulk,
                ckl::InputLifecycle::HeldBulk,
                ckl::OutputLifecycle::Streaming,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::PackTileReconfig::Output,
                ckl::OperandKind::Block,
                ckl::OperandKind::Row>(ckl::EltwiseShape::grid(nb, v_tiles));

            // ---- P6: head-major -> row-major.  Each of the `nb` blocks
            // untilizes one token's [HV, V] tile-row into a contiguous 32x V
            // row-major region whose linear index h*V + c IS the flat feature
            // index.  cb_rm_flat_rows ACCUMULATES across the chunk loop and is
            // only complete (one full [tokens_per_block, FLAT] stripe) after the
            // last chunk.
            ckl::untilize<v_tiles, cb_onorm, cb_rm_flat_rows>(nb);
        }

        // ---- P7a: row-major -> flat token-major.  The tilize row stride IS
        // `flat_tiles`, which is exactly the stripe P6 built.
        ckl::tilize<flat_tiles, cb_rm_flat_rows, cb_flat_tiles>(tile_rows_per_block);

        for (uint32_t g = 0; g < gate_chunks; ++g) {
            // ---- P7b: sigmoid(gate) on the SFPU.  The op owns the sigmoid
            // (gate arrives pre-sigmoid) and normalization has already happened.
            // The result is deliberately PACKED TO L1 rather than kept in DEST —
            // see P7c.
            ckl::unary<ckl::Sigmoid<>, cb_gate_tiles, cb_gate_sig>(ckl::EltwiseShape::tiles(gate_chunk_tiles));

            // ---- P7c: the gate multiply, on the **FPU**, fed from L1 by the
            // unpacker.  This is the op's highest-volume multiply (every output
            // tile of every block), so the engine choice dominates: an SFPU
            // multiply measures 0.58x, and fusing sigmoid->multiply through DEST
            // into an FPU consumer measures 0.82x while the L1 round-trip is
            // 1.22x faster.  Hence the deliberate cb_gate_sig hop.  Do NOT
            // collapse P7b/P7c into one DEST-resident chain.
            ckl::mul<cb_flat_tiles, cb_gate_sig, cb_out_tiles>(ckl::EltwiseShape::tiles(gate_chunk_tiles));
        }
    }
}
