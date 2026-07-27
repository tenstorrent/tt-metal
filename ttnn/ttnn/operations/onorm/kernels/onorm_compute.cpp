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
// own DEST window, which is precisely the fusion the helper exists to expose —
// plus the P7b `SIGMOID_ENGINE == pack` branch, justified immediately below.
//
// HELPER SUBSTITUTION, declared up front (P7b, `SIGMOID_ENGINE == "pack"`).
// The default P7b engine is the helper `unary<Sigmoid<>, ...>`, and it stays the
// default.  The `pack` engine cannot be expressed through it: running the SFPU
// activation on TRISC2 means REPLACING the chain's `tile_regs_wait()` with
// `compute_kernel_lib::apply_activation_from_pack()` (its own math/pack SEMWAIT +
// packer dest-offset flip + WAIT_SFPU stall), and `eltwise_chain` has no
// packer-activation slot — the slot exists only on `matmul_block` /
// `add_bias_bcast_rows`, and this op has no matmul.  That is a named, specific
// helper limitation, not a preference for raw code.  The activation itself is
// still driven by the kernel_lib helpers `ActivationInitHelper<SIGMOID>::init()`
// and `apply_activation_from_pack<SIGMOID>()` from sfpu_activation_helpers.hpp;
// only the surrounding CopyTile/PackTile/CB scaffolding is hand-written, because
// there is no helper that composes the two.  If a packer-activation slot ever
// lands on `eltwise_chain`, this branch collapses into one `unary<>` call.
//
// Apart from that branch there is no raw tile_regs_*, no raw reduce_tile, no raw
// mul_tiles, and no CB op wrapped around any helper call anywhere below.
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
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    // CB slot map — injected as preprocessor defines from the ONE host-side
    // source of truth (`_CB_SLOTS` in onorm_program_descriptor.py).  These are
    // deliberately NOT literals here: a slot number restated in two files
    // drifts the moment either side is renumbered.
    constexpr uint32_t cb_o_tiles = ONORM_CB_O_TILES;
    constexpr uint32_t cb_gate_tiles = ONORM_CB_GATE_TILES;
    constexpr uint32_t cb_weight = ONORM_CB_WEIGHT;
    constexpr uint32_t cb_scaler = ONORM_CB_SCALER;
    constexpr uint32_t cb_out_tiles = ONORM_CB_OUT_TILES;
    constexpr uint32_t cb_sumsq = ONORM_CB_SUMSQ;
    constexpr uint32_t cb_rstd = ONORM_CB_RSTD;
    constexpr uint32_t cb_normed = ONORM_CB_NORMED;
    constexpr uint32_t cb_onorm = ONORM_CB_ONORM;
    // ROW-MAJOR staging for the re-tile.  `cb_rm_local` is where THIS core's
    // untilized token rows land, and `cb_rm_flat_rows` is the [TOKENS_PER_BLOCK,
    // cols_per_core*TILE_W] stripe this core tilizes.  When the token-block is not
    // split across cores (RETILE_GROUP_CORES == 1) the host makes them the SAME
    // slot — compute untilizes straight into the stripe it later tilizes, exactly
    // as before Refinement 2.  When it IS split, `cb_rm_local` is a separate
    // staging CB the writer drains and scatters, and `cb_rm_flat_rows` is filled
    // by REMOTE writers (see onorm_writer.cpp) — so this kernel's view of both is
    // identical either way and no code here branches on the group size.
    constexpr uint32_t cb_rm_local = ONORM_CB_RM_LOCAL;
    constexpr uint32_t cb_rm_flat_rows = ONORM_CB_RM_FLAT_ROWS;
    constexpr uint32_t cb_flat_tiles = ONORM_CB_FLAT_TILES;
    constexpr uint32_t cb_gate_sig = ONORM_CB_GATE_SIG;

    // --- Blocking Model parameters (compile-time; one source of truth on host) ---
    // Every count here is PER CORE: with the token-block split across
    // RETILE_GROUP_CORES cores, `norm_chunks` covers this core's token slice and
    // `cols_per_core` its flat output column slice.  At group size 1 both are the
    // whole block.
    constexpr uint32_t nb = get_compile_time_arg_val(0);                   // NORM_CHUNK_TOKENS (clamped)
    constexpr uint32_t norm_chunks = get_compile_time_arg_val(1);          // tokens_per_core / nb
    constexpr uint32_t v_tiles = get_compile_time_arg_val(2);              // V / TILE_W
    constexpr uint32_t cols_per_core = get_compile_time_arg_val(3);        // flat_tiles / group_cores
    constexpr uint32_t tile_rows_per_block = get_compile_time_arg_val(4);  // TOKENS_PER_BLOCK / TILE_H
    constexpr uint32_t gate_chunk_tiles = get_compile_time_arg_val(5);     // GATE_CHUNK_TILES (clamped)
    constexpr uint32_t gate_chunks = get_compile_time_arg_val(6);          // this core's out tiles ÷ chunk
    // Which TRISC issues the op's whole SFPU volume (SIGMOID_ENGINE knob).  The
    // ONORM_SIGMOID_* codes are defines emitted from the host's
    // `_SIGMOID_ENGINE_CODES` — this file never restates the integers.
    constexpr uint32_t sigmoid_engine = get_compile_time_arg_val(7);
    static_assert(
        sigmoid_engine == ONORM_SIGMOID_MATH || sigmoid_engine == ONORM_SIGMOID_PACK ||
            sigmoid_engine == ONORM_SIGMOID_ABLATE,
        "onorm: unknown SIGMOID_ENGINE code");
    // Tiles per DEST window in the two gate phases (GATE_DEST_TILES knob).  At 1
    // this is byte-identical to Phase 0: `InputLifecycle::Chunked` with a block of
    // one waits one tile / pops one tile per outer iter, exactly as
    // `InputLifecycle::Streaming` did.  Above 1 the chain stages that many tiles in
    // DEST per acquire/commit/wait/release round-trip.
    constexpr uint32_t gate_dest_tiles = get_compile_time_arg_val(8);
    // The gate walk is 1-D (`EltwiseShape::tiles`), so its iteration count IS the
    // tile count and `OperandKind::Block` is the only kind `Chunked` is legal with
    // (eltwise_chain.hpp:359-372) — a chunk-scaled wait on a Scalar-kind operand
    // would out-run the window and deadlock.
    constexpr auto gate_shape = ckl::EltwiseShape::tiles(gate_chunk_tiles, gate_dest_tiles);

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
            {
                MaybeDeviceZoneScope("onorm_p1_sumsq");
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
            }

            // ---- P2: mean over V (via the 1/V scaler tile) + eps + rsqrt ----
            // `o`'s tiled row axis is HV, so the reduction is over W (= V):
            // REDUCE_ROW with Ht = 1, Wt = 1 and `nb` batches.  A REDUCE_COL
            // here would silently reduce across heads.
            {
                MaybeDeviceZoneScope("onorm_p2_reduce_eps_rsqrt");
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
            }

            // ---- P4: normalize.  cb_rstd is a REDUCE_ROW result (col-0 valid)
            // so it broadcasts back across columns => BroadcastDim::Col, indexed
            // by row (OperandKind::Col).  `Bulk` on cb_o_tiles performs the
            // deferred pop of P1's held window — that release is what lets
            // O_DEPTH = 2 prefetch the next chunk.
            {
                MaybeDeviceZoneScope("onorm_p4_normalize");
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
            }

            // ---- P5: * weight.  weight is [1, V] (row-0 valid) so it
            // broadcasts down the rows => BroadcastDim::Row, indexed by column
            // tile (OperandKind::Row).  Row/Col kinds require a non-draining
            // lifecycle, hence HeldBulk: the weight is re-waited every chunk and
            // never popped.
            {
                MaybeDeviceZoneScope("onorm_p5_weight_scale");
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
            }

            // ---- P6: head-major -> row-major.  Each of the `nb` blocks
            // untilizes one token's [HV, V] tile-row into a contiguous 32x V
            // row-major region whose linear index h*V + c IS the flat feature
            // index — i.e. that token's whole flat feature row.
            //
            // Where those rows go depends on whether the token-block is split:
            // at group size 1 cb_rm_local IS cb_rm_flat_rows, so the rows
            // ACCUMULATE across the chunk loop into the [tokens_per_block, FLAT]
            // stripe P7a tilizes.  When split, cb_rm_local is a staging buffer the
            // writer drains per chunk and scatters column-wise, and P7a's stripe is
            // filled by the group's writers instead.  Either way this call is the
            // same and the bytes it emits are the same.
            {
                MaybeDeviceZoneScope("onorm_p6_untilize");
                ckl::untilize<v_tiles, cb_onorm, cb_rm_local>(nb);
            }
        }

        // ---- P7a: row-major -> flat token-major.  The tilize row stride IS
        // `cols_per_core` tiles, which is exactly the stripe width this core owns
        // (the whole block's FLAT width when the block is not split).
        {
            MaybeDeviceZoneScope("onorm_p7a_tilize");
            ckl::tilize<cols_per_core, cb_rm_flat_rows, cb_flat_tiles>(tile_rows_per_block);
        }

        for (uint32_t g = 0; g < gate_chunks; ++g) {
            // ---- P7b: sigmoid(gate) on the SFPU.  The op owns the sigmoid
            // (gate arrives pre-sigmoid) and normalization has already happened.
            // The result is deliberately PACKED TO L1 rather than kept in DEST —
            // see P7c.  Which TRISC issues the SFPU is the SIGMOID_ENGINE knob;
            // all three branches move the same `gate_chunk_tiles` tiles through
            // the same CBs with the same wait/push counts, so the CB ledger in
            // op_design.md §8.1 is engine-independent.
            {
                MaybeDeviceZoneScope("onorm_p7b_sigmoid");
                if constexpr (sigmoid_engine == ONORM_SIGMOID_MATH) {
                    // MATH thread (TRISC1): the helper's own chain is
                    // CopyTile(D0) -> sigmoid_tile(D0) -> PackTile, run
                    // `gate_dest_tiles` tiles per DEST window.
                    ckl::unary<
                        ckl::Sigmoid<>,
                        cb_gate_tiles,
                        cb_gate_sig,
                        ckl::InputLifecycle::Chunked,
                        ckl::OutputLifecycle::Chunked,
                        ckl::CopyTileReconfig::Input,
                        ckl::PackTileReconfig::Output,
                        ckl::OperandKind::Block>(gate_shape);
                } else if constexpr (sigmoid_engine == ONORM_SIGMOID_PACK) {
                    // PACK thread (TRISC2): the same 6-entry-LUT sigmoid, issued
                    // at the pack stage.  See the HELPER SUBSTITUTION note at the
                    // head of this file for why this cannot be a `unary<>` call.
                    //
                    // The pack-side SFPU init is re-issued per gate chunk rather
                    // than once at boot: `_init_sigmoid_` parks the LUT in
                    // LReg0/1/2/4/5/6, and the SFPU (hence those LRegs) is shared
                    // with the MATH thread's `rsqrt_tile_init()` in P2 — a
                    // boot-only init would be clobbered by the first chunk.
                    using SigmoidPack = ckl::ActivationInitHelper<KernelActivation::SIGMOID>;
                    SigmoidPack::init();
                    copy_tile_init(cb_gate_tiles);
                    // Same `gate_dest_tiles` DEST window as the two chain-driven
                    // engines, so the knob means one thing across all three.
                    for (uint32_t t = 0; t < gate_chunk_tiles; t += gate_dest_tiles) {
                        cb_wait_front(cb_gate_tiles, gate_dest_tiles);
                        cb_reserve_back(cb_gate_sig, gate_dest_tiles);
                        tile_regs_acquire();
                        for (uint32_t j = 0; j < gate_dest_tiles; ++j) {
                            copy_tile(cb_gate_tiles, j, j);
                        }
                        tile_regs_commit();
                        // REPLACES tile_regs_wait(): does the math/pack SEMWAIT,
                        // flips the packer dest offset, runs sigmoid_tile_pack on
                        // TRISC2 for each DEST tile, then stalls the packer on
                        // SFPU completion.
                        ckl::apply_activation_from_pack<KernelActivation::SIGMOID>(gate_dest_tiles);
                        for (uint32_t j = 0; j < gate_dest_tiles; ++j) {
                            pack_tile(j, cb_gate_sig);
                        }
                        tile_regs_release();
                        cb_push_back(cb_gate_sig, gate_dest_tiles);
                        cb_pop_front(cb_gate_tiles, gate_dest_tiles);
                    }
                } else {
                    // ABLATION (measurement only, numerically WRONG): the sigmoid
                    // payload removed, every CB wait/push, DEST window and NoC
                    // transfer around it kept.  `device_ns(math) - device_ns(ablate)`
                    // is the SFPU payload's true contribution to the critical path
                    // — the number the /perf-measure ablation method asks for, and
                    // the one a per-phase zone (which includes cb_wait_front) cannot
                    // give.
                    ckl::copy<
                        cb_gate_tiles,
                        cb_gate_sig,
                        ckl::InputLifecycle::Chunked,
                        ckl::OutputLifecycle::Chunked,
                        ckl::CopyTileReconfig::Input,
                        ckl::PackTileReconfig::Output,
                        ckl::OperandKind::Block>(gate_shape);
                }
            }

            // ---- P7c: the gate multiply, on the **FPU**, fed from L1 by the
            // unpacker.  This is the op's highest-volume multiply (every output
            // tile of every block), so the engine choice dominates: an SFPU
            // multiply measures 0.58x, and fusing sigmoid->multiply through DEST
            // into an FPU consumer measures 0.82x while the L1 round-trip is
            // 1.22x faster.  Hence the deliberate cb_gate_sig hop.  Do NOT
            // collapse P7b/P7c into one DEST-resident chain.
            //
            // P7c is P7b's 1:1 twin, so it carries the SAME `gate_dest_tiles`
            // block factor.  Coarsening one gate phase alone would just push the
            // per-DEST-window cost onto the other half of the pair.
            {
                MaybeDeviceZoneScope("onorm_p7c_gate_mul");
                ckl::mul<
                    cb_flat_tiles,
                    cb_gate_sig,
                    cb_out_tiles,
                    ckl::BroadcastDim::None,
                    ckl::InputLifecycle::Chunked,
                    ckl::InputLifecycle::Chunked,
                    ckl::OutputLifecycle::Chunked,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::PackTileReconfig::Output,
                    ckl::OperandKind::Block,
                    ckl::OperandKind::Block>(gate_shape);
            }
        }
    }
}
