// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// moe_fused_swiglu — COMPUTE.
//
// Per M-block, on every one of the HGROUPS x KGROUPS worker cores:
//   1. fused tilize of the row-major x slice this core injects (bf16 path only);
//   2. gate matmul and up matmul over the SAME resident x block (one K-block == the whole
//      per-row K extent, so `in0_policy = WaitAndRetainOnLastBlock` retains it for both and the
//      kernel pops it once at the end — the "cb_x_tiles consumed twice" contract, and NOT a
//      second multicast of x);
//   3. the cross-column reduce + SwiGLU epilogue, in ONE OF TWO SHAPES (MOE_SWIGLU_REDUCE):
//        `tree`    — the reduce adds funnel whole blocks up a binary tree (in-place FPU add per
//                    child) and the ROOT alone runs the epilogue: its final gate add carries SiLU on
//                    the PACKER thread, walked m_eff times, then the SwiGLU multiply through L1;
//        `scatter` — PERF 2, the default. Every core reduces only its OWN SLICE of the block over all
//                    KGROUPS contributors and runs the epilogue on that slice, so the epilogue — 85 %
//                    of the shipped stage's cost, dominated by the 48-tile SFPU SiLU — is
//                    parallelised KGROUPS ways and the m_eff-call bias walk collapses to one call;
//   4. the `down` matmul over HGROUPS phase-2 K-blocks with packer L1 accumulation, then the one
//      genuine dtype boundary (bf16 partials -> bfp8 output).
//
// Everything here is a kernel_lib helper. The ONE raw access is the L1 mailbox read of the
// device-resident token count: the M-block trip count must be identical on all three TRISCs, and
// `cb_wait_front` in a compute kernel is UNPACK-only, so a CB handoff would let MATH/PACK diverge.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#include "moe_fused_swiglu_common.hpp"  // the ONE definition of the mailbox word layout

using namespace compute_kernel_lib;

// PER-STAGE ZONES — PERMANENT, always compiled, free with the profiler off. A compute TU does NOT
// get the profiler through `dataflow_api.h` (it must not see the dataflow API at all), which is
// exactly why `perf_instrumentation.hpp` exists. 6 records per M-block here.

constexpr uint32_t M_BLOCK = get_compile_time_arg_val(0);
constexpr uint32_t KR_PAD = get_compile_time_arg_val(1);
constexpr uint32_t HN_PAD = get_compile_time_arg_val(2);
constexpr uint32_t EC_MAX = get_compile_time_arg_val(3);  // phase-2 N stride (uniform CB increment)
constexpr uint32_t HGROUPS = get_compile_time_arg_val(4);
constexpr uint32_t HID_T = get_compile_time_arg_val(5);
constexpr uint32_t INPUT_FORMAT = get_compile_time_arg_val(6);
constexpr uint32_t OUT_SUBBLOCK_H_GU = get_compile_time_arg_val(7);
constexpr uint32_t MAILBOX_MAGIC = get_compile_time_arg_val(8);
// Smallest legal `m_eff` (= OUT_SUBBLOCK_H_GU rounded up to a power of two, so the gate/up
// `m_eff / OUT_SUBBLOCK_H_GU` below is always exact). One host-side definition, identical in all
// three kernels — see m_tiles_eff().
constexpr uint32_t M_EFF_MIN = get_compile_time_arg_val(9);
// `down` output sub-block HEIGHT (Refinement 2 lever 3). Separate from the gate/up height because
// `down`'s width is `ec` (2-3) against a DEST budget of 8, while gate/up's is already HN_PAD = 6.
// Host-derived as the largest power of two whose sub-block fits DEST; taken as min(.., m_eff) below
// so it never constrains the runtime M shrink.
constexpr uint32_t OUT_SUBBLOCK_H_DN = get_compile_time_arg_val(10);
// Concurrent child landing slots in cb_reduce_*_in (Refinement 2 lever 1). The reduce loop below
// walks the parent's invite WAVES with exactly this granularity, so it must be the same number the
// reader uses; 1 is the Phase-0 one-child-at-a-time protocol.
constexpr uint32_t REDUCE_SLOTS = get_compile_time_arg_val(11);
// gate/up in1 sub-block WIDTH in hidden tiles (Refinement 2 lever 2). HN_PAD == one sub-block ==
// the Phase-0 shape; a divisor of it splits the block so a downstream per-sub-block reduce has
// something to pipeline against. Layout-safe at OUT_SUBBLOCK_H_GU == 1 — SubblockMajor walks
// in0_subblock outer / in1_subblock inner, so the tile order stays m*HN_PAD + n either way.
constexpr uint32_t HN_BLOCK = get_compile_time_arg_val(12);
// PERF 3 — the hidden-axis chunk the gate/up weight stream is published and consumed in. 1 restores
// the single whole-block matmul per matrix. Host-guaranteed to divide HN_PAD, to be a multiple of
// HN_BLOCK, and to leave every column group at least one real column per chunk.
constexpr uint32_t GU_CHUNK_W = HN_PAD / GU_CHUNKS;
// PERF 1 — eltwise DEST-window block size. Tiles per `tile_regs_acquire/commit/wait/release` cycle
// in every eltwise pass below. See ELTWISE_BLK in the program descriptor for the mechanism and the
// measurement; 1 reproduces the pre-Perf-1 per-tile shape byte-for-byte.
constexpr uint32_t ELTWISE_BLK = get_compile_time_arg_val(13);

constexpr uint32_t cb_x_in = get_compile_time_arg_val(14);
constexpr uint32_t cb_x_tiles = get_compile_time_arg_val(15);
constexpr uint32_t cb_x_stage = get_compile_time_arg_val(16);
constexpr uint32_t cb_w_gate = get_compile_time_arg_val(17);
constexpr uint32_t cb_w_up = get_compile_time_arg_val(18);
constexpr uint32_t cb_w_down = get_compile_time_arg_val(19);
constexpr uint32_t cb_gate_acc = get_compile_time_arg_val(20);
constexpr uint32_t cb_up_acc = get_compile_time_arg_val(21);
constexpr uint32_t cb_gate_send = get_compile_time_arg_val(22);
constexpr uint32_t cb_up_send = get_compile_time_arg_val(23);
constexpr uint32_t cb_gate_silu = get_compile_time_arg_val(24);
constexpr uint32_t cb_reduce_gate_in = get_compile_time_arg_val(25);
constexpr uint32_t cb_reduce_up_in = get_compile_time_arg_val(26);
constexpr uint32_t cb_h_local = get_compile_time_arg_val(27);
constexpr uint32_t cb_h = get_compile_time_arg_val(28);
constexpr uint32_t cb_out_interm = get_compile_time_arg_val(29);
constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(30);

// PERF 2 — REDUCE-SCATTER WITH A DISTRIBUTED EPILOGUE (MOE_SWIGLU_REDUCE=scatter). 1 selects the
// scatter path below; 0 reproduces the binary reduce tree byte-for-byte. See the knob in the program
// descriptor for the measurement (2.80x isolated, ~85 % of it the epilogue) and the predicate.
constexpr uint32_t SCATTER = get_compile_time_arg_val(31);
constexpr uint32_t KGROUPS = get_compile_time_arg_val(32);       // column height == contributor count
constexpr uint32_t GATHER_PAGES = get_compile_time_arg_val(33);  // the WHOLE landing CB, in tiles
constexpr uint32_t DEST_LIMIT = get_compile_time_arg_val(34);    // DEST_AUTO_LIMIT_TILES
constexpr uint32_t cb_gather_gate = get_compile_time_arg_val(35);
constexpr uint32_t cb_gather_up = get_compile_time_arg_val(36);
constexpr uint32_t cb_slice_gate = get_compile_time_arg_val(37);
constexpr uint32_t cb_slice_up = get_compile_time_arg_val(38);
constexpr uint32_t cb_h_slice = get_compile_time_arg_val(39);

constexpr uint32_t TILE_H = 32;

// ---------------------------------------------------------------------------
// PERF 1 — BLOCKED ELTWISE PASSES.
//
// `input(cb)` / `output(cb)` default to per-TILE wait/pop/reserve/push, and `eltwise_chain` only
// honours a block size when every CB reader uses `Upfront` / `Cumulative` / `None+None` /
// `PerChunk+PerChunk` (`eltwise_chain.inl:1511`); otherwise it SILENTLY clamps `block_size` to 1
// (`eltwise_chain.inl:3054`). So the convenient spelling costs one full DEST sync round trip PER
// TILE against a DEST budget of 8. These specs opt into the chunked lifecycle so the same math runs
// in `ceil(n / ELTWISE_BLK)` DEST windows instead of `n`.
//
// The tail is safe by construction: numeric `EltwiseShape::tiles(n, blk)` uses
// `BlockTailSync::ValidTiles`, so the last window synchronizes only its valid remainder and the
// per-CB wait/pop/reserve/push TOTALS are unchanged — which is what the cross-core reduce requires,
// since the child ships and the parent consumes whole `m_eff * HN_PAD` blocks (6/12/24/48 tiles).
// ELTWISE_BLK == 1 collapses every one of these back to the pre-Perf-1 shape.
// `OperandKind::Block` is REQUIRED, not decorative: `is_legal_input_policy_for_kind`
// (`eltwise_chain.inl:152-172`) admits `PerChunk+PerChunk` only for `Block`. The default `Scalar`
// kind pins the read to tile 0 and relies on the per-tile POP to advance the CB read pointer — which
// is precisely the mechanism a chunked lifecycle removes, so the index has to advance with the walk
// instead. Getting this wrong is a compile error, not a silent wrong answer.
constexpr auto blk_in(uint32_t cb) { return input(cb, WaitPolicy::PerChunk, PopPolicy::PerChunk, OperandKind::Block); }
constexpr auto blk_out(uint32_t cb) { return output(cb, ReservePolicy::PerChunk, PushPolicy::PerChunk); }
ALWI auto blk_shape(uint32_t n) { return EltwiseShape::tiles(n, ELTWISE_BLK); }

// Per-K-block FMA step count for the gate/up matmul: the padded K slot is KR_PAD tiles wide but
// only `kr` of them are real, so the loop bound shrinks and the pad tiles are never touched.
struct KrSteps {
    uint32_t kr;
    ALWI uint32_t operator()(uint32_t, uint32_t) const { return kr; }
};

// Per-K-block FMA step count for the `down` matmul: every h round carries HN_PAD hidden tiles,
// except the last column-group, which owns fewer (HGROUPS * HN_PAD >= HID_T by construction).
struct HnSteps {
    uint32_t last;
    ALWI uint32_t operator()(uint32_t block, uint32_t block_k) const { return (block == HGROUPS - 1) ? last : block_k; }
};

void kernel_main() {
    const uint32_t mailbox_addr = get_arg_val<uint32_t>(0);
    const uint32_t kr = get_arg_val<uint32_t>(1);
    const uint32_t hn = get_arg_val<uint32_t>(2);
    const uint32_t ec = get_arg_val<uint32_t>(3);
    const uint32_t is_root = get_arg_val<uint32_t>(4);
    const uint32_t num_children = get_arg_val<uint32_t>(5);
    const uint32_t my_col = get_arg_val<uint32_t>(6);  // grid column == this core's x-injection slot
    const uint32_t my_row = get_arg_val<uint32_t>(7);  // row in the column == which scatter slice I own

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_tiles, cb_w_gate, cb_gate_acc);
    // SiLU rides the packer thread of the root's final reduce add; the helpers never issue this.
    ActivationInitHelper<KernelActivation::SILU>::init();

    // Device-resident token count. All three TRISCs spin here independently so the M-block trip
    // count is thread-uniform (see the file header). The `fence` is exactly what
    // `invalidate_l1_cache()` compiles to on Blackhole (risc_common.h) — spelled out here because
    // that helper lives behind a dataflow-only include.
    volatile tt_l1_ptr uint32_t* mbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox_addr);
    while (mbox[moe_fused_swiglu::MBOX_READY] != MAILBOX_MAGIC) {
        asm volatile("fence" ::: "memory");
    }
    const uint32_t m_t = mbox[moe_fused_swiglu::MBOX_M_T];
    const uint32_t m_blocks = mbox[moe_fused_swiglu::MBOX_M_BLOCKS];

    CircularBuffer x_buf(cb_x_tiles);
    CircularBuffer wg_buf(cb_w_gate);
    CircularBuffer wu_buf(cb_w_up);
    CircularBuffer wd_buf(cb_w_down);
    CircularBuffer gate_buf(cb_gate_acc);
    CircularBuffer up_buf(cb_up_acc);
    CircularBuffer rg_buf(cb_reduce_gate_in);
    CircularBuffer ru_buf(cb_reduce_up_in);
    CircularBuffer silu_buf(cb_gate_silu);
    // PERF 2 — the scatter path's landing + slice buffers. A `CircularBuffer` only wraps the index
    // (no L1 access at construction), so naming both paths' CBs unconditionally is free; the inactive
    // path's CBs are allocated ONE page each host-side and never touched.
    CircularBuffer gg_buf(cb_gather_gate);
    CircularBuffer gu_buf(cb_gather_up);
    CircularBuffer sg_buf(cb_slice_gate);
    CircularBuffer h_buf(cb_h);
    CircularBuffer out_interm_buf(cb_out_interm);
    CircularBuffer out_tiles_buf(cb_out_tiles);

    // The reduce CBs are the ONE M-scaled pair that is always pushed WHOLE: the child unicasts to
    // its own cb_reduce_*_in write pointer (+ its slot stride) as a proxy for the parent's, which
    // only holds while every push wraps back to the CB base. Live tokens occupy the first
    // m_eff*HN_PAD tiles of each slot; the reader pushes REDUCE_SLOTS slots per invite WAVE.
    constexpr uint32_t REDUCE_SLOT_TILES = M_BLOCK * HN_PAD;

    const uint32_t hn_last = HID_T - (HGROUPS - 1) * HN_PAD;

    for (uint32_t b = 0; b < m_blocks; ++b) {
        // The RUNTIME token tile-rows this block works on — the same number the reader uses for its
        // x-multicast rounds and the writer for its CB waits (moe_fused_swiglu_common.hpp). Every
        // shape and trip count below is derived from it, so count 128 does HALF the gate/up matmul,
        // reduce and `down` work of count 256 instead of the same amount.
        const uint32_t m_eff = moe_fused_swiglu::m_tiles_eff(m_t, b, M_BLOCK, M_EFF_MIN);
        const uint32_t x_slot_tiles = m_eff * KR_PAD;
        const uint32_t gu_block_tiles = m_eff * HN_PAD;
        const uint32_t out_block_tiles = m_eff * EC_MAX;

        // gate/up: [m_eff, HN_PAD] = x[m_eff, kr] @ W[kr, HN_PAD]. ONE K-block whose width is the
        // whole per-row K extent, which is what lets both matmuls read the same resident in0.
        // PERF 3 — the in1 sub-blocking is now WITHIN one N-chunk; the host keeps HN_BLOCK a divisor
        // of GU_CHUNK_W, so at GU_CHUNKS == 1 (GU_CHUNK_W == HN_PAD) this is the Phase-0 shape.
        constexpr uint32_t GU_IN1_SUBBLOCKS = GU_CHUNK_W / HN_BLOCK;

        // down: [m_eff, ec] = h[m_eff, HGROUPS*HN_PAD] @ W_down[.., ec], HGROUPS K-blocks.
        // The FMA width is the real `ec`, but the in1 read stride and the output row stride are the
        // uniform EC_MAX so every phase-2 CB increment is core-independent.
        // Both heights are powers of two and m_eff is a power of two, so min(.,.) DIVIDES m_eff
        // exactly — the `down` height never forces a larger m_eff (Refinement 1 stays intact).
        const uint32_t sbh_dn = (OUT_SUBBLOCK_H_DN < m_eff) ? OUT_SUBBLOCK_H_DN : m_eff;
        const MatmulBlockShape shape_dn = MatmulBlockShape::of(m_eff / sbh_dn, 1, sbh_dn, ec, HN_PAD, HGROUPS);

        // ---- 1. fused tilize of the x tile-rows this core injects (bf16 ROW_MAJOR only) ----
        if constexpr (INPUT_FORMAT == 0) {
            MaybeDeviceZoneScope("compute_tilize");
            const uint32_t n_inject = moe_fused_swiglu::inject_rows(m_eff, my_col, HGROUPS);
            for (uint32_t i = 0; i < n_inject; ++i) {
                // Asymmetric page mode: TILE_H row-major stick slices in -> KR_PAD bfp8 tiles out.
                tilize<KR_PAD, cb_x_in, cb_x_stage>(1, TILE_H);
            }
        }

        // ---- 2. gate and up over the same resident x block ----
        //
        // PERF 3 — the two matmuls walk the HIDDEN axis in `GU_CHUNKS` chunks, INTERLEAVED
        // (gate c, up c, gate c+1, ...), so each chunk's matmul runs while the NEXT chunk's bfp4
        // block is still in DRAM. Chunks are independent full-K matmuls, so there is no
        // cross-chunk accumulation to pay for; the only thing the split needs is that each chunk
        // pack into ITS OWN columns of one shared m-major block rather than producing a
        // chunk-major one, which is what `out_col_offset` + `caller_owns_pack_target` buy. The
        // block the reduce and the scatter see is byte-identical to the single-call layout.
        //
        // INTERLEAVED, not gate-then-up: with gate first, `up`'s whole stream lands during gate's
        // matmul and only gate's own chunking overlaps anything. Alternating spends both NoCs'
        // streams under both matmuls.
        {
            MaybeDeviceZoneScope("compute_gateup");
            gate_buf.reserve_back(gu_block_tiles);
            up_buf.reserve_back(gu_block_tiles);
            for (uint32_t c = 0; c < GU_CHUNKS; ++c) {
                // The ragged column (hn < HN_PAD) narrows the FMA width of the chunk it falls in;
                // the host guarantees every chunk keeps at least one real column. 0 means "full".
                const uint32_t h0 = c * GU_CHUNK_W;
                const uint32_t valid = (hn > h0) ? (hn - h0) : 0;
                MatmulBlockShape shape_c = MatmulBlockShape::of(
                    m_eff / OUT_SUBBLOCK_H_GU, GU_IN1_SUBBLOCKS, OUT_SUBBLOCK_H_GU, HN_BLOCK, KR_PAD, 1);
                shape_c.last_in1_subblock_w_valid =
                    (valid < GU_CHUNK_W) ? (valid - (GU_IN1_SUBBLOCKS - 1) * HN_BLOCK) : 0;

                matmul_block<
                    /*transpose=*/false,
                    /*packer_l1_acc=*/true,
                    LastBlockTarget::Interm,
                    OutputCBLayout::TileRowMajor,
                    matmul_config::InitMode::Short,
                    InputPolicy::WaitAndRetainOnLastBlock,
                    InputPolicy::WaitAndPopPerKBlock,
                    NoPostCompute,
                    NoPreKBlock,
                    NoPostKBlock,
                    /*untilize_block_ct_dim=*/0,
                    KrSteps,
                    NoIn0Source,
                    NoIn1BaseOffset,
                    /*caller_owns_pack_target=*/true>(
                    x_buf,
                    wg_buf,
                    gate_buf,
                    gate_buf,
                    shape_c,
                    {},
                    {},
                    /*in1_per_core_w=*/GU_CHUNK_W,
                    /*out_row_width=*/HN_PAD,
                    {},
                    KrSteps{kr},
                    {},
                    {},
                    /*out_col_offset=*/h0);

                matmul_block<
                    /*transpose=*/false,
                    /*packer_l1_acc=*/true,
                    LastBlockTarget::Interm,
                    OutputCBLayout::TileRowMajor,
                    matmul_config::InitMode::Short,
                    InputPolicy::WaitAndRetainOnLastBlock,
                    InputPolicy::WaitAndPopPerKBlock,
                    NoPostCompute,
                    NoPreKBlock,
                    NoPostKBlock,
                    /*untilize_block_ct_dim=*/0,
                    KrSteps,
                    NoIn0Source,
                    NoIn1BaseOffset,
                    /*caller_owns_pack_target=*/true>(
                    x_buf,
                    wu_buf,
                    up_buf,
                    up_buf,
                    shape_c,
                    {},
                    {},
                    /*in1_per_core_w=*/GU_CHUNK_W,
                    /*out_row_width=*/HN_PAD,
                    {},
                    KrSteps{kr},
                    {},
                    {},
                    /*out_col_offset=*/h0);
            }
            gate_buf.push_back(gu_block_tiles);
            up_buf.push_back(gu_block_tiles);
            // packer_l1_acc leaves L1 accumulation ENABLED after the last chunk (the `down` matmul
            // below carries the same note). The reduce chain that follows would otherwise ACCUMULATE
            // onto stale L1 instead of overwriting.
            pack_reconfig_l1_acc(0);
        }

        // PERF 2 — MY SLICE of this block's T = m_eff*HN_PAD tile block, from the ONE shared plan in
        // moe_fused_swiglu_common.hpp. It SHRINKS with the runtime m_eff exactly as every other shape
        // here does, and it is a pure function of (m_eff, KGROUPS, my_row) — the same three numbers on
        // every core and every RISC-V, which is what keeps the column's all-to-all deadlock-free.
        // 0 = an idle core: it still CONTRIBUTES its own partial (the dataflow kernels ship it), it
        // just owns no slice to reduce.
        const uint32_t slice_tiles =
            (SCATTER != 0) ? moe_fused_swiglu::slice_assigned(gu_block_tiles, KGROUPS, my_row) : 0;

        // ---- 3. cross-column reduce + SwiGLU ----
        {
            MaybeDeviceZoneScope("compute_reduce");
            if constexpr (SCATTER) {
                // REDUCE-SCATTER, worker side. Every one of the KGROUPS contributors has pushed its
                // slice of gate and up into slot `row` of my two landing CBs, so the whole reduce is
                // `slice_tiles` wide instead of the block's full m_eff*HN_PAD — that factor IS the
                // win, and it is why the SiLU below is one DEST window instead of m_eff of them.
                //
                // Contributor 0 SEEDS the accumulator with a `copy`, contributors 1..K-2 accumulate
                // IN PLACE, and the LAST one folds in with SiLU riding the PACKER thread. The
                // in-place `add<blk_in(acc), blk_in(in), blk_out(acc)>` is safe for the same reason
                // the tree's is: `eltwise_chain` pops its inputs in `elem_apply_compute` and reserves
                // the output only later in `elem_apply_pack`, so a `slice_pages`-deep accumulator
                // recycles its own pages in ring order. What is NOT safe — and what hung round 1 of
                // this experiment — is letting a SECOND RISC-V push the same CB: `cb_push_back`
                // overwrites the shared `tiles_received` word with the PUSHING RISC-V's own local
                // count, so PACK is the ONLY pusher of cb_slice_gate/cb_slice_up here, and the
                // dataflow kernels are the only pusher of the landing CBs.
                if (slice_tiles) {
                    copy<blk_in(cb_gather_gate), blk_out(cb_slice_gate)>(blk_shape(slice_tiles));
                    for (uint32_t c = 0; c + 2 < KGROUPS; ++c) {
                        add<blk_in(cb_slice_gate), blk_in(cb_gather_gate), blk_out(cb_slice_gate)>(
                            blk_shape(slice_tiles));
                    }
                    // The final gate add, with SiLU on the packer thread. A slice is at most one DEST
                    // window, so the root's m_eff-call bias walk (the helper's bias index does not
                    // advance with in0_subblock, bias_add_helpers.inl:141) collapses to ONE call FOR
                    // FREE — that collapse is a large part of the measured win, not a side effect.
                    gg_buf.wait_front(slice_tiles);
                    for (uint32_t t0 = 0; t0 < slice_tiles; t0 += DEST_LIMIT) {
                        uint32_t w = slice_tiles - t0;
                        if (w > DEST_LIMIT) {
                            w = DEST_LIMIT;
                        }
                        add_bias_bcast_rows<
                            BiasBroadcast::Elementwise,
                            OutputCBLayout::SubblockMajor,
                            bias_add_config::NoPostBias,
                            SiluActivation>(sg_buf, gg_buf, silu_buf, BiasAddShape::of(1, 1, 1, w), {}, t0);
                    }
                    gg_buf.pop_front(slice_tiles);

                    copy<blk_in(cb_gather_up), blk_out(cb_slice_up)>(blk_shape(slice_tiles));
                    for (uint32_t c = 0; c + 1 < KGROUPS; ++c) {
                        add<blk_in(cb_slice_up), blk_in(cb_gather_up), blk_out(cb_slice_up)>(blk_shape(slice_tiles));
                    }

                    // Drain the landing CBs' padding tail so the pop total equals the reader's
                    // WHOLE-CB push. That is not tidiness: the whole-CB push is what returns the
                    // landing write pointer to the CB base on EVERY core every M-block, which is the
                    // contract the contributors' own-write-pointer address proxy stands on.
                    constexpr uint32_t CAP = GATHER_PAGES;
                    const uint32_t live = KGROUPS * slice_tiles;
                    if (CAP > live) {
                        gg_buf.pop_front(CAP - live);
                        gu_buf.pop_front(CAP - live);
                    }
                }
            } else {
                // THE TREE. Walk the reader's invite WAVES (Refinement 2 lever 1): the reader reserves
                // and pushes the WHOLE CB — REDUCE_SLOTS slots — per wave, so this side consumes
                // exactly that much per wave, adding the `wave` slots that carry a child and draining
                // the rest.
                for (uint32_t c0 = 0; c0 < num_children; c0 += REDUCE_SLOTS) {
                    uint32_t wave = num_children - c0;
                    if (wave > REDUCE_SLOTS) {
                        wave = REDUCE_SLOTS;
                    }
                    for (uint32_t c = c0; c < c0 + wave; ++c) {
                        const bool final_child = (c + 1 == num_children);
                        if (is_root && final_child) {
                            // Root's last gate add: SiLU is fused on the PACKER thread, so the
                            // activation overlaps the math thread instead of costing a separate SFPU
                            // pass.
                            //
                            // One call per token tile-row: the helper's bias index does not advance
                            // with in0_subblock (bias_add_helpers.inl:141), so an Elementwise bias
                            // spanning M_BLOCK tile-rows is walked with bias_offset instead, one M-row
                            // per call. The slot arrives WHOLE (see REDUCE_SLOT_TILES) but only its
                            // first m_eff tile-rows carry live tokens, so the bias walk stops at m_eff
                            // and the tail is dropped. THIS is the m_eff-call walk the scatter path
                            // above collapses to one call, and it is ~85 % of this stage's cost.
                            rg_buf.wait_front(REDUCE_SLOT_TILES);
                            for (uint32_t m = 0; m < m_eff; ++m) {
                                add_bias_bcast_rows<
                                    BiasBroadcast::Elementwise,
                                    OutputCBLayout::SubblockMajor,
                                    bias_add_config::NoPostBias,
                                    SiluActivation>(
                                    gate_buf,
                                    rg_buf,
                                    silu_buf,
                                    BiasAddShape::of(1, 1, OUT_SUBBLOCK_H_GU, HN_PAD),
                                    {},
                                    m * HN_PAD);
                            }
                            rg_buf.pop_front(REDUCE_SLOT_TILES);
                        } else {
                            add<blk_in(cb_gate_acc), blk_in(cb_reduce_gate_in), blk_out(cb_gate_acc)>(
                                blk_shape(gu_block_tiles));
                            if (gu_block_tiles < REDUCE_SLOT_TILES) {
                                rg_buf.pop_front(REDUCE_SLOT_TILES - gu_block_tiles);
                            }
                        }
                        add<blk_in(cb_up_acc), blk_in(cb_reduce_up_in), blk_out(cb_up_acc)>(blk_shape(gu_block_tiles));
                        if (gu_block_tiles < REDUCE_SLOT_TILES) {
                            ru_buf.pop_front(REDUCE_SLOT_TILES - gu_block_tiles);
                        }
                    }
                    // Drain the slots of this wave that carried no child, so the pop total matches the
                    // reader's whole-CB push and the write pointer stays block-aligned on every core.
                    if (wave < REDUCE_SLOTS) {
                        const uint32_t idle = (REDUCE_SLOTS - wave) * REDUCE_SLOT_TILES;
                        rg_buf.pop_front(idle);
                        ru_buf.pop_front(idle);
                    }
                }
            }
        }

        {
            MaybeDeviceZoneScope("compute_swiglu");
            if constexpr (SCATTER) {
                // The SwiGLU multiply on MY SLICE ONLY, straight into the CB the writer unicasts out
                // of. Nothing here assembles the column's h block: the workers' finished slices tile
                // the ROOT's cb_h_local as they LAND, so the gather IS the assembly — no landing CB
                // and no root-side copy (measured worth 8.6 % and 52 224 B/core against the version
                // that lands them separately and copies).
                if (slice_tiles) {
                    mul<blk_in(cb_gate_silu), blk_in(cb_slice_up), blk_out(cb_h_slice)>(blk_shape(slice_tiles));
                }
            } else if (is_root) {
                // FPU multiply through L1 (deliberately not SFPU and not DEST-reuse — the L1
                // round-trip measured faster for an FPU consumer, examples/compute_fusion).
                mul<blk_in(cb_gate_silu), blk_in(cb_up_acc), blk_out(cb_h_local)>(blk_shape(gu_block_tiles));
            } else {
                copy<blk_in(cb_gate_acc), blk_out(cb_gate_send)>(blk_shape(gu_block_tiles));
                copy<blk_in(cb_up_acc), blk_out(cb_up_send)>(blk_shape(gu_block_tiles));
            }
        }

        // ---- 4. down matmul: HGROUPS K-blocks, packer L1 accumulation into a caller-owned
        // interm region (so every K-block accumulates at the SAME L1 address) ----
        {
            MaybeDeviceZoneScope("compute_down");
            out_interm_buf.reserve_back(out_block_tiles);
            matmul_block<
                /*transpose=*/false,
                /*packer_l1_acc=*/true,
                LastBlockTarget::Interm,
                OutputCBLayout::TileRowMajor,
                matmul_config::InitMode::Short,
                InputPolicy::WaitAndPopPerKBlock,
                InputPolicy::WaitAndPopPerKBlock,
                NoPostCompute,
                NoPreKBlock,
                NoPostKBlock,
                /*untilize_block_ct_dim=*/0,
                HnSteps,
                NoIn0Source,
                NoIn1BaseOffset,
                /*caller_owns_pack_target=*/true>(
                h_buf,
                wd_buf,
                out_tiles_buf,
                out_interm_buf,
                shape_dn,
                {},
                {},
                /*in1_per_core_w=*/EC_MAX,
                /*out_row_width=*/EC_MAX,
                {},
                HnSteps{hn_last});
            out_interm_buf.push_back(out_block_tiles);
            // matmul_block leaves packer L1 accumulation ENABLED after its last K-block, and neither
            // the eltwise chain (L1Accumulation::Disabled is a compile-time no-op) nor a
            // packer_l1_acc=false matmul resets it. Without this the copy below — and the next
            // M-block's gate matmul — would ACCUMULATE onto stale L1 instead of overwriting.
            pack_reconfig_l1_acc(0);
        }

        {
            // The one genuine dtype boundary: bf16 accumulation -> bfp8 output tiles.
            MaybeDeviceZoneScope("compute_out_pack");
            copy<blk_in(cb_out_interm), blk_out(cb_out_tiles)>(blk_shape(out_block_tiles));
        }

        // The resident x block was retained by both matmuls; release it now.
        x_buf.pop_front(x_slot_tiles);
    }
}
