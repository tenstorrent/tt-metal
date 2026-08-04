// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// moe_fused_swiglu — COMPUTE.
//
// Per M-block, on every one of the HGROUPS x KGROUPS worker cores:
//   1. fused tilize of the row-major x slice this core injects (bf16 activation path only);
//   2. gate and up matmuls over the SAME resident x block, interleaved chunk by chunk over the
//      hidden axis. For a full-size x slot, up consumes each progressively published M row-group,
//      then gate reuses the resident x; compute pops the complete slot once at the end;
//   3. the reduce-scatter + SwiGLU epilogue: every core folds only ITS OWN SLICE over all KGROUPS
//      contributors and runs the epilogue on that slice, so the SFPU SiLU — the dominant cost —
//      is parallelised KGROUPS ways and the bias walk collapses to one call;
//   4. the `down` matmul over HGROUPS phase-2 K-blocks with packer L1 accumulation, then the one
//      genuine dtype boundary (bf16 partials -> bfp8 output).
//
// Everything here is a kernel_lib helper except `fold_dest`, which needs a runtime tile offset that
// compile-time element specs cannot express. The ONE raw access is the L1 mailbox read of the token
// count: the M-block trip count must be identical on all three TRISCs, and `cb_wait_front` in a
// compute kernel is UNPACK-only, so a CB handoff would let MATH and PACK diverge.
//
// The measurement behind every choice here is in perf_experiments/DESIGN_NOTES.md.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#include "moe_fused_swiglu_common.hpp"   // the ONE definition of the mailbox word layout
#include "moe_fused_swiglu_ct_args.hpp"  // the ONE definition of the compile-time arg order

using namespace compute_kernel_lib;

// PER-STAGE ZONES — PERMANENT, always compiled, free with the profiler off. A compute TU does NOT
// get the profiler through `dataflow_api.h` (it must not see the dataflow API at all), which is
// exactly why `perf_instrumentation.hpp` exists. 6 records per M-block here.

MOE_DECLARE_CT_ENUM(MOE_COMPUTE_CT_ARGS);

constexpr uint32_t M_BLOCK = CT(M_BLOCK);
constexpr uint32_t KR_PAD = CT(KR_PAD);
constexpr uint32_t HN_PAD = CT(HN_PAD);
constexpr uint32_t EC_MAX = CT(EC_MAX);  // phase-2 N stride (uniform CB increment)
constexpr uint32_t HGROUPS = CT(HGROUPS);
constexpr uint32_t KGROUPS = CT(KGROUPS);  // column height == contributor count
constexpr uint32_t HID_T = CT(HID_T);
constexpr uint32_t INPUT_FORMAT = CT(INPUT_FORMAT);
constexpr uint32_t OUT_SUBBLOCK_H_GU = CT(OUT_SUBBLOCK_H_GU);
// `down` output sub-block HEIGHT. Separate from the gate/up height because down's width is `ec`
// (2-3) against a DEST budget of 8, while gate/up's is already HN_PAD. Host-derived as the largest
// power of two that fits; taken as min(.., m_eff) below so it never constrains the runtime shrink.
constexpr uint32_t OUT_SUBBLOCK_H_DN = CT(OUT_SUBBLOCK_H_DN);
constexpr uint32_t MAILBOX_MAGIC = CT(MAILBOX_MAGIC);
// Smallest legal `m_eff` (= OUT_SUBBLOCK_H_GU rounded up to a power of two, so `m_eff /
// OUT_SUBBLOCK_H_GU` is always exact). One host definition, identical in all three kernels.
constexpr uint32_t M_EFF_MIN = CT(M_EFF_MIN);
// gate/up in1 sub-block WIDTH in hidden tiles — a sub-division of one chunk.
constexpr uint32_t HN_BLOCK = CT(HN_BLOCK);
// The hidden-axis chunk the gate/up weight stream is published and consumed in, so the matmul on
// chunk c overlaps the DRAM read of c+1. Host-guaranteed to divide HN_PAD.
constexpr uint32_t GU_CHUNKS = CT(GU_CHUNKS);
constexpr uint32_t GU_CHUNK_W = HN_PAD / GU_CHUNKS;
// Eltwise DEST-window block size: tiles per acquire/commit/wait/release cycle.
constexpr uint32_t ELTWISE_BLK = CT(ELTWISE_BLK);
constexpr uint32_t DEST_LIMIT = CT(DEST_LIMIT);
constexpr uint32_t GATHER_PAGES = CT(GATHER_PAGES);  // the WHOLE landing CB, in tiles

constexpr uint32_t cb_x_in = CT(CB_X_IN);
constexpr uint32_t cb_x_tiles = CT(CB_X_TILES);
constexpr uint32_t cb_x_stage = CT(CB_X_STAGE);
constexpr uint32_t cb_w_gate = CT(CB_W_GATE);
constexpr uint32_t cb_w_up = CT(CB_W_UP);
constexpr uint32_t cb_w_down = CT(CB_W_DOWN);
constexpr uint32_t cb_gate_acc = CT(CB_GATE_ACC);
constexpr uint32_t cb_up_acc = CT(CB_UP_ACC);
constexpr uint32_t cb_gate_silu = CT(CB_GATE_SILU);
constexpr uint32_t cb_h_local = CT(CB_H_LOCAL);
constexpr uint32_t cb_h = CT(CB_H);
constexpr uint32_t cb_out_interm = CT(CB_OUT_INTERM);
constexpr uint32_t cb_out_tiles = CT(CB_OUT_TILES);
constexpr uint32_t cb_gather_gate = CT(CB_GATHER_GATE);
constexpr uint32_t cb_gather_up = CT(CB_GATHER_UP);
constexpr uint32_t cb_slice_gate = CT(CB_SLICE_GATE);
constexpr uint32_t cb_slice_up = CT(CB_SLICE_UP);
constexpr uint32_t cb_h_slice = CT(CB_H_SLICE);

constexpr uint32_t TILE_H = 32;

// ---------------------------------------------------------------------------
// BLOCKED ELTWISE PASSES. `input(cb)`/`output(cb)` default to per-TILE lifecycles, and
// `eltwise_chain` SILENTLY clamps `block_size` to 1 unless every CB uses a compatible policy — so
// the convenient spelling costs one full DEST sync round trip PER TILE against a budget of 8.
// These specs opt into the chunked lifecycle; `OperandKind::Block` is required, not decorative
// (getting it wrong is a compile error, not a wrong answer). See DESIGN_NOTES.md §7.
constexpr auto blk_in(uint32_t cb) { return input(cb, WaitPolicy::PerChunk, PopPolicy::PerChunk, OperandKind::Block); }
constexpr auto blk_out(uint32_t cb) { return output(cb, ReservePolicy::PerChunk, PushPolicy::PerChunk); }
ALWI auto blk_shape(uint32_t n) { return EltwiseShape::tiles(n, ELTWISE_BLK); }

// `dest_acc` — the running sum lives in DEST for the WHOLE fold, packed to L1 exactly ONCE.
// `add_tiles_init(IN, IN, acc_to_dest=true)` makes `add_tiles` compute `DEST[dst] += A + B`, so one
// call folds TWO contributors with no repack between. Worth nc-1 packs, nc-1 re-reads and nc-2
// inits per DEST window, at no extra L1. See DESIGN_NOTES.md §7.
//
// RAW compute API on purpose: `eltwise_chain` would need one element per contributor reading the
// same CB at a RUNTIME tile offset, and element specs are compile-time.
template <uint32_t ACC, uint32_t IN>
ALWI void fold_dest(uint32_t nc, uint32_t n) {
    cb_wait_front(IN, nc * n);
    cb_reserve_back(ACC, n);
    // RECONFIG BEFORE INIT, and BOTH sides. The `*_init` calls below set the math MOP; they do NOT
    // set the unpacker's DATA FORMAT registers, which still hold whatever the gate/up matmul left
    // (cb_w_gate / cb_x_tiles). Omitting the srcA/srcB reconfig produced `inf` at pcc = 1.000000 —
    // the right pattern read through the wrong format — and it did so even with every accumulate
    // removed, i.e. it is the raw block's format state, not the accumulation.
    reconfig_data_format(IN, IN);
    pack_reconfig_data_format(ACC);
    for (uint32_t t0 = 0; t0 < n; t0 += ELTWISE_BLK) {
        uint32_t w = n - t0;
        if (w > ELTWISE_BLK) {
            w = ELTWISE_BLK;
        }
        tile_regs_acquire();
        // SEED, always — never accumulate onto DEST's entry state. Measured, not defensive:
        // relying on "DEST is zero after acquire" produced `inf` at pcc = 1.000000. Parity decides
        // the seed WIDTH so the remainder pairs up exactly: odd nc -> one `copy_tile`, even nc ->
        // one non-accumulating `add_tiles`. Both OVERWRITE DEST.
        uint32_t c;
        if (nc & 1u) {
            copy_tile_to_dst_init_short(IN);
            for (uint32_t i = 0; i < w; ++i) {
                copy_tile(IN, t0 + i, i);
            }
            c = 1;
        } else {
            add_tiles_init(IN, IN, /*acc_to_dest=*/false);
            for (uint32_t i = 0; i < w; ++i) {
                add_tiles(IN, IN, t0 + i, n + t0 + i, i);
            }
            c = 2;
        }
        // ...and the rest, TWO contributors per call, straight into the sticky DEST accumulator.
        // Slice `c` starts at tile `c * n` of the landing CB.
        add_tiles_init(IN, IN, /*acc_to_dest=*/true);
        for (; c + 1 < nc; c += 2) {
            for (uint32_t i = 0; i < w; ++i) {
                add_tiles(IN, IN, c * n + t0 + i, (c + 1) * n + t0 + i, i);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < w; ++i) {
            pack_tile(i, ACC);  // THE only pack; in order, so the CB write pointer walks it
        }
        tile_regs_release();
    }
    // Clear the latched `acc_to_dest`. It is a STICKY math-config bit, and the helpers that run next
    // may use a short init that does not re-assert it — in which case every later FPU op would keep
    // folding into DEST instead of overwriting it.
    add_tiles_init(IN, IN, /*acc_to_dest=*/false);
    // Hand the machine back in the state the FOLLOWING helper chain expects. That chain is
    // `add<blk_in(ACC), blk_in(IN), blk_out(ACC)>`, and its own reconfig is compile-time-elided
    // against a static CB sequence this raw block is invisible to — so the hardware must already
    // match, whether or not the chain re-emits.
    reconfig_data_format(ACC, IN);
    cb_pop_front(IN, nc * n);
    cb_push_back(ACC, n);
}

// Fold `nc` contributors of `IN` into `ACC`. All `nc` accumulate in DEST behind ONE pack; see
// `fold_dest` for why that is worth 2.6-4.0 % and for the reconfig-before-init trap it hides.
template <uint32_t ACC, uint32_t IN>
ALWI void fold_chain(uint32_t nc, uint32_t n) {
    fold_dest<ACC, IN>(nc, n);
}

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
    const uint32_t my_col = get_arg_val<uint32_t>(4);  // grid column == this core's x-injection slot
    const uint32_t my_row = get_arg_val<uint32_t>(5);  // row in the column == which scatter slice I own

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_tiles, cb_w_gate, cb_gate_acc);
    // SiLU rides the packer thread of the root's final reduce add; the helpers never issue this.
    ActivationInitHelper<KernelActivation::SILU>::init();

    // Device-resident token count. All three TRISCs spin here independently so the M-block trip
    // count is thread-uniform (see the file header). A plain fence stands in for
    // `invalidate_l1_cache()` — identical on Blackhole, but that helper is dataflow-only.
    const auto mb = moe_fused_swiglu::mailbox_wait(mailbox_addr, MAILBOX_MAGIC, [] {
        asm volatile("fence" ::: "memory");
    });
    const uint32_t m_t = mb.m_t;
    const uint32_t m_blocks = mb.m_blocks;

    CircularBuffer x_buf(cb_x_tiles);
    CircularBuffer wg_buf(cb_w_gate);
    CircularBuffer wu_buf(cb_w_up);
    CircularBuffer wd_buf(cb_w_down);
    CircularBuffer gate_buf(cb_gate_acc);
    CircularBuffer up_buf(cb_up_acc);
    CircularBuffer silu_buf(cb_gate_silu);
    // The reduce-scatter's landing + slice buffers. A `CircularBuffer` only wraps the index — no
    // L1 access at construction.
    CircularBuffer gg_buf(cb_gather_gate);
    CircularBuffer gu_buf(cb_gather_up);
    CircularBuffer sg_buf(cb_slice_gate);
    CircularBuffer h_buf(cb_h);
    CircularBuffer out_interm_buf(cb_out_interm);
    CircularBuffer out_tiles_buf(cb_out_tiles);

    // One contributor's slot in the landing CBs. Always pushed WHOLE, which is what returns every
    // core's write pointer to the CB base and lets a contributor use its OWN pointer as the
    // destination address.
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

        // PAGE ACCOUNTING vs ARITHMETIC. Everything above is a PAGE count and stays on `m_eff`,
        // because those are what must divide M_BLOCK and agree across cores without communication.
        // `m_rows` is the ARITHMETIC count — the rows the GATE/UP matmul actually produces. It is
        // <= m_eff, and strictly less exactly when this is a tail block whose remainder is not a
        // power of two: m_t 5 computes 5 rows of an 8-row block, m_t 3 computes 3 of 4.
        //
        // Only gate/up may use it, and that is a property of the HELPER's input policy rather than a
        // choice: both calls use the retaining `WaitAndRetainPerMSubblock` lifecycle, with its
        // runtime shape bit selecting a progressive prefix wait or one bulk wait. Neither pops
        // because gate/up has num_k_blocks == 1. This kernel pops x itself at `x_slot_tiles`
        // (= m_eff * KR_PAD) below. `down` has no such freedom — see shape_dn.
        const uint32_t m_rows = moe_fused_swiglu::m_tiles_real(m_t, b, M_BLOCK);

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
        // `down` STAYS ON m_eff — the `m_rows` shrink deliberately does not reach it. Its in0 is `h` under
        // `InputPolicy::WaitAndPopPerKBlock`, and the helper derives BOTH the wait and the pop from
        // the shape (`in0_block_num_tiles = in0_subblock_num_tiles * shape.in0_num_subblocks`,
        // matmul_block_helpers.inl:196). Shrinking the shape shrinks the pop, so cb_h_local drifts by
        // (m_eff - m_rows) * HN_PAD tiles per K-block and the op HANGS — measured at M 192.
        //
        // The escape hatch is in0_policy = NoWaitNoPop with a caller-owned wait/pop at m_eff, but
        // that means ONE wait for the whole h block instead of one per K-block, which serialises the
        // column gather against the matmul it currently overlaps. That trade is the opposite of the
        // one this knob is trying to make, so `down` is left alone.
        const MatmulBlockShape shape_dn = MatmulBlockShape::of(m_eff / sbh_dn, 1, sbh_dn, ec, HN_PAD, HGROUPS);

        // ---- 1. fused tilize of the x tile-rows this core injects (bf16 ROW_MAJOR only) ----
        if constexpr (INPUT_FORMAT == 0) {
            MaybeDeviceZoneScope("compute_tilize");
            const uint32_t n_inject =
                moe_fused_swiglu::inject_rows(m_eff, moe_fused_swiglu::inject_first(my_col), HGROUPS);
            for (uint32_t i = 0; i < n_inject; ++i) {
                // Asymmetric page mode: TILE_H row-major stick slices in -> KR_PAD bfp8 tiles out.
                tilize<KR_PAD, cb_x_in, cb_x_stage>(1, TILE_H);
            }
        }

        // ---- 2. gate and up over the same resident x block ----
        // The two matmuls walk the HIDDEN axis in GU_CHUNKS chunks, interleaved per chunk. Full-size
        // M slots run up then gate because W_up is available during x multicast; smaller slots
        // retain gate then up. `out_col_offset` + `caller_owns_pack_target` keep the layout m-major.
        {
            MaybeDeviceZoneScope("compute_gateup");
            gate_buf.reserve_back(gu_block_tiles);
            up_buf.reserve_back(gu_block_tiles);
            // The ONE reconfig this phase needs, hoisted out of the loop (see the calls below).
            reconfig_data_format(cb_w_gate, cb_x_tiles);
            pack_reconfig_data_format(cb_gate_acc);
            for (uint32_t c = 0; c < GU_CHUNKS; ++c) {
                // The ragged column (hn < HN_PAD) narrows the FMA width of the chunk it falls in;
                // the host guarantees every chunk keeps at least one real column. 0 means "full".
                const uint32_t h0 = c * GU_CHUNK_W;
                const uint32_t valid = (hn > h0) ? (hn - h0) : 0;
                if (valid == 0) {
                    // This chunk is entirely PAD on the ragged column. The dataflow kernels still
                    // push it (unread) so cb_w_gate/cb_w_up's residency wrap is core-independent, so
                    // consume it here without a matmul. The output columns it leaves untouched are
                    // pad, and `down`'s HnSteps narrows the last K-block past them.
                    constexpr uint32_t CHUNK_W_TILES = KR_PAD * GU_CHUNK_W;
                    wg_buf.wait_front(CHUNK_W_TILES);
                    wg_buf.pop_front(CHUNK_W_TILES);
                    wu_buf.wait_front(CHUNK_W_TILES);
                    wu_buf.pop_front(CHUNK_W_TILES);
                    continue;
                }
                MatmulBlockShape shape_c = MatmulBlockShape::of(
                    moe_fused_swiglu::round_up_capped(m_rows, OUT_SUBBLOCK_H_GU, m_eff) / OUT_SUBBLOCK_H_GU,
                    GU_IN1_SUBBLOCKS,
                    OUT_SUBBLOCK_H_GU,
                    HN_BLOCK,
                    KR_PAD,
                    1);
                shape_c.last_in1_subblock_w_valid =
                    (valid < GU_CHUNK_W) ? (valid - (GU_IN1_SUBBLOCKS - 1) * HN_BLOCK) : 0;

                // One two-pass body keeps M runtime without cloning the helper instantiation.
                // Full-size M slots run up first with progressive waits; smaller slots retain the
                // original gate-first order and bulk wait because their short multicast cannot
                // repay the extra CB bookkeeping.
                const bool stream_m = (m_eff == M_BLOCK);
                for (uint32_t pass = 0; pass < 2; ++pass) {
                    const bool run_up = stream_m ? (pass == 0) : (pass != 0);
                    CircularBuffer& weight_buf = run_up ? wu_buf : wg_buf;
                    CircularBuffer& accum_buf = run_up ? up_buf : gate_buf;
                    shape_c.wait_in0_per_m_subblock = stream_m && (pass == 0);
                    matmul_block<
                        /*transpose=*/false,
                        /*packer_l1_acc=*/true,
                        LastBlockTarget::Interm,
                        OutputCBLayout::TileRowMajor,
                        matmul_config::InitMode::Short,
                        InputPolicy::WaitAndRetainPerMSubblock,
                        InputPolicy::WaitAndPopPerKBlock,
                        NoPostCompute,
                        NoPreKBlock,
                        NoPostKBlock,
                        /*untilize_block_ct_dim=*/0,
                        KrSteps,
                        NoIn0Source,
                        NoIn1BaseOffset,
                        /*caller_owns_pack_target=*/true,
                        NoneActivation,
                        // Every gate/up call has identical operand formats; the phase-level
                        // reconfig above is sufficient for either order. Measured at up to 1.19x.
                        matmul_config::DataFormatReconfig::NONE>(
                        x_buf,
                        weight_buf,
                        accum_buf,
                        accum_buf,
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
            }
            gate_buf.push_back(gu_block_tiles);
            up_buf.push_back(gu_block_tiles);
            // packer_l1_acc leaves L1 accumulation ENABLED after the last chunk (the `down` matmul
            // below carries the same note). The reduce chain that follows would otherwise ACCUMULATE
            // onto stale L1 instead of overwriting.
            pack_reconfig_l1_acc(0);
        }

        // MY SLICE of this block's m_eff*HN_PAD tiles, from the ONE shared plan in
        // moe_fused_swiglu_common.hpp — a pure function of (m_eff, KGROUPS, my_row), the same
        // three numbers on every core and every RISC-V, which is what keeps the all-to-all
        // deadlock-free. 0 = an idle core: it still contributes its partial, it just owns no
        // slice to reduce.
        const uint32_t slice_tiles = moe_fused_swiglu::slice_assigned(gu_block_tiles, KGROUPS, my_row);

        // ---- 3. cross-column reduce + SwiGLU ----
        {
            MaybeDeviceZoneScope("compute_reduce");
            // REDUCE-SCATTER, worker side. Every KGROUPS contributor has pushed its slice into
            // slot `row` of my two landing CBs, so the reduce is `slice_tiles` wide instead of the
            // whole block — that factor IS the win. PACK must be the ONLY pusher of the slice CBs:
            // `cb_push_back` writes the shared `tiles_received` word with the pushing RISC-V's own
            // count, and a second pusher hung round 1 of this experiment.
            if (slice_tiles) {
                // KGROUPS-1 contributors fold here; the last one rides the SiLU-fused add below.
                fold_chain<cb_slice_gate, cb_gather_gate>(KGROUPS - 1, slice_tiles);
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

                fold_chain<cb_slice_up, cb_gather_up>(KGROUPS, slice_tiles);

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
        }

        {
            MaybeDeviceZoneScope("compute_swiglu");
            // The SwiGLU multiply on MY SLICE ONLY, straight into the CB the writer unicasts out
            // of. Nothing here assembles the column's h block: the workers' finished slices tile
            // the ROOT's cb_h_local as they LAND, so the gather IS the assembly — no landing CB
            // and no root-side copy (measured worth 8.6 % and 52 224 B/core against the version
            // that lands them separately and copies).
            if (slice_tiles) {
                // Inherits phase 1's hoisted cb_gate_acc pack format, which is correct exactly
                // because cb_h_slice is bfp8 — the epilogue's single dtype boundary.
                mul<blk_in(cb_gate_silu), blk_in(cb_slice_up), blk_out(cb_h_slice)>(blk_shape(slice_tiles));
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

        // The resident x block was retained by both matmuls; release it now. A ragged full-size
        // slot (for example m_rows=7, m_eff=8) deliberately computes only the real prefix, so
        // explicitly front the final padding-row publication before popping the whole reservation.
        // Exact full blocks already fronted the last row in the progressive matmul; smaller slots
        // are published atomically, so neither needs another wait.
        if (m_eff == M_BLOCK && m_rows != m_eff) {
            x_buf.wait_front(x_slot_tiles);
        }
        x_buf.pop_front(x_slot_tiles);
    }
}
