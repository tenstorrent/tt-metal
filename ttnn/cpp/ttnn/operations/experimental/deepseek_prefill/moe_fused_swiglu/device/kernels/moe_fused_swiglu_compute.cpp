// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// moe_fused_swiglu — COMPUTE.
//
// Per M-block, on every one of the HGROUPS x KGROUPS worker cores:
//   1. fused tilize of the row-major x slice this core injects directly into the resident x slot
//      (bf16 activation path only); a one-page CB signals completion to the reader without making
//      compute a second producer of the resident CB;
//   2. gate and up matmuls over the SAME resident x block, interleaved chunk by chunk over the
//      hidden axis. For a full-size x slot, up consumes each progressively published M row-group,
//      then gate reuses the resident x; compute pops the complete slot once at the end;
//   3. the reduce-scatter + SwiGLU epilogue: every core folds only ITS OWN SLICE over all KGROUPS
//      contributors and runs the epilogue on that slice, so the SFPU SiLU — the dominant cost —
//      is parallelised KGROUPS ways and the bias walk collapses to one call;
//   4. the `down` matmul. Full-size runtime M slots use M_BLOCK independent M-row calls against one
//      complete resident weight shard; ragged blocks retain the HGROUPS-K-block accumulation path.
//      Both finish at the one genuine dtype boundary (bf16 partials -> bfp8 output).
//
// The operation-local helpers below cover the fixed matmul, SiLU, and tilize paths. `fold_dest`
// needs a runtime tile offset that compile-time element specs cannot express. The ONE raw access is the L1 mailbox read
// of the token count: the M-block trip count must be identical on all three TRISCs, and `cb_wait_front` in a compute
// kernel is UNPACK-only, so a CB handoff would let MATH and PACK diverge.
//
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#ifdef SITU_GLU
#include "api/compute/situ_glu.h"
#endif
#include "moe_fused_swiglu_compute_helpers.hpp"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"

#include "moe_fused_swiglu_common.hpp"   // the ONE definition of the mailbox word layout
#include "moe_fused_swiglu_ct_args.hpp"  // the ONE definition of the compile-time arg order

using namespace moe_fused_swiglu::compute;

#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)

// PER-STAGE ZONES — PERMANENT, always compiled, free with the profiler off. A compute TU does NOT
// get the profiler through `dataflow_api.h` (it must not see the dataflow API at all), which is
// exactly why `perf_instrumentation.hpp` exists. 6 records per M-block here.

MOE_DECLARE_CT_ENUM(MOE_COMPUTE_CT_ARGS);

constexpr uint32_t M_BLOCK = CT(M_BLOCK);
constexpr uint32_t KR_PAD = CT(KR_PAD);
constexpr uint32_t HN_PAD = CT(HN_PAD);
constexpr uint32_t EC_MAX = CT(EC_MAX);  // phase-2 N stride (uniform CB increment)
constexpr uint32_t WD_EC_MAX = CT(WD_EC_MAX);
constexpr uint32_t EC_GROUP_MAX = CT(EC_GROUP_MAX);
constexpr uint32_t HGROUPS = CT(HGROUPS);
constexpr uint32_t KGROUPS = CT(KGROUPS);  // column height == contributor count
constexpr uint32_t HID_T = CT(HID_T);
constexpr uint32_t INPUT_FORMAT = CT(INPUT_FORMAT);
constexpr uint32_t OUT_SUBBLOCK_H_GU = CT(OUT_SUBBLOCK_H_GU);
// Uniform-safe `down` output sub-block HEIGHT, derived against EC_MAX. Narrower cores may grow it
// further at runtime against their real `ec`; both choices stay powers of two dividing m_eff.
constexpr uint32_t OUT_SUBBLOCK_H_DN = CT(OUT_SUBBLOCK_H_DN);
constexpr uint32_t OUT_SUBBLOCK_H_DN_MAX = CT(OUT_SUBBLOCK_H_DN_MAX);
constexpr uint32_t MAILBOX_MAGIC = CT(MAILBOX_MAGIC);
// Smallest legal `m_eff` (= OUT_SUBBLOCK_H_GU rounded up to a power of two, so `m_eff /
// OUT_SUBBLOCK_H_GU` is always exact). One host definition, identical in all three kernels.
constexpr uint32_t M_EFF_MIN = CT(M_EFF_MIN);
constexpr uint32_t DEPTH_X = CT(DEPTH_X);
// gate/up in1 sub-block WIDTH in hidden tiles — a sub-division of one chunk.
constexpr uint32_t HN_BLOCK = CT(HN_BLOCK);
constexpr uint32_t WD_MROW_ROUNDS = CT(WD_MROW_ROUNDS);
constexpr uint32_t WD_MGROUPS = CT(WD_MGROUPS);
constexpr uint32_t WD_MGROUP_MIN_BLOCKS = CT(WD_MGROUP_MIN_BLOCKS);
constexpr uint32_t MGROUP_ROWS = CT(MGROUP_ROWS);
constexpr uint32_t WD_RESIDENT = CT(WD_RESIDENT);
constexpr bool WD_PACKED = WD_RESIDENT && moe_fused_swiglu::hidden_blocks_are_balanced(HID_T, HGROUPS, HN_PAD);
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
constexpr uint32_t cb_tilize_done = CT(CB_X_STAGE);  // the compute->reader per-row completion edge
constexpr uint32_t cb_mailbox_compute = CT(CB_MAILBOX_COMPUTE);
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

// The running sum stays in DEST for the WHOLE fold, packed to L1 once: `acc_to_dest` makes one
// `add_tiles` fold two contributors. Raw compute API because `eltwise_chain` element specs are
// compile-time and this reads one CB at a RUNTIME tile offset.
template <uint32_t ACC, uint32_t IN>
ALWI void fold_dest(uint32_t num_contributors, uint32_t n) {
    cb_wait_front(IN, num_contributors * n);
    cb_reserve_back(ACC, n);
    // RECONFIG BEFORE INIT, BOTH sides: `*_init` sets the math MOP but NOT the unpacker's format
    // registers, which still hold the gate/up matmul's operands. Without this the tiles are decoded
    // through the wrong format — the right bit pattern, the wrong exponents.
    reconfig_data_format(IN, IN);
    pack_reconfig_data_format(ACC);
    for (uint32_t t0 = 0; t0 < n; t0 += ELTWISE_BLK) {
        uint32_t w = n - t0;
        if (w > ELTWISE_BLK) {
            w = ELTWISE_BLK;
        }
        tile_regs_acquire();
        // SEED, always: `tile_regs_acquire()` does not zero DEST, so accumulating onto its entry
        // state folds in whatever the previous window left. Parity picks the seed WIDTH so the
        // remainder pairs up exactly; both spellings OVERWRITE rather than accumulate.
        uint32_t c;
        if (num_contributors & 1u) {
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
        for (; c + 1 < num_contributors; c += 2) {
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
    cb_pop_front(IN, num_contributors * n);
    cb_push_back(ACC, n);
}

// Fold `num_contributors` contributors of `IN` into `ACC`, all accumulating in DEST behind ONE pack. See
// `fold_dest` for the reconfig-before-init requirement this hides.
template <uint32_t ACC, uint32_t IN>
ALWI void fold_chain(uint32_t num_contributors, uint32_t n) {
    fold_dest<ACC, IN>(num_contributors, n);
}

// The only eltwise_convenience use in this kernel: a blocked, FIFO-preserving
// elementwise multiply.  Keeping it here avoids carrying the generic eltwise
// chain library (and its unrelated operation catalogue) onto origin/main.
template <uint32_t A, uint32_t B, uint32_t OUT>
ALWI void mul_blocked(uint32_t n) {
    reconfig_data_format(A, B);
    pack_reconfig_data_format(OUT);
    mul_tiles_init(A, B);
    cb_wait_front(A, n);
    cb_wait_front(B, n);
    cb_reserve_back(OUT, n);
    for (uint32_t base = 0; base < n; base += ELTWISE_BLK) {
        uint32_t width = n - base;
        if (width > ELTWISE_BLK) {
            width = ELTWISE_BLK;
        }
        tile_regs_acquire();
        for (uint32_t i = 0; i < width; ++i) {
            mul_tiles(A, B, base + i, base + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < width; ++i) {
            pack_tile(i, OUT);
        }
        tile_regs_release();
    }
    cb_pop_front(A, n);
    cb_pop_front(B, n);
    cb_push_back(OUT, n);
}

#ifdef SITU_GLU
// Reduce both halves and apply SiTU while their sums are still in DEST. Four gate and four up
// outputs fill the eight-tile window, eliminating the old pack-to-L1 + reload boundary.
template <uint32_t GATE, uint32_t UP, uint32_t OUT>
ALWI void fold_situ_glu_blocked(uint32_t num_contributors, uint32_t n) {
    constexpr uint32_t OUTPUTS_PER_WINDOW = DEST_LIMIT / 2;
    pack_reconfig_data_format(OUT);
    cb_wait_front(GATE, num_contributors * n);
    cb_wait_front(UP, num_contributors * n);
    cb_reserve_back(OUT, n);
    for (uint32_t base = 0; base < n; base += OUTPUTS_PER_WINDOW) {
        uint32_t width = n - base;
        if (width > OUTPUTS_PER_WINDOW) {
            width = OUTPUTS_PER_WINDOW;
        }
        tile_regs_acquire();

        reconfig_data_format(GATE, GATE);
        uint32_t c;
        if (num_contributors & 1u) {
            copy_tile_to_dst_init_short(GATE);
            for (uint32_t i = 0; i < width; ++i) {
                copy_tile(GATE, base + i, i);
            }
            c = 1;
        } else {
            add_tiles_init(GATE, GATE, /*acc_to_dest=*/false);
            for (uint32_t i = 0; i < width; ++i) {
                add_tiles(GATE, GATE, base + i, n + base + i, i);
            }
            c = 2;
        }
        add_tiles_init(GATE, GATE, /*acc_to_dest=*/true);
        for (; c + 1 < num_contributors; c += 2) {
            for (uint32_t i = 0; i < width; ++i) {
                add_tiles(GATE, GATE, c * n + base + i, (c + 1) * n + base + i, i);
            }
        }

        reconfig_data_format(UP, UP);
        if (num_contributors & 1u) {
            copy_tile_to_dst_init_short(UP);
            for (uint32_t i = 0; i < width; ++i) {
                copy_tile(UP, base + i, width + i);
            }
            c = 1;
        } else {
            add_tiles_init(UP, UP, /*acc_to_dest=*/false);
            for (uint32_t i = 0; i < width; ++i) {
                add_tiles(UP, UP, base + i, n + base + i, width + i);
            }
            c = 2;
        }
        add_tiles_init(UP, UP, /*acc_to_dest=*/true);
        for (; c + 1 < num_contributors; c += 2) {
            for (uint32_t i = 0; i < width; ++i) {
                add_tiles(UP, UP, c * n + base + i, (c + 1) * n + base + i, width + i);
            }
        }
        add_tiles_init(UP, UP, /*acc_to_dest=*/false);

        for (uint32_t i = 0; i < width; ++i) {
            situ_glu_tile(i, width + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < width; ++i) {
            pack_tile(i, OUT);
        }
        tile_regs_release();
    }
    cb_pop_front(GATE, num_contributors * n);
    cb_pop_front(UP, num_contributors * n);
    cb_push_back(OUT, n);
}
#endif

// Per-K-block FMA step count for the gate/up matmul: the padded K slot is KR_PAD tiles wide but
// only `kr_rows` of them are real, so the loop bound shrinks and the pad tiles are never touched.
struct KrSteps {
    uint32_t kr_rows;
    ALWI uint32_t operator()(uint32_t, uint32_t) const { return kr_rows; }
};

// Per-K-block FMA step count for the `down` matmul. Uniform-start grids narrow only their last
// block; balanced grids may narrow several blocks, but the fixed HN_PAD CB stride is unchanged.
struct HnSteps {
    ALWI uint32_t operator()(uint32_t hn_block, uint32_t) const {
        return moe_fused_swiglu::hidden_block_rows(hn_block, HID_T, HGROUPS, HN_PAD);
    }
};

struct PackedWdOffset {
    ALWI uint32_t operator()(uint32_t hn_block) const {
        return moe_fused_swiglu::hidden_block_start(hn_block, HID_T, HGROUPS, HN_PAD) * WD_EC_MAX;
    }
};

void kernel_main() {
    (void)get_arg_val<uint32_t>(0);  // retained runtime slot for cache-compatible argument layout
    const uint32_t kr_rows = get_arg_val<uint32_t>(1);
    const uint32_t hn_cols = get_arg_val<uint32_t>(2);
    const uint32_t ec = get_arg_val<uint32_t>(3);
    const uint32_t ec_group = get_arg_val<uint32_t>(4);
    const uint32_t my_col = get_arg_val<uint32_t>(5);  // grid column == this core's x-injection slot
    const uint32_t my_row = get_arg_val<uint32_t>(6);  // row in the column == which scatter slice I own

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_tiles, cb_w_gate, cb_gate_acc);
    // The activation is compile-time selected and therefore gets its own cached program.
#ifdef SITU_GLU
    situ_glu_tile_init();
#else
    // SiLU rides the packer thread of the root's final reduce add.
    silu_tile_init_pack();
#endif

    CircularBuffer tilize_done(cb_tilize_done);
    CircularBuffer mailbox_compute(cb_mailbox_compute);
    // UNPACK waits on the reader's program-local mailbox publication, then
    // broadcasts the two scalar loop bounds to MATH and PACK through the
    // hardware inter-TRISC mailbox. A CB wait is UNPACK-only, so the explicit
    // broadcast is what keeps all three threads on the same trip count.
    uint32_t m_t = 0;
    uint32_t m_blocks = 0;
    UNPACK(({
        mailbox_compute.wait_front(1);
        const uint32_t mailbox_addr = get_local_cb_interface(cb_mailbox_compute).fifo_rd_ptr << 4;
        const auto mb = moe_fused_swiglu::mailbox_wait(mailbox_addr, MAILBOX_MAGIC, [] {
            asm volatile("fence" ::: "memory");
        });
        m_t = mb.m_t;
        m_blocks = mb.m_blocks;
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, m_t);
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, m_blocks);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, m_t);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, m_blocks);
        mailbox_compute.pop_front(1);
    }));
    MATH(({
        m_t = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
        m_blocks = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
    }));
    PACK(({
        m_t = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
        m_blocks = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
    }));
    const bool wd_mgroup = WD_MGROUPS && (m_blocks >= WD_MGROUP_MIN_BLOCKS) && (m_t != 0) && ((m_t % M_BLOCK) == 0);

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

    for (uint32_t block_idx = 0; block_idx < m_blocks; ++block_idx) {
        // The RUNTIME token tile-rows this block works on — the same number the reader uses for its
        // x-multicast rounds and the writer for its CB waits (moe_fused_swiglu_common.hpp). Every
        // shape and trip count below is derived from it, so count 128 does HALF the gate/up matmul,
        // reduce and `down` work of count 256 instead of the same amount.
        const uint32_t m_eff = moe_fused_swiglu::m_tiles_eff(m_t, block_idx, M_BLOCK, M_EFF_MIN);
        const bool wd_mrow = WD_MROW_ROUNDS && (m_eff == M_BLOCK);
        const uint32_t x_slot_tiles = m_eff * KR_PAD;
        const uint32_t gu_block_tiles = m_eff * HN_PAD;
        const uint32_t down_ec = wd_mgroup ? ec_group : ec;
        const uint32_t down_rows = wd_mgroup ? MGROUP_ROWS : m_eff;
        const uint32_t out_ec_max = wd_mgroup ? EC_GROUP_MAX : EC_MAX;
        const uint32_t out_block_tiles = down_rows * out_ec_max;

        // PAGE vs ARITHMETIC. Everything above is a PAGE count on `m_eff` — those must divide
        // M_BLOCK and agree across cores. `m_rows` is the ARITHMETIC count the gate/up matmul really
        // produces, smaller only on a tail block whose remainder is not a power of two.
        //
        // GATE/UP ONLY, and that is the helper's policy rather than a choice: both calls retain via
        // WaitAndRetainPerMSubblock and neither pops (num_k_blocks == 1), so this kernel pops x
        // itself below. `down` has no such freedom — see shape_dn.
        const uint32_t m_rows = moe_fused_swiglu::m_tiles_real(m_t, block_idx, M_BLOCK);

        // gate/up: [m_eff, HN_PAD] = x[m_eff, kr_rows] @ W[kr_rows, HN_PAD]. ONE K-block whose width is the
        // whole per-row K extent, which is what lets both matmuls read the same resident in0.
        // The in1 sub-blocking sits WITHIN one N-chunk; the host keeps HN_BLOCK a divisor of
        // GU_CHUNK_W, so GU_CHUNKS == 1 degenerates to one sub-block over the whole HN_PAD.
        constexpr uint32_t GU_IN1_SUBBLOCKS = GU_CHUNK_W / HN_BLOCK;

        // down: [m_eff, ec] = h[m_eff, HGROUPS*HN_PAD] @ W_down[.., ec], HGROUPS K-blocks.
        // The FMA width is the real `ec`; the in1 read stride and output row stride stay the uniform
        // EC_MAX so every phase-2 CB increment is core-independent. Narrower cores then take more of
        // DEST (ec=3 stays 2x3, ec=2 grows to 4x2), always at power-of-two heights.
        uint32_t sbh_dn = (OUT_SUBBLOCK_H_DN < m_eff) ? OUT_SUBBLOCK_H_DN : m_eff;
        while (sbh_dn * 2 <= OUT_SUBBLOCK_H_DN_MAX && sbh_dn * 2 <= m_eff && sbh_dn * 2 * ec <= DEST_LIMIT) {
            sbh_dn *= 2;
        }
        // `down` STAYS ON m_eff: WaitAndPopPerKBlock derives the POP from the shape, so shrinking it
        // to m_rows drifts cb_h_local by (m_eff - m_rows) * HN_PAD per K-block and HANGS (seen at
        // M 192). The NoWaitNoPop escape hatch would serialise the column gather against the matmul.
        const MatmulShape shape_dn = MatmulShape::of(m_eff / sbh_dn, 1, sbh_dn, ec, HN_PAD, HGROUPS);

        // ---- 1. fused tilize of the x tile-rows this core injects (bf16 ROW_MAJOR only) ----
        if constexpr (INPUT_FORMAT == 0) {
            MaybeDeviceZoneScope("compute_tilize");
            const uint32_t x_slot_offset = (block_idx % DEPTH_X) * M_BLOCK * KR_PAD;
            const uint32_t t_first = moe_fused_swiglu::inject_first(my_col);
            for (uint32_t t = t_first; t < m_eff; t += HGROUPS) {
                // The reader reserved the whole resident slot before publishing cb_x_in. Pack the
                // converted row straight into its final multicast offset but leave cb_x_tiles'
                // FIFO state untouched: the reader remains its only pusher. The tiny ready CB is
                // the compute->reader completion edge that cb_x_stage's payload used to provide.
                tilize_done.reserve_back(1);
                {
                    MaybeDeviceZoneScope("compute_tilize_input_wait");
                    DataflowBuffer x_input(cb_x_in);
                    x_input.wait_front(TILE_H);
                }
                {
                    MaybeDeviceZoneScope("compute_tilize_body");
                    tilize_row<KR_PAD, cb_x_in, cb_x_tiles>(TILE_H, x_slot_offset + t * KR_PAD);
                }
                tilize_done.push_back(1);
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
                // The ragged column (hn_cols < HN_PAD) narrows the FMA width of the chunk it falls in;
                // the host guarantees every chunk keeps at least one real column. 0 means "full".
                const uint32_t chunk_col0 = c * GU_CHUNK_W;
                const uint32_t valid = (hn_cols > chunk_col0) ? (hn_cols - chunk_col0) : 0;
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
                MatmulShape shape_c = MatmulShape::of(
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
                    if (run_up) {
                        MaybeDeviceZoneScope("compute_up_matmul");
                        matmul_row_major<
                            /*init_matmul=*/true,
                            /*retain_in0=*/true,
                            /*retain_in1=*/false,
                            MatmulTarget::Interm>(
                            x_buf,
                            weight_buf,
                            accum_buf,
                            accum_buf,
                            shape_c,
                            /*in1_width=*/GU_CHUNK_W,
                            /*out_row_width=*/HN_PAD,
                            KrSteps{kr_rows},
                            NoPreKBlock{},
                            NoIn1Offset{},
                            chunk_col0);
                    } else {
                        MaybeDeviceZoneScope("compute_gate_matmul");
                        matmul_row_major<
                            /*init_matmul=*/true,
                            /*retain_in0=*/true,
                            /*retain_in1=*/false,
                            MatmulTarget::Interm>(
                            x_buf,
                            weight_buf,
                            accum_buf,
                            accum_buf,
                            shape_c,
                            /*in1_width=*/GU_CHUNK_W,
                            /*out_row_width=*/HN_PAD,
                            KrSteps{kr_rows},
                            NoPreKBlock{},
                            NoIn1Offset{},
                            chunk_col0);
                    }
                }
            }
            gate_buf.push_back(gu_block_tiles);
            up_buf.push_back(gu_block_tiles);
            // packer_l1_acc leaves L1 accumulation ENABLED after the last chunk (the `down` matmul
            // below carries the same note). The reduce chain that follows would otherwise ACCUMULATE
            // onto stale L1 instead of overwriting.
            pack_reconfig_l1_acc(0);
        }

        // MY SLICE of this block's m_eff*HN_PAD tiles, from the one shared plan in common.hpp — a
        // pure function of (m_eff, KGROUPS, my_row), identical on every core, which is what keeps
        // the all-to-all deadlock-free. 0 = idle: still contributes a partial, owns no slice.
        const uint32_t slice_tiles = moe_fused_swiglu::slice_assigned(gu_block_tiles, KGROUPS, my_row);

        // ---- 3. cross-column reduce + SwiGLU ----
        {
            MaybeDeviceZoneScope("compute_reduce");
            // REDUCE-SCATTER, worker side: every contributor pushed its slice into slot `row`, so
            // the reduce is `slice_tiles` wide rather than the whole block — that factor IS the win.
            // PACK must be the ONLY pusher of the slice CBs: `cb_push_back` writes the shared
            // `tiles_received` word from the pushing RISC-V's own count, so two pushers corrupt it.
            if (slice_tiles) {
#ifdef SITU_GLU
                // SiTU fuses both reductions with the binary SFPU pass below.
#else
                // KGROUPS-1 contributors fold here; the last one rides the SiLU-fused add below.
                fold_chain<cb_slice_gate, cb_gather_gate>(KGROUPS - 1, slice_tiles);
                // The final gate add, with SiLU on the packer thread. Chunked to DEST_LIMIT because
                // a slice can exceed one DEST window (slice_tiles is m_eff*HN_PAD/workers, e.g. 9 at
                // HN_PAD 9); each chunk pairs the partial with its corresponding final gate tile.
                gg_buf.wait_front(slice_tiles);
                for (uint32_t t0 = 0; t0 < slice_tiles; t0 += DEST_LIMIT) {
                    uint32_t w = slice_tiles - t0;
                    if (w > DEST_LIMIT) {
                        w = DEST_LIMIT;
                    }
                    add_silu_elementwise(sg_buf, gg_buf, silu_buf, w, t0);
                }
                gg_buf.pop_front(slice_tiles);

                fold_chain<cb_slice_up, cb_gather_up>(KGROUPS, slice_tiles);
#endif
            }
        }

        {
            MaybeDeviceZoneScope("compute_swiglu");
            // SwiGLU on MY SLICE ONLY, straight into the CB the writer unicasts from. The workers'
            // slices tile the ROOT's cb_h_local as they LAND, so the gather IS the assembly: no
            // landing CB and no root-side copy.
            if (slice_tiles) {
#ifdef SITU_GLU
                fold_situ_glu_blocked<cb_gather_gate, cb_gather_up, cb_h_slice>(KGROUPS, slice_tiles);
#else
                // Inherits phase 1's hoisted cb_gate_acc pack format, which is correct exactly
                // because cb_h_slice is bfp8 — the epilogue's single dtype boundary.
                mul_blocked<cb_gate_silu, cb_slice_up, cb_h_slice>(slice_tiles);
#endif
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

        // ---- 4. down matmul: HGROUPS K-blocks. Non-last blocks packer-L1-accumulate in the
        // fixed bf16 scratch; the last reloads that partial into DEST, adds its contribution,
        // and packs straight to the caller-owned bfp8 output region. ----
        {
            MaybeDeviceZoneScope("compute_down");
            out_tiles_buf.reserve_back(out_block_tiles);
            reconfig_data_format(cb_w_down, cb_h);
            pack_reconfig_data_format(cb_out_tiles);
            if (wd_mrow) {
                // The reader publishes the complete resident W_down shard once, while cb_h carries
                // eight consecutive 1xHID_T activation rows.  Each call has one K-block, so the
                // result never leaves DEST for an intermediate BF16 accumulation spill.
                constexpr uint32_t WD_RESIDENT_TILES = HGROUPS * HN_PAD * WD_EC_MAX;
                wd_buf.wait_front(WD_RESIDENT_TILES);
                matmul_block_init(cb_h, cb_w_down, false, down_ec, 1, HID_T);
                const MatmulShape row_shape = MatmulShape::of(1, 1, 1, down_ec, HID_T, 1);
                for (uint32_t r = 0; r < down_rows; ++r) {
                    matmul_row_major<
                        /*init_matmul=*/false,
                        /*retain_in0=*/false,
                        /*retain_in1=*/true,
                        MatmulTarget::Out>(
                        h_buf,
                        wd_buf,
                        out_tiles_buf,
                        out_tiles_buf,
                        row_shape,
                        /*in1_width=*/WD_EC_MAX,
                        /*out_row_width=*/out_ec_max,
                        FullKSteps{});
                    // The full output allocation was reserved before the loop, but publishing
                    // this completed row advances the CB write pointer to row r+1 and lets the
                    // writer issue row r while compute works on the next W_down matmul.
                    out_tiles_buf.push_back(out_ec_max);
                }
                if (!wd_mgroup) {
                    h_buf.wait_front(HID_T);
                    h_buf.pop_front(HID_T);  // ordinary path's payload-free alignment slot
                }
                wd_buf.pop_front(WD_RESIDENT_TILES);
            } else if constexpr (WD_PACKED) {
                // Balanced hidden slices sit contiguously in the resident W_down payload while the
                // producer still publishes fixed HN_PAD*EC_MAX blocks for flow control: hold the read
                // pointer at the CB base and address the packed rows explicitly. Short tails only.
                constexpr uint32_t WD_BLOCK_TILES = HN_PAD * WD_EC_MAX;
                constexpr uint32_t WD_RESIDENT_TILES = HGROUPS * WD_BLOCK_TILES;
                auto wait_packed_wd = [&](uint32_t block, uint32_t, bool) {
                    wd_buf.wait_front((block + 1) * WD_BLOCK_TILES);
                };
                matmul_row_major<
                    /*init_matmul=*/true,
                    /*retain_in0=*/false,
                    /*retain_in1=*/true,
                    MatmulTarget::Out>(
                    h_buf,
                    wd_buf,
                    out_tiles_buf,
                    out_interm_buf,
                    shape_dn,
                    /*in1_width=*/WD_EC_MAX,
                    /*out_row_width=*/EC_MAX,
                    HnSteps{},
                    wait_packed_wd,
                    PackedWdOffset{});
                wd_buf.pop_front(WD_RESIDENT_TILES);
            } else {
                matmul_row_major<
                    /*init_matmul=*/true,
                    /*retain_in0=*/false,
                    /*retain_in1=*/false,
                    MatmulTarget::Out>(
                    h_buf,
                    wd_buf,
                    out_tiles_buf,
                    out_interm_buf,
                    shape_dn,
                    /*in1_width=*/WD_EC_MAX,
                    /*out_row_width=*/EC_MAX,
                    HnSteps{});
            }
            if (!wd_mrow) {
                out_tiles_buf.push_back(out_block_tiles);
            }
            // Leave a known packer state for the next M-block's gate/up path.
            pack_reconfig_l1_acc(0);
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
