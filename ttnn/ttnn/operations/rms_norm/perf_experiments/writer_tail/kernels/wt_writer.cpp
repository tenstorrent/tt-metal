// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PERF BENCH COPY - perf_experiments/writer_tail/.  This is NOT the
// shipped kernel; the real op lives in ../../kernels/rms_norm_writer.cpp.
//
// IDEA UNDER TEST: shorten the WRITE TAIL - the wr_issue + wr_barrier the writer
// pays AFTER the last compute has landed.  Measured baseline on the focus case
// ((1,1,32,7168) bf16 TILE, W-split G=32, Wt_core=7): wr_wait 6983 (starved),
// wr_issue 814, wr_barrier 423 - i.e. 1,237 ns of a 9,050 ns wall spent after
// compute is done, for SEVEN 2 KB one-packet page writes.
//
// Four compile-time arms, all under the SAME precision contract (CT args 30..33):
//
//   WT_PRE   hoist every `out_acc.get_noc_addr()` AHEAD of `cb_wait_front`, into
//            an L1 scratch buffer.  The writer is STARVED for 6,983 ns before
//            the first tile arrives; address generation is pure RISC arithmetic
//            with no dependency on the data, so it can be paid inside that
//            starvation window instead of on the critical tail.
//   WT_SUB   push/consume the output in N-tile units instead of one whole block,
//            so the writer's issue overlaps the tail of the compute that is still
//            producing.  Waits are CUMULATIVE and there is still exactly ONE
//            barrier and ONE pop per block, so lever B7 is unchanged.
//   WT_DIAG  DIAGNOSTIC arm: keep the address generation, drop the transfer.
//            Splits wr_issue into "address generation" vs "the write call", which
//            is what decides which of the two costs is worth attacking.
//   WT_SHARE hand N/16 of each write unit to the READER, which is idle on NOC0
//            from `mcast_recv` onward while the writer issues on NOC1.
//
// RAW-LLK / HELPER NOTE: nothing here is raw LLK.  The address hoist and the
// sub-block wait are re-orderings of the same `TensorAccessor::get_noc_addr` +
// `noc_async_write<BOUND>` calls the shipped kernel already makes.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {

constexpr uint32_t cb_output_tiles = 7;
constexpr uint32_t cb_rm_out = 9;

constexpr uint32_t IS_ROW_MAJOR = get_compile_time_arg_val(0);
constexpr uint32_t REGIME_A = get_compile_time_arg_val(1);
constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(2);
constexpr uint32_t GAMMA_IS_ROW_MAJOR = get_compile_time_arg_val(3);
constexpr uint32_t Wt_core = get_compile_time_arg_val(4);
constexpr uint32_t W_PARTIAL = get_compile_time_arg_val(5);
constexpr uint32_t BLOCK_HT = get_compile_time_arg_val(6);
constexpr uint32_t WT_REDUCE_BLOCK = get_compile_time_arg_val(7);
constexpr uint32_t WT_SCALE_BLOCK = get_compile_time_arg_val(8);
constexpr uint32_t Rt = get_compile_time_arg_val(9);
constexpr uint32_t NUM_ROWS = get_compile_time_arg_val(10);
constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(11);
constexpr uint32_t ELEM_SIZE = get_compile_time_arg_val(12);
constexpr uint32_t GAMMA_ELEM_SIZE = get_compile_time_arg_val(13);
constexpr uint32_t GAMMA_ROW_BYTES = get_compile_time_arg_val(14);
constexpr uint32_t DEST_BLOCK_CT = get_compile_time_arg_val(15);
constexpr uint32_t GAMMA_TILE_BYTES = get_compile_time_arg_val(16);
constexpr uint32_t IN_TILE_BYTES = get_compile_time_arg_val(17);
constexpr uint32_t GAMMA_INGEST_BLOCK = get_compile_time_arg_val(18);
// Lever B7: 1 = one noc barrier per block (applied), 0 = one per transaction.
constexpr uint32_t BARRIER_PER_BLOCK = get_compile_time_arg_val(19);
// /perf-measure ablation: keep every CB op and barrier, issue no NoC transfer.
constexpr uint32_t SKIP_DM_PAYLOAD = get_compile_time_arg_val(20);
// Lever B5/B6: 1 = one whole-page transaction per tile (applied), 0 = two half-page ones.
constexpr uint32_t COALESCE = get_compile_time_arg_val(21);
// --- W-split work distribution (blocking_plan._choose_group_size) ------------
// Under a W split a core owns Wt_core of WT_TOTAL columns, so the DRAM tile-row
// stride is the FULL row width and every address carries this core's column base
// (RT arg 3).  W_SPLIT == 0 makes ROW_STRIDE == Wt_core and W_OFFSET == 0, i.e.
// byte-identical addressing to the pre-split row-parallel plan.
constexpr uint32_t W_SPLIT = get_compile_time_arg_val(23);
constexpr uint32_t WT_TOTAL = get_compile_time_arg_val(25);
constexpr uint32_t ROW_STRIDE = W_SPLIT ? WT_TOTAL : Wt_core;

// --- writer_tail experiment knobs (see the header) ---------------------------
constexpr uint32_t WT_PRE = get_compile_time_arg_val(30);
constexpr uint32_t WT_SUB = get_compile_time_arg_val(31);
constexpr uint32_t WT_DIAG = get_compile_time_arg_val(32);
constexpr uint32_t WT_SHARE = get_compile_time_arg_val(33);
constexpr uint32_t SEM_WR_GO = get_compile_time_arg_val(34);
constexpr uint32_t SEM_WR_DONE = get_compile_time_arg_val(35);
constexpr uint32_t cb_addr_scratch = 14;
// Largest write unit any path takes, in pages: the whole (BLOCK_HT x ws) tile
// block, or the 32 sticks of one ROW_MAJOR tile-row.  The host sizes the scratch
// CB from the same expression; above the cap the hoist compiles out, so the arm
// degrades to the baseline instead of running off the end of the buffer.
constexpr uint32_t WT_UNIT_PAGES = BLOCK_HT * (REGIME_A ? Wt_core : WT_SCALE_BLOCK);
constexpr uint32_t PRE_SLOTS = IS_ROW_MAJOR ? 32u : WT_UNIT_PAGES;
constexpr uint32_t PRE_SLOT_CAP = 256;
constexpr uint32_t WT_PRE_ON = (WT_PRE && PRE_SLOTS <= PRE_SLOT_CAP) ? 1 : 0;
// Sub-block unit; 0 (the baseline) means "one unit = the whole block".
constexpr uint32_t WT_UNIT = WT_SUB ? WT_SUB : 0;
// Word index (uint32 units) of the two publish slots that follow the address
// table: the L1 source base and the reader's page count.
constexpr uint32_t PUB_SLOT = 2 * (PRE_SLOTS < 32u ? 32u : PRE_SLOTS);

// Lever B5/B6 off-arm: the tile page split into TWO transfers.  The split point
// must stay NoC-alignment-legal on every dtype - Blackhole's DRAM alignment is
// 64 B, and a bfloat8_b tile is 1088 B, whose midpoint (544) is NOT 64 B-aligned.
// Rounding the first half DOWN to a 64 B multiple keeps both offsets legal and
// still covers the whole page (1088 -> 512 + 576).
constexpr uint32_t SPLIT_FIRST = (IN_TILE_BYTES / 2) & ~static_cast<uint32_t>(63);
constexpr uint32_t SPLIT_SECOND = IN_TILE_BYTES - SPLIT_FIRST;

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t NUM_SCALE_CHUNKS = Wt_core / WT_SCALE_BLOCK;

// --- ONE-PACKET size bounds (see the header) ----------------------------------
// TILE output: exactly one page per transaction, always within a burst.
constexpr uint32_t OUT_PAGE_BYTES = IN_TILE_BYTES;
// ROW_MAJOR output: `nw * 32 * elem_size` per stick, clamped to the row width.
// `nw` is Wt_core in Regime A and the W-chunk width in Regime B - both compile-time.
constexpr uint32_t RM_MAX_NW = REGIME_A ? Wt_core : WT_SCALE_BLOCK;
constexpr uint32_t RM_PADDED_MAX = RM_MAX_NW * TILE_DIM * ELEM_SIZE;
constexpr uint32_t RM_CHUNK_MAX_BYTES = RM_PADDED_MAX < ROW_BYTES ? RM_PADDED_MAX : ROW_BYTES;

constexpr auto output_args = TensorAccessorArgs<37>();

FORCE_INLINE uint32_t umin(uint32_t a, uint32_t b) { return a < b ? a : b; }

}  // namespace

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row_block = get_arg_val<uint32_t>(1);
    const uint32_t num_row_blocks_here = get_arg_val<uint32_t>(2);
    const uint32_t W_OFFSET = get_arg_val<uint32_t>(3);  // W-split: this core's column base (tiles)

    const auto out_acc = TensorAccessor(output_args, dst_addr);

    // --- writer_tail scratch + hand-off flags --------------------------------
    // `abuf` is the pre-generated destination-address buffer.  Declared volatile
    // because on the WT_SHARE arm the READER reads the very same L1 words: the
    // ordering that makes that safe is (volatile address stores) -> (volatile
    // flag store) -> (reader's volatile flag load) -> (reader's address loads),
    // all on the same core's L1, so a plain L1 store/load pair is the whole
    // handshake and no NoC semaphore round trip is needed.
    volatile tt_l1_ptr uint32_t* abuf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_addr_scratch));
    volatile tt_l1_ptr uint32_t* wr_go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_WR_GO));
    volatile tt_l1_ptr uint32_t* wr_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_WR_DONE));
    uint32_t go_gen = 0;    // monotone generation counters - never reset, so no
    uint32_t done_gen = 0;  // ABA window between row-blocks / W-chunks.

    auto abuf_put = [&](uint32_t i, uint64_t a) {
        abuf[2 * i] = static_cast<uint32_t>(a);
        abuf[2 * i + 1] = static_cast<uint32_t>(a >> 32);
    };
    auto abuf_get = [&](uint32_t i) -> uint64_t {
        return static_cast<uint64_t>(abuf[2 * i]) | (static_cast<uint64_t>(abuf[2 * i + 1]) << 32);
    };

    // `valid_ht` tile-rows of this row-block exist in the tensor; the rest are
    // phantom rows the reader clamped, and are dropped here.
    [[maybe_unused]] auto write_tiles = [&](uint32_t rt0, uint32_t valid_ht, uint32_t w0, uint32_t nw) {
        const uint32_t n = BLOCK_HT * nw;
        // PERMANENT per-stage instrumentation (kernel_lib/perf_instrumentation.hpp).
        // `wr_wait` is the writer STARVED on compute; `wr_issue` is the
        // RISC-serial transaction issue; `wr_barrier` the real NoC wait.  Split
        // because a starved writer's fix lives upstream, not here.
        {
            MaybeDeviceZoneScope("wr_wait");
            cb_wait_front(cb_output_tiles, n);
        }
        uint32_t addr = get_read_ptr(cb_output_tiles);
        {
            MaybeDeviceZoneScope("wr_issue");
            for (uint32_t r = 0; r < valid_ht; ++r) {
                const uint32_t row_base = (rt0 + r) * ROW_STRIDE + W_OFFSET + w0;
                for (uint32_t w = 0; w < nw; ++w) {
                    if constexpr (!SKIP_DM_PAYLOAD) {
                        // ONE-PACKET page write (see the header): the size is a
                        // template argument, not just a value.  IN_TILE_BYTES is the
                        // accessor's aligned page size - the same equality the L1
                        // stride already relies on, ASSERTed under --dev.
                        ASSERT(out_acc.get_aligned_page_size() == IN_TILE_BYTES);
                        const uint32_t src_t = addr + w * IN_TILE_BYTES;
                        if constexpr (COALESCE) {
                            noc_async_write<OUT_PAGE_BYTES>(src_t, out_acc.get_noc_addr(row_base + w), OUT_PAGE_BYTES);
                        } else {  // lever B5/B6 off-arm: two aligned partial-page transactions
                            noc_async_write<SPLIT_FIRST>(src_t, out_acc.get_noc_addr(row_base + w), SPLIT_FIRST);
                            noc_async_write<SPLIT_SECOND>(
                                src_t + SPLIT_FIRST, out_acc.get_noc_addr(row_base + w, SPLIT_FIRST), SPLIT_SECOND);
                        }
                    }
                    if constexpr (!BARRIER_PER_BLOCK) {
                        noc_async_write_barrier();  // lever B7 off-arm
                    }
                }
                addr += nw * IN_TILE_BYTES;
            }
        }  // wr_issue
        {
            MaybeDeviceZoneScope("wr_barrier");
            noc_async_write_barrier();
        }
        cb_pop_front(cb_output_tiles, n);
    };

    // =====================================================================
    // writer_tail CANDIDATE path (TILE output).  Same transactions, same
    // one-packet dispatch, same single barrier + single pop per block as
    // `write_tiles` above - only the ORDER of the work changes:
    //   * WT_PRE   : every destination address is generated BEFORE the wait.
    //   * WT_UNIT  : the wait is CUMULATIVE in WT_UNIT-page steps, so the issue
    //                loop starts as soon as the first sub-block is packed
    //                instead of after the last one.
    //   * WT_SHARE : the trailing N/16 of the block's pages are handed to the
    //                reader (NOC0) through the L1 flag pair.
    // =====================================================================
    [[maybe_unused]] auto write_tiles_wt = [&](uint32_t rt0, uint32_t valid_ht, uint32_t w0, uint32_t nw) {
        const uint32_t n = BLOCK_HT * nw;

        if constexpr (WT_PRE_ON) {
            MaybeDeviceZoneScope("wr_pre");
            for (uint32_t r = 0; r < valid_ht; ++r) {
                const uint32_t row_base = (rt0 + r) * ROW_STRIDE + W_OFFSET + w0;
                for (uint32_t w = 0; w < nw; ++w) {
                    abuf_put(r * nw + w, out_acc.get_noc_addr(row_base + w));
                }
            }
        }

        // The reader's share is the TAIL of the block, so the writer keeps a
        // contiguous prefix and both sides issue a straight run.
        const uint32_t n_reader = WT_SHARE ? ((n * WT_SHARE) >> 4) : 0;
        const uint32_t n_writer = n - n_reader;
        // WT_SHARE forces a whole-block wait: the reader is NOT a CB consumer, so
        // it has no way of its own to know a page has been packed - the writer's
        // `cb_wait_front(n)` is the only proof, and it must therefore cover every
        // page before `wr_go` is raised.  So the share arm and the sub-block arm
        // do not compose, and that is a property of the handshake, not a knob.
        const uint32_t unit = (WT_SHARE || !WT_UNIT) ? n : WT_UNIT;

        uint32_t src_base = 0;
        uint32_t r = 0, w = 0;  // tracked incrementally: the RISCs have no divider
        for (uint32_t k = 0; k < n_writer; k += unit) {
            // CUMULATIVE wait: nothing is popped mid-block, so `want` is the
            // total number of pages produced so far, not a fresh window.
            const uint32_t want = umin(k + unit, n);
            {
                MaybeDeviceZoneScope("wr_wait");
                cb_wait_front(cb_output_tiles, want);
            }
            if (k == 0) {
                src_base = get_read_ptr(cb_output_tiles);
                if constexpr (WT_SHARE) {
                    // Publish the L1 source base + the share, then release the
                    // reader.  Volatile stores to this core's own L1, in order.
                    abuf[PUB_SLOT] = src_base;
                    abuf[PUB_SLOT + 1] = n_reader;
                    abuf[PUB_SLOT + 2] = valid_ht * nw;  // phantom rows stop here
                    *wr_go = ++go_gen;
                }
            }
            {
                MaybeDeviceZoneScope("wr_issue");
                for (uint32_t i = k; i < want; ++i) {
                    if (r < valid_ht) {  // phantom tile-rows are packed, never written
                        if constexpr (!SKIP_DM_PAYLOAD) {
                            ASSERT(out_acc.get_aligned_page_size() == IN_TILE_BYTES);
                            const uint64_t da = WT_PRE_ON
                                                    ? abuf_get(i)
                                                    : out_acc.get_noc_addr((rt0 + r) * ROW_STRIDE + W_OFFSET + w0 + w);
                            if constexpr (!WT_DIAG) {
                                noc_async_write<OUT_PAGE_BYTES>(src_base + i * IN_TILE_BYTES, da, OUT_PAGE_BYTES);
                            } else {
                                // DIAGNOSTIC: the address is generated, the
                                // transfer dropped.  Splits wr_issue in two.
                                asm volatile("" ::"r"(static_cast<uint32_t>(da)) : "memory");
                            }
                        }
                    }
                    if (++w == nw) {
                        w = 0;
                        ++r;
                    }
                }
            }
        }
        {
            MaybeDeviceZoneScope("wr_barrier");
            noc_async_write_barrier();
        }
        if constexpr (WT_SHARE) {
            // The reader's share must have LANDED before the CB pages are
            // released back to the compute thread, so the join is a pop-order
            // dependency, not just bookkeeping.
            MaybeDeviceZoneScope("wr_join");
            ++done_gen;
            while (*wr_done < done_gen) {
            }
        }
        cb_pop_front(cb_output_tiles, n);
    };

    auto write_sticks = [&](uint32_t rt, bool valid, uint32_t w0, uint32_t nw) {
        // ROW_MAJOR arm of WT_PRE: 32 stick addresses per tile-row, generated
        // before the wait.  This path pays 32 `get_noc_addr` calls per tile-row
        // against the TILE path's `nw`, so it is where the hoist has the most to
        // move off the tail.  WT_SUB does not apply: the untilize helper produces
        // cb_rm_out one whole tile-row at a time.
        if constexpr (WT_PRE_ON) {
            if (valid) {
                MaybeDeviceZoneScope("wr_pre");
                const uint32_t row0 = rt * TILE_DIM;
                const uint32_t nrows = umin(TILE_DIM, NUM_ROWS - row0);
                const uint32_t byte_off = (W_OFFSET + w0) * TILE_DIM * ELEM_SIZE;
                for (uint32_t r = 0; r < nrows; ++r) {
                    abuf_put(r, out_acc.get_noc_addr(row0 + r, byte_off));
                }
            }
        }
        {
            MaybeDeviceZoneScope("wr_wait");
            cb_wait_front(cb_rm_out, nw);
        }
        if (valid) {
            const uint32_t row0 = rt * TILE_DIM;
            const uint32_t nrows = umin(TILE_DIM, NUM_ROWS - row0);
            const uint32_t byte_off = (W_OFFSET + w0) * TILE_DIM * ELEM_SIZE;
            const uint32_t padded = nw * TILE_DIM * ELEM_SIZE;
            const uint32_t chunk_bytes = umin(padded, ROW_BYTES - byte_off);

            uint32_t src = get_read_ptr(cb_rm_out);
            {
                MaybeDeviceZoneScope("wr_issue");
                for (uint32_t r = 0; r < nrows; ++r) {
                    if constexpr (!SKIP_DM_PAYLOAD) {
                        // ONE-PACKET when the chunk is provably within a burst, the
                        // any-length form when it is not - selected at compile time
                        // by RM_CHUNK_MAX_BYTES (see the header).
                        ASSERT(chunk_bytes <= RM_CHUNK_MAX_BYTES);
                        const uint64_t da = WT_PRE_ON ? abuf_get(r) : out_acc.get_noc_addr(row0 + r, byte_off);
                        if constexpr (!WT_DIAG) {
                            noc_async_write<RM_CHUNK_MAX_BYTES>(src, da, chunk_bytes);
                        } else {
                            asm volatile("" ::"r"(static_cast<uint32_t>(da)) : "memory");
                        }
                    }
                    if constexpr (!BARRIER_PER_BLOCK) {
                        noc_async_write_barrier();  // lever B7 off-arm
                    }
                    src += padded;
                }
            }  // wr_issue
            {
                MaybeDeviceZoneScope("wr_barrier");
                noc_async_write_barrier();
            }
        }
        cb_pop_front(cb_rm_out, nw);
    };

    auto write_chunk = [&](uint32_t rt0, uint32_t valid_ht, uint32_t w0, uint32_t nw) {
        if constexpr (IS_ROW_MAJOR) {
            for (uint32_t r = 0; r < BLOCK_HT; ++r) {
                write_sticks(rt0 + r, r < valid_ht, w0, nw);
            }
        } else if constexpr (WT_PRE || WT_SUB || WT_SHARE || WT_DIAG) {
            write_tiles_wt(rt0, valid_ht, w0, nw);
        } else {
            write_tiles(rt0, valid_ht, w0, nw);  // the op's CURRENT writer, verbatim
        }
    };

    for (uint32_t b = 0; b < num_row_blocks_here; ++b) {
        const uint32_t rt0 = (start_row_block + b) * BLOCK_HT;
        const uint32_t valid_ht = umin(BLOCK_HT, Rt - rt0);

        if constexpr (REGIME_A) {
            write_chunk(rt0, valid_ht, 0, Wt_core);
        } else {
            for (uint32_t c = 0; c < NUM_SCALE_CHUNKS; ++c) {
                write_chunk(rt0, valid_ht, c * WT_SCALE_BLOCK, WT_SCALE_BLOCK);
            }
        }
    }
}
