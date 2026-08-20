// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm writer (BRISC / NoC1 - the reader owns NoC0, so the read and write
// streams overlap instead of contending, lever B9).
//
// TILE output      : drains cb_output_tiles, one barrier per block (lever B7).
// ROW_MAJOR output : drains cb_rm_out (tile-sized pages produced by the
//                    untilize helper) and writes only the VALID sticks and only
//                    W_true bytes of each - tile padding is never written back.
//
// The CB-WRAP INVARIANT documented in the reader applies here too: every
// cb_wait_front/cb_pop_front is a fixed BLOCK_HT * WT_SCALE_BLOCK (TILE) or
// WT_SCALE_BLOCK (ROW_MAJOR) pages.  The final row-block of the tensor still
// carries a full BLOCK_HT of tile-rows; this kernel simply does not WRITE the
// phantom ones.
//
// HELPER SUBSTITUTION: the row-major drain does not call
// dataflow_kernel_lib::write_sticks_after_untilize() because that helper owns
// its own block loop over `total_num_rows` and cannot be interleaved with this
// op's (row-block x W-chunk) iteration order, which must match the compute
// kernel's untilize call sequence exactly, nor can it skip phantom tile-rows.
// The body below is the same wait/write/barrier/pop shape, driven by this op's
// loop nest.
//
// HELPER SUBSTITUTION 2 - ONE-PACKET DISPATCH.  Neither drain goes through
// `noc_async_write_page` / `noc_async_write_tile`.  Both call `noc_async_write<BOUND>`
// with a COMPILE-TIME size bound as the template `max_page_size`, which is the only
// thing that selects the ONE-PACKET NoC dispatch
// (`if constexpr (max_page_size <= NOC_MAX_BURST_SIZE)` -> `noc_async_write_one_packet`,
// dataflow_api.h:838) over the ANY-LENGTH one (`ncrisc_noc_fast_write_any_len`, whose
// body is a runtime `while (len_bytes > NOC_MAX_BURST_SIZE)` chunking loop).  Dropping
// that dispatch measured 12.0% on the isolated writer (13.7% on the isolated reader) at
// 2 KB pages; the bake-off is in perf_experiments/stateful_noc/.
//
// CAPABILITY gap, not ergonomics: `noc_async_write_page` hard-codes
// `noc_async_write<NOC_MAX_BURST_SIZE + 1, false, posted>` (dataflow_api.h:1258), so no
// call site that names a page id can reach the one-packet path even when it knows the
// page size at compile time (this kernel is handed it as compile-time arg 17,
// IN_TILE_BYTES).  `TensorAccessor` exposes only per-page `get_noc_addr` and
// `PagesAddressIteratorInterleaved` is by its own docs "accessor.get_noc_addr for each
// page without complex optimizations" (pages_address_iterator.h:267), so there is no
// surface that expresses "issue this page SET".  The actionable fix is a `max_page_size`
// template parameter on `noc_async_write_page`, or a `pages_affine(acc, first_page, n)`
// page-run iterator owning both the one-packet dispatch and the constant advance.
//
// The bound is an UPPER bound, so the polarity is right: above NOC_MAX_BURST_SIZE the
// any-length form is selected automatically (the hardware primitive cannot take a
// larger transfer).  That matters because nothing checks the size at runtime -
// sanitize.h has no burst-size rule, so an oversized one-packet write would be silent
// corruption.  Every TILE page (bf16 2048, fp32 4096, bfloat8_b 1088) is under the bound
// on both Wormhole (8192) and Blackhole (16384), so on the TILE path the fallback folds
// out of the binary entirely; only the ROW_MAJOR stick chunk can exceed it (very large
// W), and there the fallback is live.  See the two *_BYTES constants below.

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

constexpr auto output_args = TensorAccessorArgs<33>();

FORCE_INLINE uint32_t umin(uint32_t a, uint32_t b) { return a < b ? a : b; }

}  // namespace

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row_block = get_arg_val<uint32_t>(1);
    const uint32_t num_row_blocks_here = get_arg_val<uint32_t>(2);
    const uint32_t W_OFFSET = get_arg_val<uint32_t>(3);  // W-split: this core's column base (tiles)

    const auto out_acc = TensorAccessor(output_args, dst_addr);

    // `valid_ht` tile-rows of this row-block exist in the tensor; the rest are
    // phantom rows the reader clamped, and are dropped here.
    auto write_tiles = [&](uint32_t rt0, uint32_t valid_ht, uint32_t w0, uint32_t nw) {
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

    auto write_sticks = [&](uint32_t rt, bool valid, uint32_t w0, uint32_t nw) {
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
                        noc_async_write<RM_CHUNK_MAX_BYTES>(src, out_acc.get_noc_addr(row0 + r, byte_off), chunk_bytes);
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
        } else {
            write_tiles(rt0, valid_ht, w0, nw);
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
