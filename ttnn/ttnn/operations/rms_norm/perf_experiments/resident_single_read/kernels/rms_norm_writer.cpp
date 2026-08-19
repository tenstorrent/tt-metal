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

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

// --- lab-only zone gate ------------------------------------------------------
// EVERY per-stage zone call site is spelled `RSR_ZONE(x)`.  With -DRMSN_NO_ZONES
// the RAII zone becomes a no-op, so a chunk/depth sweep measured under --profile
// is not paying a marker cost that differs between arms (Regime A/C run a
// DIFFERENT NUMBER of zone executions than Regime B, which would otherwise leak
// straight into the ns delta being attributed to the idea).  Zones ON is still
// available (`no_zones=0`) for the per-stage breakdown.
#ifdef RMSN_NO_ZONES
#define RSR_ZONE(name) ((void)0)
#else
#define RSR_ZONE(name) MaybeDeviceZoneScope(name)
#endif

namespace {

constexpr uint32_t cb_output_tiles = 7;
constexpr uint32_t cb_rm_out = 9;

constexpr uint32_t IS_ROW_MAJOR = get_compile_time_arg_val(0);
// REGIME (CT arg 1) — three plans, not two:
//   0 = B  STREAMING-MASKED   two DRAM reads of x (reduce pass + scale pass).
//   1 = A  RESIDENT-FUSED     one DRAM read; the WHOLE per-core width of x AND
//                             gamma AND cb_normed AND the output CB are resident.
//   2 = C  RESIDENT-X         one DRAM read; ONLY x is resident, and the scale
//                             pass walks W in WT_SCALE_BLOCK chunks (gamma /
//                             normed / output CBs are sized per chunk).
constexpr uint32_t REGIME = get_compile_time_arg_val(1);
constexpr bool REGIME_A = (REGIME == 1);
constexpr bool REGIME_C = (REGIME == 2);
// x is read from DRAM exactly ONCE per row-block and stays in cb_input_tiles for
// the scale pass.  This is the property the experiment is about.
constexpr bool RESIDENT_X = REGIME_A || REGIME_C;
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

// Lever B5/B6 off-arm: the tile page split into TWO transfers.  The split point
// must stay NoC-alignment-legal on every dtype - Blackhole's DRAM alignment is
// 64 B, and a bfloat8_b tile is 1088 B, whose midpoint (544) is NOT 64 B-aligned.
// Rounding the first half DOWN to a 64 B multiple keeps both offsets legal and
// still covers the whole page (1088 -> 512 + 576).
constexpr uint32_t SPLIT_FIRST = (IN_TILE_BYTES / 2) & ~static_cast<uint32_t>(63);
constexpr uint32_t SPLIT_SECOND = IN_TILE_BYTES - SPLIT_FIRST;

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t NUM_SCALE_CHUNKS = Wt_core / WT_SCALE_BLOCK;

constexpr auto output_args = TensorAccessorArgs<25>();

FORCE_INLINE uint32_t umin(uint32_t a, uint32_t b) { return a < b ? a : b; }

}  // namespace

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row_block = get_arg_val<uint32_t>(1);
    const uint32_t num_row_blocks_here = get_arg_val<uint32_t>(2);

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
            RSR_ZONE("wr_wait");
            cb_wait_front(cb_output_tiles, n);
        }
        uint32_t addr = get_read_ptr(cb_output_tiles);
        {
            RSR_ZONE("wr_issue");
            for (uint32_t r = 0; r < valid_ht; ++r) {
                const uint32_t row_base = (rt0 + r) * Wt_core + w0;
                for (uint32_t w = 0; w < nw; ++w) {
                    if constexpr (!SKIP_DM_PAYLOAD) {
                        if constexpr (COALESCE) {
                            noc_async_write_tile(row_base + w, out_acc, addr + w * IN_TILE_BYTES);
                        } else {  // lever B5/B6 off-arm: two aligned partial-page transactions
                            const uint32_t src_t = addr + w * IN_TILE_BYTES;
                            noc_async_write(src_t, out_acc.get_noc_addr(row_base + w), SPLIT_FIRST);
                            noc_async_write(
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
            RSR_ZONE("wr_barrier");
            noc_async_write_barrier();
        }
        cb_pop_front(cb_output_tiles, n);
    };

    auto write_sticks = [&](uint32_t rt, bool valid, uint32_t w0, uint32_t nw) {
        {
            RSR_ZONE("wr_wait");
            cb_wait_front(cb_rm_out, nw);
        }
        if (valid) {
            const uint32_t row0 = rt * TILE_DIM;
            const uint32_t nrows = umin(TILE_DIM, NUM_ROWS - row0);
            const uint32_t byte_off = w0 * TILE_DIM * ELEM_SIZE;
            const uint32_t padded = nw * TILE_DIM * ELEM_SIZE;
            const uint32_t chunk_bytes = umin(padded, ROW_BYTES - byte_off);

            uint32_t src = get_read_ptr(cb_rm_out);
            {
                RSR_ZONE("wr_issue");
                for (uint32_t r = 0; r < nrows; ++r) {
                    if constexpr (!SKIP_DM_PAYLOAD) {
                        noc_async_write(src, out_acc.get_noc_addr(row0 + r, byte_off), chunk_bytes);
                    }
                    if constexpr (!BARRIER_PER_BLOCK) {
                        noc_async_write_barrier();  // lever B7 off-arm
                    }
                    src += padded;
                }
            }  // wr_issue
            {
                RSR_ZONE("wr_barrier");
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

        // Regime C is CHUNKED on the write side (its output CB is sized per
        // W-chunk), so it takes the same path as Regime B - no third branch.
        if constexpr (REGIME_A) {
            write_chunk(rt0, valid_ht, 0, Wt_core);
        } else {
            for (uint32_t c = 0; c < NUM_SCALE_CHUNKS; ++c) {
                write_chunk(rt0, valid_ht, c * WT_SCALE_BLOCK, WT_SCALE_BLOCK);
            }
        }
    }
}
