// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BAKE-OFF ARTIFACT — idea `write_issue_menu`. NOT the real op.
//
// A byte-for-byte reconstruction of the real op's writer
// (`ttnn/ttnn/operations/tilize/kernels/tilize_writer.cpp`, `store_block`) with
// ONE compile-time dial, `WRITE_ISSUE_MODE`, selecting the MECHANISM by which
// the batch's `rows_this_batch * block_width_tiles` whole-tile-page writes are
// issued and drained. Everything else — the block resolve, the batch cap, the
// CB handshake, the zone instrumentation, the bytes, the destinations and the
// transaction count/size — is identical in every mode, which is what makes the
// menu a fair comparison and every mode bit-identical by construction (the two
// exceptions are called out per mode below).
//
// THE MODES (host picks one via `defines={"WRITE_ISSUE_MODE": "<n>"}`):
//
//  0 BASELINE      today's op verbatim: row+column issue-order rotation by
//                  `block_id`, all `pages_this_batch` writes issued through
//                  `noc_async_write<out_tile_bytes>` (i.e. the ONE-PACKET path
//                  on `write_cmd_buf`), then ONE `noc_async_write_barrier()`.
//
//  --- (a) issue ORDER / DRAM-bank coverage -------------------------------
//  1 ROT_NONE      no rotation at all (ascending rows, ascending columns).
//                  This is the DIAGNOSTIC CONTROL, not a candidate: on the
//                  focus shape it collapses the grid's set of first-written
//                  banks from 12 to 3 (see the enumeration below). If it ties
//                  BASELINE, then write issue ORDER does not matter on this op
//                  at all and every rotation in this family is a null — which
//                  is a stronger, cheaper result than searching the family.
//  2 ROT_BANKUNIF  a rotation solved for MAXIMALLY UNIFORM starting-bank
//                  coverage (derivation below), replacing the current
//                  `block_id % block_width_tiles`.
//
//  --- (b) barrier placement ----------------------------------------------
//  3 FLUSH_HALF    issue half the batch, `noc_async_writes_flushed()` (waits
//                  for departure, not completion), issue the other half, then
//                  the single full barrier — so command issue and drain
//                  overlap instead of issue-all-then-drain.
//  4 BARRIER_HALF  a FULL `noc_async_write_barrier()` after each half (two
//                  barriers per batch). The pessimistic bracket of (b).
//
//  --- (c) NoC command-buffer spreading -----------------------------------
//  5 CMDBUF2       round-robin the batch's writes over TWO of NoC1's four
//                  command buffers, {WR=0, WR_REG=2}.
//  6 CMDBUF4       round-robin over all four, {0, 2, 3, 1}.
//                  `noc_async_write_one_packet` spins on
//                  `noc_cmd_buf_ready(noc, write_cmd_buf)` — ONE buffer — so
//                  back-pressure reaching the NIU stalls the RISC even though
//                  three more command buffers sit idle. These two modes ask
//                  whether more command buffers buy more outstanding writes.
//                  SAFE re CMD BUF 1 (BRISC_RD_CMD_BUF): the only read this
//                  kernel ever issues is the split reader's, and
//                  `read_sticks_rotated` ends every tile-row with
//                  `noc_async_read_barrier()`, so buffer 1 is quiescent by the
//                  time `store_block` runs. Barriers are unaffected: the
//                  non-posted write counters
//                  (`noc_nonposted_writes_num_issued/_acked`) are per-NOC, not
//                  per-command-buffer, and `ncrisc_noc_fast_write` updates them
//                  identically whichever buffer it is handed.
//
//  --- (e) other dataflow_api write entry points --------------------------
//  7 POSTED        the same one-packet write with `posted = true` (no ack
//                  requested: `NOC_CMD_RESP_MARKED` is dropped), drained with
//                  `noc_async_posted_writes_flushed()`.
//                  *** THIS MODE IS NOT A DROP-IN. *** A posted write has only
//                  a "sent" signal, never an "acked"/"flushed" one, so the
//                  drain proves the payload has LEFT this core (which is all
//                  the CB needs before `cb_pop_front` lets compute overwrite
//                  the L1 source) but does NOT prove it LANDED in DRAM before
//                  the program reports complete. It is measured here for the
//                  menu's sake and reported WITH that hazard, never silently
//                  chosen. `noc_async_full_barrier` (the strongest thing the
//                  API offers) also only waits `ncrisc_noc_posted_writes_sent`
//                  for posted traffic, so no barrier in the API closes the gap.
//  8 FLUSH_ONLY    THE HAZARD-FREE FORM OF THE SAME MECHANISM, added after
//                  POSTED measured a real (small) win and the question became
//                  "can that win be had safely?". Writes stay NON-POSTED
//                  (every one is still acked and still counted), but the
//                  PER-BATCH drain drops from `noc_async_write_barrier()`
//                  (spins until ACKED, i.e. landed at the destination) to
//                  `noc_async_writes_flushed()` (spins until SENT, i.e. the
//                  payload has been read out of L1 and the request has
//                  departed) — and ONE `noc_async_write_barrier()` is issued
//                  after the whole block loop, before the kernel returns.
//                  Both halves of the contract are then met exactly:
//                    * SENT is precisely and only what `cb_pop_front` needs,
//                      because the CB page it releases is the write's SOURCE.
//                      This is not an inference — `prefetcher/.../writer_l1.cpp`
//                      states it and switches on it ("cb_pop_front below frees
//                      the local CB source page, so we must drain the writes
//                      whose SOURCE is that page before releasing it").
//                    * ACKED (landed) is still established for every write of
//                      the kernel before the kernel exits, so the program's
//                      completion still means the output buffer is complete —
//                      the guarantee POSTED gives up.
//                  It removes the ack ROUND TRIP from the inner loop, which is
//                  the same thing POSTED removes, while keeping the ack itself.
//
// (d) LARGER EFFECTIVE TRANSACTION is INEXPRESSIBLE on this plan and is
// therefore not a mode. A core's batch writes output pages `page_base + i`,
// `i = 0..block_width_tiles-1` — CONSECUTIVE page ids. An interleaved buffer
// maps page `p` to bank `p % num_banks` with the in-bank offset advancing only
// once per full trip round the banks (`tensor_accessor.h`,
// `get_bank_and_offset_from_page_id`: `bank_id = page_id % num_banks;
// bank_offset = page_id / num_banks`), so consecutive pages are on DIFFERENT
// banks — i.e. different NoC endpoints — and no single `noc_async_write` can
// cover two of them. Coalescing would require the pages a core owns to be
// bank-STRIDED, which is a work-split change, not a write-issue change (and one
// already measured at 26947 ns on this shape because it shatters the 512 B
// stick read into eight 64 B reads).
//
// ROT_BANKUNIF DERIVATION. Block `b` writes pages `page_base + i` with
// `page_base = row * C + col_base`, `col_base = b * bw` on the flagged plan
// (R = 1, one block per core), so its first write lands on bank
// `(bw*b + rot(b)) % NB`. Because only `bw` consecutive residues are reachable
// from a given `page_base`, a permutation can choose WHICH of those `bw` banks
// is hit first and nothing more. `bw*b mod NB` is periodic in `b` with period
// `P = NB / gcd(bw, NB)` (= 3 for bw=8, NB=12), i.e. the grid's blocks fall
// into `P` base classes; sweeping `rot` uniformly WITHIN a class is what makes
// the global histogram flat. Hence `rot(b) = (b / P) % bw`.
// Enumerated over b = 0..63, bw = 8, NB = 12 (bank -> #cores whose first write
// lands there; ideal 64/12 = 5.33):
//     ROT_NONE      {0:22, 4:21, 8:21}                     <- 3 banks only
//     BASELINE      {0:6,1:6,2:4,3:6,4:6,5:4,6:6,7:6,8:4,9:6,10:6,11:4}
//     ROT_BANKUNIF  {0:6,1:5,2:5,3:5,4:6,5:6,6:5,7:5,8:6,9:5,10:5,11:5}
// So the premise that the graduated rotation leaves the grid clustered is FALSE
// on this shape: `b % 8` already reaches all twelve banks nearly evenly, because
// `b mod 8` and `b mod 3` are independent (CRT mod 24). ROT_BANKUNIF only
// removes a 4-vs-6 ripple. The honest prediction is a null, and ROT_NONE is the
// control that says whether the whole axis is live.
//
// ============================ MEASURED (the menu) ===========================
// Wormhole B0 n150, 8x8 = 64/64 cores, 1 GHz, 12 DRAM banks. Every mode
// bit-identical (`torch.equal`) on all six sweep shapes except CMDBUF4, which
// hung. Numbers are PAIRED medians: all modes are dispatched ROUND-ROBIN inside
// ONE profiled process (`test_interleaved`), because run-to-run drift within a
// single invocation is ~+-5% here — the same size as every effect in this menu,
// so one-mode-per-invocation cannot separate them. WRITE-STAGE =
// `TILIZE_ABLATE=reads,compute`; WHOLE-OP = un-ablated. Both are
// `DEVICE KERNEL DURATION [ns]`.
//
// Focus shape [1,1,32,16384] (5 reps, then 12 paired reps for the two movers):
//   mode           write-stage           whole-op
//   BASELINE          8412  (+0.0%)        12475  (+0.0%)
//   ROT_NONE          8795  (+4.6%)        13910 (+11.5%)   <- CONTROL
//   ROT_BANKUNIF      8726  (+3.7%)        12329  (-1.2%)
//   FLUSH_HALF        8569  (+1.9%)        12533  (+0.5%)
//   BARRIER_HALF      8693  (+3.3%)        12714  (+1.9%)
//   CMDBUF2           8356  (-0.7%)        12597  (+1.0%)
//   POSTED            8100  (-4.0%)        12370  (-1.5%)   <- 12 paired reps
//   FLUSH_ONLY        8697  (+4.6%)        12413  (-0.3%)
//
// WHAT THE MENU SAYS, in one line each:
//
// * ROT_NONE is the headline, and it is a DON'T-TOUCH rather than a win: the
//   rotation the op already ships is worth +11.5% / +6.0% (two runs, opposite
//   mode orderings) on the whole op here and +2.3..+7.4% on the write stage of
//   every other shape measured. Removing it is a real regression.
// * ROT_BANKUNIF is a NULL, and the enumeration above says why in advance:
//   `block_id % block_width_tiles` ALREADY reaches all twelve banks nearly
//   evenly. There is no headroom left on the (a) axis to collect.
// * FLUSH_HALF / BARRIER_HALF are NULL / mild regression. Splitting the batch's
//   barrier does not overlap anything, because the batch is already one
//   in-flight group and the drain is not RISC-serial.
// * CMDBUF2 is a NULL with an INFORMATIVE shape: `writer_issue` drops
//   (4177/4205/4332 -> 3957/3726/4146 ns) and `writer_barrier` rises by almost
//   exactly as much (813/816/839 -> 1068/1071/1079 ns). A second command buffer
//   really does let the RISC hand commands off sooner; the bytes then simply
//   wait somewhere else. This is the direct measurement that the write stage is
//   FABRIC-bound, not command-issue-bound, and it is why every other issue-side
//   mechanism in this menu is a null.
// * POSTED is the only mover, and FLUSH_ONLY is the experiment that explains
//   WHY. FLUSH_ONLY keeps the ack and only defers WAITING for it (per-batch
//   `noc_async_writes_flushed`, one real barrier at the end): NULL everywhere
//   (+4.6% / -0.3% here, -3.7% / +0.8% on [1,1,16384,32]). POSTED removes the
//   ack PACKETS from the NoC altogether: -4.0% on the write stage here, -8.6%
//   on [1,1,2048,64]. So the saving is not a RISC wait that can be moved — it
//   is return-path fabric traffic that has to be deleted. Consistent with
//   CMDBUF2: the only lever that moves a fabric-bound stage is one that removes
//   fabric traffic.
//
// POSTED across the domain (whole-op paired median vs BASELINE; write-stage in
// brackets; two independent runs where two numbers are given):
//   [1,1,32,16384]   -1.5%          [-4.0%]
//   [1,1,32,32768]   -3.2%          [+1.2%]
//   [1,1,1024,1024]  -2.5%          [-2.2%]
//   [1,1,2048,2048]  -1.0% / +1.6%  [-0.8% / +0.4%]   <- flat, the big shape
//   [1,1,16384,32]   -1.9% / -3.1%  [-2.0% / -7.8%]
//   [1,1,2048,64]    -6.3% / -7.4%  [-9.9% / -8.6%]   <- best: 64 tiny batches
// Zones on the focus shape confirm the mechanism: `writer_barrier` 839 -> 400 ns
// (ablated) and 1020-1366 -> 616-832 ns (whole op), `writer_issue` unchanged.
//
// CMDBUF4 HUNG THE DEVICE (dispatch timeout on the focus shape, cores 19-26 and
// 18-26; the run before it, CMDBUF2, was clean on all six shapes). So command
// buffers 0 and 2 are usable for a raw `ncrisc_noc_fast_write` from BRISC and
// buffer 3 (AT) and/or 1 (RD) are NOT. Most likely cause: `ncrisc_noc_fast_write`
// writes only NOC_TARG_ADDR_LO / NOC_RET_ADDR_LO / NOC_RET_ADDR_COORDINATE and
// inherits the ADDR_MID (upper address bits) and any other per-buffer state that
// `noc_local_state_init` set up for that buffer's INTENDED use. Not chased
// further, because CMDBUF2 already answers the question the (c) option asked and
// answers it "no".
// ===========================================================================
//
// RAW-API JUSTIFICATION (this file is the raw-LLK sandbox for the idea; the
// real writer's own justification is at the head of `tilize_writer.cpp` and is
// inherited verbatim — `write_sticks_after_untilize` addresses ROW_MAJOR sticks
// and computes wrong addresses for this TILE-page destination;
// `local_copy_helpers_dataflow` requires an L1 destination). Additionally
// bypassed HERE: `noc_async_write` itself, in modes 5/6/7, in favour of
// `ncrisc_noc_fast_write` on a caller-chosen command buffer and of the `posted`
// template parameter. Gap class: CAPABILITY for the command buffer (no
// dataflow_api entry point takes a `cmd_buf` argument at all — it is hard-wired
// to `write_cmd_buf` in every write overload), ERGONOMICS for `posted` (the
// parameter exists and is reachable as `noc_async_write<size, true, true>`, but
// the matching drain is a DIFFERENT function, `noc_async_posted_writes_flushed`,
// with weaker semantics than its name suggests — the caller has to own that
// lifecycle by hand and nothing in the signature warns them).
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#include "tilize_stick_read.hpp"

#ifndef WRITE_ISSUE_MODE
#define WRITE_ISSUE_MODE 0
#endif
#ifndef NUM_DRAM_BANKS
#define NUM_DRAM_BANKS 12
#endif

#define WIM_BASELINE 0
#define WIM_ROT_NONE 1
#define WIM_ROT_BANKUNIF 2
#define WIM_FLUSH_HALF 3
#define WIM_BARRIER_HALF 4
#define WIM_CMDBUF2 5
#define WIM_CMDBUF4 6
#define WIM_POSTED 7
#define WIM_FLUSH_ONLY 8

#if WRITE_ISSUE_MODE == WIM_CMDBUF2
constexpr uint32_t WIM_NUM_CMD_BUFS = 2;
constexpr uint32_t WIM_CMD_BUFS[4] = {0, 2, 0, 2};
#elif WRITE_ISSUE_MODE == WIM_CMDBUF4
constexpr uint32_t WIM_NUM_CMD_BUFS = 4;
constexpr uint32_t WIM_CMD_BUFS[4] = {0, 2, 3, 1};
#endif

namespace {

// `gcd` at compile time, for the ROT_BANKUNIF period.
constexpr uint32_t wim_gcd(uint32_t a, uint32_t b) { return b == 0 ? a : wim_gcd(b, a % b); }

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_row_blocks = get_compile_time_arg_val(2);  // R
    constexpr uint32_t tensor_col_tiles = get_compile_time_arg_val(3);   // C
    constexpr uint32_t num_row_groups = get_compile_time_arg_val(4);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t write_rows_per_barrier = get_compile_time_arg_val(6);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t col_tile_offset = get_compile_time_arg_val(8);
    // Split-reader compile args, carried verbatim for CT-arg-index
    // compatibility with the real op's `writer_ct_args` list. NOT this idea's
    // target: the split read below is copied byte-for-byte from the real
    // kernel so the shapes that trigger it still run correctly.
    constexpr uint32_t split_reader_rows = get_compile_time_arg_val(9);
    constexpr uint32_t cb_input_rows_split = get_compile_time_arg_val(10);
    constexpr uint32_t tile_h = get_compile_time_arg_val(11);
    constexpr uint32_t block_row_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t split_writer_share_pct = get_compile_time_arg_val(13);
    constexpr uint32_t col_byte_offset = col_tile_offset * (block_row_bytes / block_width_tiles);
    constexpr auto out_args = TensorAccessorArgs<14>();
    [[maybe_unused]] constexpr auto in_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block_id = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);
    const uint32_t block_stride = get_arg_val<uint32_t>(3);
    const uint32_t src_addr = get_arg_val<uint32_t>(4);

    const auto out_acc = TensorAccessor(out_args, dst_addr);
    [[maybe_unused]] const auto in_acc = TensorAccessor(in_args, src_addr);

    // ROT_BANKUNIF's period (see the derivation at the head of the file).
    constexpr uint32_t wim_rot_period = NUM_DRAM_BANKS / wim_gcd(block_width_tiles, NUM_DRAM_BANKS);

    for (uint32_t b = 0; b < num_blocks_this_core; ++b) {
        const uint32_t block_id = start_block_id + b * block_stride;
        const uint32_t row_group = block_id / num_w_chunks;
        const uint32_t w_chunk = block_id - row_group * num_w_chunks;

        const uint32_t row_start = (row_group * tensor_row_blocks) / num_row_groups;
        const uint32_t row_end = ((row_group + 1) * tensor_row_blocks) / num_row_groups;
        const uint32_t block_row_extent = row_end - row_start;
        const uint32_t col_base = col_tile_offset + w_chunk * block_width_tiles;

        // load_block, TRAILING HALF — the real kernel's split-reader block,
        // verbatim. Not this idea's lever; present so the tall/narrow sweep
        // point runs the same pipeline it does in the op.
        if constexpr (split_reader_rows > 0) {
            uint32_t rows_writer = (block_row_extent * split_writer_share_pct) / 100;
            if (rows_writer >= block_row_extent) {
                rows_writer = block_row_extent - 1;
            }
            if (rows_writer > 0) {
                MaybeDeviceZoneScope("writer_split_read");
                const uint32_t rows_reader = block_row_extent - rows_writer;
                tilize_kernel::read_sticks_rotated<cb_input_rows_split, block_width_tiles, tile_h, block_row_bytes>(
                    in_acc,
                    /* num_tile_rows           */ rows_writer,
                    /* start_page              */ (row_start + rows_reader) * tile_h,
                    /* byte_offset_within_page */ col_byte_offset + w_chunk * block_row_bytes,
                    /* rotation                */ block_id);
            }
        }

        uint32_t rows_done = 0;
        while (rows_done < block_row_extent) {
            uint32_t rows_this_batch = block_row_extent - rows_done;
            if (rows_this_batch > write_rows_per_barrier) {
                rows_this_batch = write_rows_per_barrier;
            }
            {
                const LocalCBInterface& cb = get_local_cb_interface(cb_output_tiles);
                const uint32_t contig_rows = ((cb.fifo_limit - cb.fifo_rd_ptr) / cb.fifo_page_size) / block_width_tiles;
                if (rows_this_batch > contig_rows) {
                    rows_this_batch = contig_rows;
                }
            }

            const uint32_t pages_this_batch = rows_this_batch * block_width_tiles;
            {
                MaybeDeviceZoneScope("writer_wait_out");
                cb_wait_front(cb_output_tiles, pages_this_batch);
            }

            const uint32_t l1_read_base = get_read_ptr(cb_output_tiles);

            // ---- (a) the issue-order rotation, per mode --------------------
#if WRITE_ISSUE_MODE == WIM_ROT_NONE
            const uint32_t col_rot = 0;
            const uint32_t row_rot = 0;
#elif WRITE_ISSUE_MODE == WIM_ROT_BANKUNIF
            // Uniform sweep WITHIN this block's base class (see derivation).
            const uint32_t col_rot = (block_id / wim_rot_period) % block_width_tiles;
            const uint32_t row_rot = block_id % rows_this_batch;
#else
            const uint32_t col_rot = block_id % block_width_tiles;
            const uint32_t row_rot = block_id % rows_this_batch;
#endif

#if WRITE_ISSUE_MODE == WIM_FLUSH_HALF || WRITE_ISSUE_MODE == WIM_BARRIER_HALF
            const uint32_t wim_half = pages_this_batch >> 1;
            uint32_t wim_issued = 0;
#endif
#if WRITE_ISSUE_MODE == WIM_CMDBUF2 || WRITE_ISSUE_MODE == WIM_CMDBUF4
            uint32_t wim_buf_idx = 0;
#endif
            {
                MaybeDeviceZoneScope("writer_issue");
                for (uint32_t rs = 0; rs < rows_this_batch; ++rs) {
                    uint32_t r = rs + row_rot;
                    if (r >= rows_this_batch) {
                        r -= rows_this_batch;
                    }
                    const uint32_t page_base = (row_start + rows_done + r) * tensor_col_tiles + col_base;
                    const uint32_t l1_row_addr = l1_read_base + r * block_width_tiles * out_tile_bytes;
                    for (uint32_t cs = 0; cs < block_width_tiles; ++cs) {
                        uint32_t i = cs + col_rot;
                        if (i >= block_width_tiles) {
                            i -= block_width_tiles;
                        }
#ifdef TILIZE_ABLATE_WRITES
                        (void)out_acc.get_noc_addr(page_base + i);
#else
                        const uint32_t wim_src = l1_row_addr + i * out_tile_bytes;
                        const uint64_t wim_dst = out_acc.get_noc_addr(page_base + i);
#if WRITE_ISSUE_MODE == WIM_CMDBUF2 || WRITE_ISSUE_MODE == WIM_CMDBUF4
                        {
                            // `noc_async_write_one_packet` with the command
                            // buffer opened up. Everything else (VC, non-mcast,
                            // non-linked, num_dests=1, non-posted, the
                            // per-NOC issued/acked counters that the barrier
                            // reads) is exactly what that function passes.
                            const uint32_t wim_buf = WIM_CMD_BUFS[wim_buf_idx];
                            wim_buf_idx = (wim_buf_idx + 1 == WIM_NUM_CMD_BUFS) ? 0 : wim_buf_idx + 1;
                            DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc_index, wim_dst, wim_src, out_tile_bytes);
                            while (!noc_cmd_buf_ready(noc_index, wim_buf));
                            ncrisc_noc_fast_write<noc_mode>(
                                noc_index,
                                wim_buf,
                                wim_src,
                                wim_dst,
                                out_tile_bytes,
                                NOC_UNICAST_WRITE_VC,
                                false /* mcast */,
                                false /* linked */,
                                1 /* num_dests */,
                                true /* multicast_path_reserve */,
                                false /* posted */);
                        }
#elif WRITE_ISSUE_MODE == WIM_POSTED
                        noc_async_write<out_tile_bytes, true /* tracing */, true /* posted */>(
                            wim_src, wim_dst, out_tile_bytes);
#else
                        noc_async_write<out_tile_bytes>(wim_src, wim_dst, out_tile_bytes);
#endif
#endif  // TILIZE_ABLATE_WRITES
#if WRITE_ISSUE_MODE == WIM_FLUSH_HALF
                        if (++wim_issued == wim_half && wim_half != 0) {
                            noc_async_writes_flushed();
                        }
#elif WRITE_ISSUE_MODE == WIM_BARRIER_HALF
                        if (++wim_issued == wim_half && wim_half != 0) {
                            noc_async_write_barrier();
                        }
#endif
                    }
                }
            }

            {
                MaybeDeviceZoneScope("writer_barrier");
#if WRITE_ISSUE_MODE == WIM_POSTED
                noc_async_posted_writes_flushed();
#elif WRITE_ISSUE_MODE == WIM_FLUSH_ONLY
                // SENT, not ACKED — all `cb_pop_front` below needs. The ACK is
                // still collected, once, by the barrier after the block loop.
                noc_async_writes_flushed();
#else
                noc_async_write_barrier();
#endif
            }
            cb_pop_front(cb_output_tiles, pages_this_batch);
            rows_done += rows_this_batch;
        }
    }

#if WRITE_ISSUE_MODE == WIM_FLUSH_ONLY
    // The ONE real barrier: every write this core issued is acked (landed)
    // before the kernel returns, so program completion still implies a
    // complete output buffer.
    {
        MaybeDeviceZoneScope("writer_final_barrier");
        noc_async_write_barrier();
    }
#endif
}
