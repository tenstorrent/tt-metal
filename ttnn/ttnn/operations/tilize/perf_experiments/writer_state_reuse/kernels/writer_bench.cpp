// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// writer_state_reuse bench — the writer (BRISC), isolated.
//
// Reconstructs tilize's `store_block` verbatim as VARIANT 0 (`BASELINE`) and
// adds four candidates that change ONLY how the per-page NoC write command is
// programmed. The block-resolution math, the CB wait/pop and the barrier are
// IDENTICAL across every variant — the isolation the coordinator asked for.
//
// VARIANT 0 BASELINE — today's op, unchanged:
//     noc_async_write<out_tile_bytes>(l1_addr, out_acc.get_noc_addr(page), out_tile_bytes)
//   `out_acc.get_noc_addr` re-derives (bank_id, bank_page_offset) from `page`
//   via mod/div EVERY call, and `noc_async_write` (one-packet path) reprograms
//   all 6 NOC command-buffer registers EVERY call (NOC_CTRL, TARG_ADDR_LO,
//   RET_ADDR_LO, RET_ADDR_COORDINATE, AT_LEN_BE, CMD_CTRL) — see
//   `ncrisc_noc_fast_write` in noc_nonblocking_api.h.
//
// VARIANT 1 SET_STATE_NAIVE — the literal reading of the assigned idea:
//   `noc_async_write_one_packet_set_state()` ONCE per row, then
//   `noc_async_write_one_packet_with_state()` for the row's other pages.
//   THIS IS INCORRECT for an interleaved DRAM destination and is kept here
//   ONLY to make that concrete and measured, not as a candidate to graduate.
//   `ncrisc_noc_write_set_state` programs `NOC_RET_ADDR_COORDINATE` from the
//   FIRST page's address and `ncrisc_noc_write_with_state` NEVER reprograms
//   it (noc_nonblocking_api.h:1246-1317) — it only rewrites TARG_ADDR_LO,
//   RET_ADDR_LO and re-sends. `TensorAccessor` round-robins DRAM banks by
//   `page_id % num_banks` (tensor_accessor.h:324-326), so on THIS box
//   (12 banks) and every block_width_tiles < 12 plan (every shape measured
//   here), consecutive pages in a row land on 12 DISTINCT banks — the
//   coordinate is stale for every page after the first. The data is written
//   at the CORRECT low address but to the WRONG bank/core: silent data
//   corruption, not a hang. Verified by the bit-identity gate failing.
//
// VARIANT 2 ADDR_RECURRENCE — candidate 3 in the menu. Same full
//   `noc_async_write<out_tile_bytes>` command path as BASELINE (still
//   reprograms all 6 registers, still correct for any bank sequence), but the
//   address is no longer re-derived per page. CORRECTION vs the assignment's
//   framing: a plain (non-sharded) interleaved DRAM tensor does NOT go
//   through the generic `TensorAccessor::get_bank_and_offset` (mod/div on a
//   RUNTIME bank count) — `TensorAccessor(out_args, dst_addr)` resolves, via
//   partial specialization on `IsInterleaved`
//   (tensor_accessor.h:379-394), to a type that PUBLICLY INHERITS
//   `InterleavedAddrGen<IsDram>` (dataflow_api_addrgen.h:284-312) instead, and
//   its `get_noc_addr` already divides by the COMPILE-TIME-constant
//   `NUM_DRAM_BANKS` via `udivsi3_const_divisor` (a magic-multiply reciprocal,
//   not a division loop) when banks aren't a power of 2 (12 here), or a shift
//   when they are. So the per-page cost this variant removes is smaller than
//   a naive division would be: ONE `get_bank_offset_index`/`get_bank_index`
//   pair (magic-multiply + subtract) seeds `(bank_offset_index, bank_index)`
//   for the row's first page, and each subsequent page steps it with
//   `bank_index+1, wrap->bank_offset_index+1` — exact for consecutive integer
//   page ids, which is what a row of tile columns always is. This trades N
//   magic-multiply divisions + N bank-to-xy table lookups for 1 of each plus
//   (N-1) compare-and-maybe-increments; the table lookup
//   (`interleaved_addr_gen::get_noc_xy`) still happens every call because the
//   bank identity itself still changes every call (see variant 1's finding).
//
// VARIANT 3 COORD_REUSE_RAW — candidate 5 ("anything else the measurement
//   points at"). RAW-LLK, bypassing `noc_async_write_one_packet_set_state`
//   itself. `NOC_CTRL` (cmd field: copy+write+static-VC+resp-marked) and
//   `NOC_AT_LEN_BE` (size) are IDENTICAL on every single write this kernel
//   ever issues — same VC, same `posted=false`, same `out_tile_bytes` — for
//   the kernel's ENTIRE run, not just one row. Program them ONCE at kernel
//   start via `noc_async_write_one_packet_set_state`, then per page write
//   ONLY `NOC_RET_ADDR_COORDINATE` by hand (the part that DOES change every
//   call, unlike variant 1's bug) and issue through
//   `noc_async_write_one_packet_with_state` (TARG_ADDR_LO, RET_ADDR_LO,
//   CMD_CTRL). Net: 4 register writes/call instead of BASELINE's 6, and
//   correct for every bank sequence because the coordinate is still
//   reprogrammed every time. The helper bypassed is
//   `noc_async_write_one_packet_set_state` itself (it only exposes
//   "reprogram everything" or "reprogram nothing"); the gap is ERGONOMICS,
//   not a capability gap — the raw NOC_CMD_BUF_WRITE_REG this uses is the
//   exact primitive `ncrisc_noc_write_set_state` is built from.
//
// VARIANT 4 COMBINED — variant 2's address recurrence feeding variant 3's
//   coordinate-only reuse: 1 division per row, 4 registers per page.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {
constexpr uint32_t VARIANT_BASELINE = 0;
constexpr uint32_t VARIANT_SET_STATE_NAIVE = 1;
constexpr uint32_t VARIANT_ADDR_RECURRENCE = 2;
constexpr uint32_t VARIANT_COORD_REUSE_RAW = 3;
constexpr uint32_t VARIANT_COMBINED = 4;
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_row_blocks = get_compile_time_arg_val(2);  // R
    constexpr uint32_t tensor_col_tiles = get_compile_time_arg_val(3);   // C
    constexpr uint32_t num_row_groups = get_compile_time_arg_val(4);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t write_rows_per_barrier = get_compile_time_arg_val(6);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t col_tile_offset = get_compile_time_arg_val(8);
    constexpr uint32_t variant = get_compile_time_arg_val(9);
    // NOTE on attribution: rather than nested `writer_addrgen`/`writer_cmd`
    // sub-zones (marker-budget risk on any shape with more than a handful of
    // pages per core — see device-zone-scope-attribution.md), the two
    // candidate costs named in the assignment are separated by VARIANT
    // CONTRAST instead: ADDR_RECURRENCE removes ONLY the per-page divide,
    // COORD_REUSE_RAW removes ONLY the per-page CTRL/LEN reprogram, and
    // COMBINED removes both. `writer_issue`'s delta between these and
    // BASELINE attributes each cost cleanly, at zero extra marker pressure,
    // on every shape in the domain sweep.
    constexpr auto out_args = TensorAccessorArgs<10>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block_id = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);
    const uint32_t block_stride = get_arg_val<uint32_t>(3);

    const auto out_acc = TensorAccessor(out_args, dst_addr);

    // VARIANT 3/4 raw-LLK state: CTRL + LEN + VC programmed ONCE, for the
    // kernel's whole run (every write is out_tile_bytes, VC NOC_UNICAST_WRITE_VC,
    // posted=false — see the file header). The seed address's coordinate bits
    // are dead on arrival (overwritten per-page below) so any valid address
    // works; page 0 is the natural, always-valid choice.
    if constexpr (variant == VARIANT_COORD_REUSE_RAW || variant == VARIANT_COMBINED) {
        const uint32_t seed_page = col_tile_offset;  // block (0,0)'s first column
        noc_async_write_one_packet_set_state(out_acc.get_noc_addr(seed_page), out_tile_bytes);
    }

    for (uint32_t b = 0; b < num_blocks_this_core; ++b) {
        const uint32_t block_id = start_block_id + b * block_stride;
        const uint32_t row_group = block_id / num_w_chunks;
        const uint32_t w_chunk = block_id - row_group * num_w_chunks;

        const uint32_t row_start = (row_group * tensor_row_blocks) / num_row_groups;
        const uint32_t row_end = ((row_group + 1) * tensor_row_blocks) / num_row_groups;
        const uint32_t block_row_extent = row_end - row_start;
        const uint32_t col_base = col_tile_offset + w_chunk * block_width_tiles;

        uint32_t rows_done = 0;
        while (rows_done < block_row_extent) {
            uint32_t rows_this_batch = block_row_extent - rows_done;
            if (rows_this_batch > write_rows_per_barrier) {
                rows_this_batch = write_rows_per_barrier;
            }
            {
                const LocalCBInterface& cb = get_local_cb_interface(cb_out);
                const uint32_t contig_rows = ((cb.fifo_limit - cb.fifo_rd_ptr) / cb.fifo_page_size) / block_width_tiles;
                if (rows_this_batch > contig_rows) {
                    rows_this_batch = contig_rows;
                }
            }
            const uint32_t pages_this_batch = rows_this_batch * block_width_tiles;

            {
                MaybeDeviceZoneScope("writer_wait_out");
                cb_wait_front(cb_out, pages_this_batch);
            }

            uint32_t l1_read_addr = get_read_ptr(cb_out);
            {
                MaybeDeviceZoneScope("writer_issue");
                for (uint32_t r = 0; r < rows_this_batch; ++r) {
                    const uint32_t page_base = (row_start + rows_done + r) * tensor_col_tiles + col_base;

                    if constexpr (variant == VARIANT_BASELINE) {
                        for (uint32_t i = 0; i < block_width_tiles; ++i) {
                            noc_async_write<out_tile_bytes>(
                                l1_read_addr, out_acc.get_noc_addr(page_base + i), out_tile_bytes);
                            l1_read_addr += out_tile_bytes;
                        }

                    } else if constexpr (variant == VARIANT_SET_STATE_NAIVE) {
                        // The idea AS STATED: program state from page 0, reuse
                        // it for the rest of the row. See file header — this
                        // is expected to write the WRONG bank for i > 0: the
                        // coordinate programmed by set_state (page 0's bank)
                        // is never touched again, while with_state's low
                        // address bits keep advancing to each page's OWN
                        // (different-bank) offset.
                        const uint64_t addr0 = out_acc.get_noc_addr(page_base);
                        noc_async_write_one_packet_set_state(addr0, out_tile_bytes);
                        noc_async_write_one_packet_with_state(l1_read_addr, (uint32_t)addr0);
                        l1_read_addr += out_tile_bytes;
                        for (uint32_t i = 1; i < block_width_tiles; ++i) {
                            const uint64_t full_addr = out_acc.get_noc_addr(page_base + i);
                            noc_async_write_one_packet_with_state(l1_read_addr, (uint32_t)full_addr);
                            l1_read_addr += out_tile_bytes;
                        }

                    } else if constexpr (variant == VARIANT_ADDR_RECURRENCE) {
                        // ONE magic-multiply divide (`get_bank_offset_index`) for the
                        // whole row, seeding (bank_offset_index, bank_index); see the
                        // corrected file-header note on why this is not a naive divide.
                        constexpr bool kIsDram = true;  // this bench's output is always DRAM interleaved
                        uint32_t boi = interleaved_addr_gen::get_bank_offset_index<kIsDram>(page_base);
                        uint32_t bank = interleaved_addr_gen::get_bank_index<kIsDram>(page_base, boi);
                        for (uint32_t i = 0; i < block_width_tiles; ++i) {
                            const uint32_t local_addr = out_acc.get_addr(/*id (unused)*/ 0, boi, bank, /*offset*/ 0);
                            const uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<kIsDram>(bank, noc_index);
                            const uint64_t addr = get_noc_addr_helper(noc_xy, local_addr);
                            noc_async_write<out_tile_bytes>(l1_read_addr, addr, out_tile_bytes);
                            l1_read_addr += out_tile_bytes;
                            if (++bank >= NUM_DRAM_BANKS) {
                                bank = 0;
                                ++boi;
                            }
                        }

                    } else if constexpr (variant == VARIANT_COORD_REUSE_RAW) {
                        for (uint32_t i = 0; i < block_width_tiles; ++i) {
                            const uint64_t full_addr = out_acc.get_noc_addr(page_base + i);
                            // MUST wait for the cmd buf to have fully drained the PRIOR
                            // transaction before mutating any of its registers -- this is
                            // the same guard `ncrisc_noc_write_set_state`/`_with_state`
                            // open with (noc_nonblocking_api.h:1248,1300). Skipping it
                            // races the hardware's read of the in-flight command's fields
                            // and HUNG THE DEVICE in this bench's first measured run (see
                            // file header) -- not a perf shortcut, a correctness bug.
                            while (!noc_cmd_buf_ready(noc_index, write_cmd_buf));
                            NOC_CMD_BUF_WRITE_REG(
                                noc_index,
                                write_cmd_buf,
                                NOC_RET_ADDR_COORDINATE,
                                (uint32_t)(full_addr >> NOC_ADDR_COORD_SHIFT));
                            noc_async_write_one_packet_with_state(l1_read_addr, (uint32_t)full_addr);
                            l1_read_addr += out_tile_bytes;
                        }

                    } else if constexpr (variant == VARIANT_COMBINED) {
                        constexpr bool kIsDram = true;
                        uint32_t boi = interleaved_addr_gen::get_bank_offset_index<kIsDram>(page_base);
                        uint32_t bank = interleaved_addr_gen::get_bank_index<kIsDram>(page_base, boi);
                        for (uint32_t i = 0; i < block_width_tiles; ++i) {
                            const uint32_t local_addr = out_acc.get_addr(0, boi, bank, 0);
                            const uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<kIsDram>(bank, noc_index);
                            const uint64_t addr = get_noc_addr_helper(noc_xy, local_addr);
                            while (!noc_cmd_buf_ready(noc_index, write_cmd_buf));
                            NOC_CMD_BUF_WRITE_REG(
                                noc_index,
                                write_cmd_buf,
                                NOC_RET_ADDR_COORDINATE,
                                (uint32_t)(addr >> NOC_ADDR_COORD_SHIFT));
                            noc_async_write_one_packet_with_state(l1_read_addr, (uint32_t)addr);
                            l1_read_addr += out_tile_bytes;
                            if (++bank >= NUM_DRAM_BANKS) {
                                bank = 0;
                                ++boi;
                            }
                        }
                    }
                }
            }

            {
                MaybeDeviceZoneScope("writer_barrier");
                noc_async_write_barrier();
            }
            cb_pop_front(cb_out, pages_this_batch);
            rows_done += rows_this_batch;
        }
    }
}
