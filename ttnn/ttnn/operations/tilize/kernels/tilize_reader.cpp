// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize reader (NCRISC / NoC0).
//
// Modes, selected by compile-time args:
//
//   alias_mode == 1  (Path B, zero-copy)
//       cb_rm_input is aliased onto the resident L1 ROW_MAJOR shard, so the
//       bytes are already at the CB's address. One cb_reserve_back /
//       cb_push_back arms the whole shard; there is no NoC traffic at all.
//
//   alias_mode == 0, split_read == 0  (Path A / C, single reader)
//       Chunk-outer, tile-row-inner. For each column chunk we hand a whole
//       tile-row band to dataflow_kernel_lib::read_sticks_for_tilize in TILE
//       granularity, which owns cb_reserve_back / 32 strided reads / one
//       noc_async_read_barrier / cb_push_back per block. `stateful_read`
//       selects StickReadMode::Stateful (lever B13) inside the helper.
//
//   alias_mode == 0, split_read == 1  (Path A / C, split reader — lever C7)
//       The 32 stick reads of each block are shared with BRISC (the writer
//       kernel), which parks in cb_wait_front for the whole read window
//       otherwise. NCRISC keeps sole ownership of the CB — a circular buffer
//       must have exactly ONE producer, so BRISC never reserves or pushes
//       cb_rm_input; it is handed the reserved window through two counting
//       semaphores:
//           NCRISC: reserve -> sem_reserve = blk+1 -> read half -> barrier
//                   -> wait sem_done >= blk+1 -> push
//           BRISC : wait sem_reserve >= blk+1 -> read half -> barrier
//                   -> sem_done = blk+1
//       Both semaphores are monotonic per-launch counters and both live in this
//       core's own L1, so set/wait are plain local loads and stores (no NoC
//       round trip). Requires depth == 1 so the reserved window is always the
//       CB base address, which is what BRISC's untouched get_write_ptr returns
//       (see the host gate in tilize_program_descriptor.py).
//
//   alias_mode == 0, prefetch_blocks == 2  (Path A / C — lever B8)
//       Trid double-issue. Each chunk-block's 32 stick reads carry one of two NoC
//       transaction ids, and the barrier is `noc_async_read_barrier_with_trid` on
//       the PREVIOUS id — so block i+1's reads are already in flight while block
//       i drains, instead of the NoC request queue emptying once per block. This
//       is the read-side analogue of the write side's `noc_async_writes_flushed()`
//       (Phase-0 verification fix #1): the writer can let writes stay in flight
//       because it only needs them to have DEPARTED, whereas the reader needs the
//       bytes PRESENT before it pushes, which is what the second trid buys.
//
//       NB the host gate keys on the BUSIEST core's block count, so on an uneven
//       split (`_split_contiguous` gives `total % parts` cores one extra unit) some
//       cores run this path with `total_blocks == 1`. That is safe by construction —
//       the prologue reserve covers the single push, the barrier parity matches the
//       trid the prologue set, and the tag is restored — and it is covered by
//       `test_tilize_refinement2.py::test_b8_is_bit_exact_on_an_uneven_split`.
//
//       It needs a THIRD CB window. `cb_reserve_back` does not move the write
//       pointer, so `get_write_ptr` returns the *current* block's window until
//       `cb_push_back` — the reader cannot ask the CB for the next block's address
//       before publishing the current one. The next window is therefore computed
//       from the CB base (`cb_base + (block % depth) * chunk_bytes`, exactly what
//       the FIFO's own pointer does after `depth` pushes) and its freedom is
//       guaranteed by reserving TWO windows. At depth 2 that reserve would demand
//       a fully drained CB and serialize compute behind the reader, hence
//       depth == 3 (host gate).
//
//   vc_spread == 1  (lever B10)
//       Program a per-core static unicast VC for this core's reads. In
//       DM_DEDICATED_NOC — what ReaderConfigDescriptor selects —
//       `noc_async_read`'s `read_req_vc` argument is DEAD: `ncrisc_noc_fast_read`
//       only writes NOC_CTRL under DM_DYNAMIC_NOC
//       (noc_nonblocking_api.h:415-437). NOC_CTRL is instead programmed once by
//       `noc_init` (static VC 1) and is STICKY, so one
//       `noc_async_read_one_packet_set_state<use_vc=true>` retargets every
//       subsequent read on this core — and must be undone before the kernel ends,
//       or the next program on this core inherits the custom VC and loses DRAM
//       bandwidth (same hazard the dram-sharded matmul reader documents).
//
//       When the source is ROW_MAJOR-*sharded* with more than one page per
//       logical row (`row_page_stride > 1`) neither helper path can be used:
//       their page index advances by exactly 1 per row, hard-coding "one page ==
//       one full logical row", and the signature exposes no row-stride
//       parameter. The raw fallback below mirrors the helper's block structure
//       exactly (reserve chunk_wt, 32 reads, one barrier, push chunk_wt) so
//       lever B7 (one barrier per block) still holds.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_rm_input = 0;
    constexpr uint32_t tile_height = 32;  // rows per tile-row block

    constexpr uint32_t alias_mode = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_wt = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t row_page_stride = get_compile_time_arg_val(3);
    constexpr uint32_t source_page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t shard_tiles = get_compile_time_arg_val(5);
    // Perf-ablation only (TILIZE_SKIP_DM=1): drop the noc_async_read *payload* and
    // keep every CB op, barrier, handshake and loop trip count, so /perf-measure
    // can attribute time to the read stage. Never set on a correctness run.
    constexpr uint32_t skip_dm = get_compile_time_arg_val(6);
    constexpr uint32_t stateful_read = get_compile_time_arg_val(7);  // lever B13
    constexpr uint32_t split_read = get_compile_time_arg_val(8);     // lever C7
    constexpr uint32_t sem_reserve_id = get_compile_time_arg_val(9);
    constexpr uint32_t sem_done_id = get_compile_time_arg_val(10);
    constexpr uint32_t prefetch_blocks = get_compile_time_arg_val(11);  // lever B8
    constexpr uint32_t vc_spread = get_compile_time_arg_val(12);        // lever B10 (bitmask)
    constexpr bool read_vc_spread = (vc_spread & 1u) != 0;              // bit 0 == spread the reads
    constexpr uint32_t cb_depth = get_compile_time_arg_val(13);
    constexpr uint32_t trid_a = get_compile_time_arg_val(14);
    constexpr uint32_t trid_b = get_compile_time_arg_val(15);
    constexpr uint32_t default_read_vc = get_compile_time_arg_val(16);
    constexpr auto src_args = TensorAccessorArgs<17>();

    using dataflow_kernel_lib::StickReadMode;
    constexpr StickReadMode read_mode = stateful_read ? StickReadMode::Stateful : StickReadMode::Generic;
    static_assert(!split_read || row_page_stride == 1, "the split reader needs one source page per logical row");
    static_assert(prefetch_blocks == 1 || prefetch_blocks == 2, "B8 double-issues exactly two transaction ids");
    static_assert(prefetch_blocks == 1 || !split_read, "B8 and C7 both own the read window; they are exclusive");
    // cb_depth >= 3 is EXACT, not conservative: `cb_reserve_back(2 * chunk_wt)`
    // guarantees blocks 0..b-(depth-2) are popped, and block b+1's window last held
    // block b+1-depth, so depth 3 gives precisely the needed guarantee with zero
    // margin. Do not relax it (any depth > 3 is also sound).
    static_assert(prefetch_blocks == 1 || cb_depth >= 3, "B8 needs a third CB window (see the header)");
    // The host would otherwise size the CB to 3 windows and then silently get the
    // raw strided fallback (correct output, lever quietly lost, no diagnostic).
    static_assert(prefetch_blocks == 1 || row_page_stride == 1, "B8 needs one source page per logical row");

    if constexpr (alias_mode) {
        // Data is already resident at the CB address — just hand it to compute.
        cb_reserve_back(cb_rm_input, shard_tiles);
        cb_push_back(cb_rm_input, shard_tiles);
        return;
    } else {
        const uint32_t src_addr = get_arg_val<uint32_t>(0);
        const uint32_t start_row = get_arg_val<uint32_t>(1);
        const uint32_t num_rows = get_arg_val<uint32_t>(2);
        const uint32_t chunk_start = get_arg_val<uint32_t>(3);
        const uint32_t chunk_count = get_arg_val<uint32_t>(4);

        const auto accessor = TensorAccessor(src_args, src_addr);

        // --- lever B10: retarget this core's reads onto its own static VC ------
        // NOC_CTRL is sticky and `noc_async_read` never rewrites it in dedicated
        // mode, so one armed set_state moves every read below onto `read_vc`.
        if constexpr (read_vc_spread) {
            const uint32_t read_vc = get_arg_val<uint32_t>(5);
            noc_async_read_one_packet_set_state<true>(accessor.get_noc_addr(start_row), chunk_row_bytes, read_vc);
        }

        if constexpr (row_page_stride == 1 && prefetch_blocks == 2) {
            // --- lever B8: trid double-issue over the whole (chunk, block) run --
            // The sequence is flattened so the pipeline spans chunk boundaries too;
            // the order stays chunk-outer / tile-row-inner, which is what the
            // writer and compute both assume.
            constexpr uint32_t tile_bytes = get_tile_size(cb_rm_input);
            constexpr uint32_t window_bytes = chunk_wt * tile_bytes;
            const uint32_t blocks_per_chunk = num_rows / tile_height;
            const uint32_t total_blocks = chunk_count * blocks_per_chunk;
            // The FIFO write pointer starts at the CB base and advances exactly one
            // window per `cb_push_back(chunk_wt)`, wrapping after cb_depth pushes, so
            // window w is always base + w*window_bytes. Read BEFORE the first reserve
            // on purpose: the firmware re-runs
            // setup_local_cb_read_write_interfaces() at the top of every launch
            // (ncrisc.cc), so fifo_wr_ptr == fifo_addr here regardless of how many
            // pushes the PREVIOUS launch of this cached program made.
            const uint32_t cb_base = get_write_ptr(cb_rm_input);

            // Two windows free => the one block 0 lands in, and the one block 1
            // will land in before block 0 is published.
            cb_reserve_back(cb_rm_input, 2 * chunk_wt);
            noc_async_read_set_trid(trid_a);
            if constexpr (!skip_dm) {
                dataflow_kernel_lib::read_stick_rows_for_tilize<StickReadMode::Generic, 1>(
                    accessor,
                    start_row,
                    chunk_row_bytes,
                    chunk_start * chunk_row_bytes,
                    cb_base,
                    chunk_row_bytes,
                    tile_height);
            }

            for (uint32_t block = 0; block < total_blocks; ++block) {
                if (block + 1 < total_blocks) {
                    const uint32_t nxt = block + 1;
                    const uint32_t nc = nxt / blocks_per_chunk;
                    const uint32_t nr = nxt - nc * blocks_per_chunk;
                    noc_async_read_set_trid((nxt & 1u) ? trid_b : trid_a);
                    if constexpr (skip_dm) {
                        // Ablation: the payload goes, the address generation stays.
                        // The non-prefetched fallback below keeps its 32 accessor
                        // calls behind a volatile sink for exactly this reason, and
                        // Refinement 1 priced address-gen at ~437 ns of 3 609 ns on
                        // `d_tall_narrow` — dropping it here would bias every
                        // skip_dm A/B in this lever's favour by that whole term.
                        for (uint32_t row = 0; row < tile_height; ++row) {
                            volatile uint32_t sink = static_cast<uint32_t>(accessor.get_noc_addr(
                                start_row + nr * tile_height + row, (chunk_start + nc) * chunk_row_bytes));
                            (void)sink;
                        }
                    } else {
                        dataflow_kernel_lib::read_stick_rows_for_tilize<StickReadMode::Generic, 1>(
                            accessor,
                            start_row + nr * tile_height,
                            chunk_row_bytes,
                            (chunk_start + nc) * chunk_row_bytes,
                            cb_base + (nxt % cb_depth) * window_bytes,
                            chunk_row_bytes,
                            tile_height);
                    }
                }
                noc_async_read_barrier_with_trid((block & 1u) ? trid_b : trid_a);
                cb_push_back(cb_rm_input, chunk_wt);
                if (block + 2 < total_blocks) {
                    // Guarantees window (block+2) % cb_depth carries no unpopped
                    // data before the next iteration issues into it.
                    cb_reserve_back(cb_rm_input, 2 * chunk_wt);
                }
            }
            // NOC_PACKET_TAG is sticky across kernel launches -- hand the cmd buf
            // back with the firmware's default tag.
            noc_async_read_set_trid(0);
            if constexpr (read_vc_spread) {
                noc_async_read_one_packet_set_state<true>(
                    accessor.get_noc_addr(start_row), chunk_row_bytes, default_read_vc);
            }
            return;
        }

        for (uint32_t c = 0; c < chunk_count; ++c) {
            const uint32_t byte_offset = (chunk_start + c) * chunk_row_bytes;

            if constexpr (row_page_stride == 1 && !split_read && !skip_dm) {
                dataflow_kernel_lib::
                    read_sticks_for_tilize<cb_rm_input, dataflow_kernel_lib::TilizeGranularity::TILE, read_mode>(
                        accessor, num_rows, chunk_row_bytes, start_row, byte_offset);
            } else if constexpr (row_page_stride == 1 && split_read) {
                // Lever C7. The CB dance stays here (single producer); the row
                // band is split with BRISC by bank group inside the helper.
                volatile tt_l1_ptr uint32_t* sem_reserve =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_reserve_id));
                volatile tt_l1_ptr uint32_t* sem_done =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_done_id));
                const uint32_t blocks = num_rows / tile_height;

                for (uint32_t block = 0; block < blocks; ++block) {
                    const uint32_t first_page = start_row + block * tile_height;
                    cb_reserve_back(cb_rm_input, chunk_wt);
                    const uint32_t l1_addr = get_write_ptr(cb_rm_input);
                    // The window is free: hand it to BRISC. Sequence numbers are
                    // per (chunk, block) so they stay monotonic across chunks.
                    const uint32_t seq = c * blocks + block + 1;
                    noc_semaphore_set(sem_reserve, seq);

                    if constexpr (!skip_dm) {
                        dataflow_kernel_lib::read_stick_rows_for_tilize<read_mode, 2>(
                            accessor,
                            first_page,
                            chunk_row_bytes,
                            byte_offset,
                            l1_addr,
                            chunk_row_bytes,
                            tile_height,
                            /*split_id=*/0);
                    }
                    noc_async_read_barrier();
                    noc_semaphore_wait_min(sem_done, seq);
                    cb_push_back(cb_rm_input, chunk_wt);
                }
            } else {
                // A chunk never straddles a source page (host guarantees
                // chunk_row_bytes divides source_page_bytes), so the whole
                // chunk lives in one page at a fixed intra-page offset.
                const uint32_t page_col = byte_offset / source_page_bytes;
                const uint32_t offset_in_page = byte_offset - page_col * source_page_bytes;
                const uint32_t blocks = num_rows / tile_height;

                for (uint32_t block = 0; block < blocks; ++block) {
                    const uint32_t row0 = start_row + block * tile_height;
                    cb_reserve_back(cb_rm_input, chunk_wt);
                    uint32_t l1_addr = get_write_ptr(cb_rm_input);
                    for (uint32_t row = 0; row < tile_height; ++row) {
                        const uint64_t noc_addr =
                            accessor.get_noc_addr((row0 + row) * row_page_stride + page_col, offset_in_page);
                        if constexpr (skip_dm) {
                            // Ablation: keep the address-gen observable so dead-code
                            // elimination cannot delete the loop being timed.
                            volatile uint32_t sink = static_cast<uint32_t>(noc_addr);
                            (void)sink;
                        } else {
                            noc_async_read(noc_addr, l1_addr, chunk_row_bytes);
                        }
                        l1_addr += chunk_row_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_rm_input, chunk_wt);
                }
            }
        }

        if constexpr (read_vc_spread) {
            // Restore the firmware default before exiting -- NOC_CTRL survives the
            // kernel launch and the next program on this core will not re-set it.
            noc_async_read_one_packet_set_state<true>(
                accessor.get_noc_addr(start_row), chunk_row_bytes, default_read_vc);
        }
    }
}
