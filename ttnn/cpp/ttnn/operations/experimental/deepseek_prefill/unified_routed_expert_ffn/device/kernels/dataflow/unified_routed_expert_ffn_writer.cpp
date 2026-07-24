// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for unified_routed_expert_ffn.
//
// Two responsibilities, both handled by this writer kernel:
//
// 1. Output drain + placement. Pop `cb_out` (the down matmul's per-core
//    final block, packed one subblock at a time) and write tiles to the
//    DRAM-interleaved output tensor at this core's (mt, nt_d) tile region,
//    looped over `effective_chunks` chunks (computed from the device-side
//    counts/idx scratch CBs). Activated tiles are distributed across the
//    M-row by the READER via L1 multicast — no DRAM scratch round-trip or
//    cross-core barrier. Placement has two modes (the `direct_write` CT flag):
//      * direct_write == 0: writes start at tile row 0. The FFN op writes to
//        a per-expert output tensor; a separate ttnn::insert handles
//        placement into any shared destination buffer.
//      * direct_write == 1: this expert's output is written directly into a
//        shared destination buffer at the expert's region offset. The kernel
//        reads start[global_expert_id] from `start` (= expert_region_offsets)
//        device-side and adds (start / TILE_HEIGHT) tile-rows to every output
//        row — fusing what ttnn::insert would otherwise do (no temp-buffer
//        DRAM round-trip).
//
// 2. Two-RISC `up`-weight read (UP_SPLIT). The writer (NCRISC) reads `up`
//    from DRAM on NoC 1 concurrent with the reader's NoC-0 `gate` read. The
//    program factory selects UP_SPLIT (writer_split_up) for ALL layouts: the
//    writer reads `up` into the gy=0 sender's cb_in1_up slot and the reader
//    multicasts it on NoC 0, ordered by a local up_go/up_done handshake. Only
//    a NoC-1 DRAM read happens here — no worker multicast and no NoC-1 atomics
//    — so it is safe beside the fabric CCL ops. The legacy writer-side NoC-1
//    multicast mode (UP_WRITER_MCAST / writer_mcasts_up) is retired and never
//    selected. Per chunk the writer produces all `up` K-blocks, then drains
//    `cb_out`.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/debug/assert.h"

constexpr uint32_t TILE_HEIGHT = 32;

void kernel_main() {
    Noc noc;

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_mt = get_arg_val<uint32_t>(1);
    const uint32_t my_nt_d = get_arg_val<uint32_t>(2);
    const uint32_t start_addr = get_arg_val<uint32_t>(3);
    // UP_SPLIT up-weight read args. Arg 4 holds expert-0's up base (kept for
    // arg-layout stability); the per-expert `up` addresses live in the array
    // starting at UP_RT and are used instead.
    const uint32_t my_nt_gu = get_arg_val<uint32_t>(5);
    const bool is_up_sender = get_arg_val<uint32_t>(6) != 0;
    // UP_SPLIT local handshake sems (see reader): up_go = slot reserved,
    // up_done = up landed.
    const uint32_t up_go_sem_id = get_arg_val<uint32_t>(7);
    const uint32_t up_done_sem_id = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(2);
    constexpr uint32_t per_core_N_gu = get_compile_time_arg_val(3);
    constexpr uint32_t per_core_N_d = get_compile_time_arg_val(4);
    constexpr uint32_t d_out_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t d_out_subblock_w = get_compile_time_arg_val(8);
    constexpr uint32_t N_gate_tiles_full = get_compile_time_arg_val(9);
    constexpr uint32_t N_down_tiles_full = get_compile_time_arg_val(10);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t chunk_M_tiles = get_compile_time_arg_val(12);
    // device-side count read.
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(13);
    constexpr uint32_t cb_idx_scratch = get_compile_time_arg_val(14);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(15);
    // M_tiles_full: total tile-row count of the FFN *input* (x). When the
    // kernel runs more chunks than strictly needed (because
    // M_tiles_full % chunk_M_tiles != 0), the last chunk has writer
    // destinations past M_tiles_full — we skip those source rows here.
    constexpr uint32_t M_tiles_full = get_compile_time_arg_val(16);
    // direct_write CT arg (17) is always 1 now: every expert's output is written
    // into the shared buffer at start[global_id]/TILE tile-rows. Kept in the
    // CT-arg layout for stability but no longer branched on.
    // dst_M_tiles: tile-row count of the *destination* buffer. Equals
    // M_tiles_full in non-direct mode; the shared buffer's row count in
    // direct-write mode (used to bound destination writes).
    constexpr uint32_t dst_M_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t cb_start_scratch = get_compile_time_arg_val(19);
    // UP_SPLIT up-weight read (see header): the writer reads `up` on NoC 1 into
    // the gy=0 sender's cb_in1_up slot; the reader mcasts it on NoC 0.
    // writer_split_up gates it (1 = UP_SPLIT, 0 = LEGACY: reader owns `up`).
    constexpr uint32_t cb_in1_up = get_compile_time_arg_val(20);
    constexpr uint32_t in0_block_w_gu = get_compile_time_arg_val(21);
    constexpr uint32_t K_gate_tiles = get_compile_time_arg_val(22);
    constexpr uint32_t writer_split_up = get_compile_time_arg_val(23);

    constexpr uint32_t d_out_subblock_num_tiles = d_out_subblock_h * d_out_subblock_w;
    constexpr uint32_t d_in1_num_subblocks_M = per_core_M / d_out_subblock_h;
    constexpr uint32_t d_in1_num_subblocks_N = per_core_N_d / d_out_subblock_w;
    constexpr uint32_t num_blocks_gu = K_gate_tiles / in0_block_w_gu;
    constexpr uint32_t g_in1_block_num_tiles = per_core_N_gu * in0_block_w_gu;

    CircularBuffer cb_out_buf(cb_out);
    CircularBuffer cb_counts_scratch_buf(cb_counts_scratch);
    CircularBuffer cb_idx_scratch_buf(cb_idx_scratch);
    CircularBuffer cb_start_scratch_buf(cb_start_scratch);

    // Accessor compile-arg stream order (host appends in this exact order):
    // out, then start (direct-write), then up (UP_SPLIT). The accessors are
    // constructed unconditionally; start_acc is used only when direct_write,
    // up_acc only when writer_split_up.
    constexpr uint32_t out_accessor_offset = 24;
    constexpr auto out_args = TensorAccessorArgs<out_accessor_offset>();
    const auto out_acc = TensorAccessor(out_args, output_addr, cb_out_buf.get_tile_size());

    constexpr uint32_t start_accessor_offset = out_args.next_compile_time_args_offset();
    constexpr auto start_args = TensorAccessorArgs<start_accessor_offset>();
    const auto start_acc = TensorAccessor(start_args, start_addr);

    constexpr uint32_t up_accessor_offset = start_args.next_compile_time_args_offset();
    constexpr auto up_args = TensorAccessorArgs<up_accessor_offset>();
    // Per-expert `up` base addresses live in a runtime-arg array after the fixed
    // writer args; UP_RT is its first index. up_addr(e) = arg[UP_RT + e].
    constexpr uint32_t UP_RT = 9;

    const uint32_t out_tile_bytes = cb_out_buf.get_tile_size();

    // Wait for the reader's counts/idx push (resident scratch). Per expert the
    // writer bounds its cb_out drain by that expert's effective_chunks; the
    // reader and compute bound theirs identically, so the pipeline stays in
    // lockstep and the writer never waits on a chunk compute never pushes.
    cb_counts_scratch_buf.wait_front(1);
    cb_idx_scratch_buf.wait_front(1);
    const volatile tt_l1_ptr uint32_t* counts_ptr =
        reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(cb_counts_scratch_buf.get_read_ptr());
    const uint32_t idx_l1 = cb_idx_scratch_buf.get_read_ptr();
    const volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(idx_l1);

    // Read the `start` (= expert_region_offsets) page ONCE into resident L1;
    // each expert's output slice begins at start[global_id] (token row).
    const uint32_t start_l1 = cb_start_scratch_buf.get_write_ptr();
    {
        const uint32_t start_page_size = start_acc.get_aligned_page_size();
        noc.async_read(start_acc, CoreLocalMem<uint32_t>(start_l1), start_page_size, {.page_id = 0}, {});
        noc.async_read_barrier();
    }
    const volatile tt_l1_ptr uint32_t* start_ptr = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(start_l1);

    // ---- UP_SPLIT up-weight read setup (see header) ----
    // The writer reads `up` from DRAM on NoC 1 concurrent with the reader's
    // NoC-0 `gate` read, into the gy=0 sender's cb_in1_up slot; the reader
    // multicasts it on NoC 0. A local same-core (BRISC reader <-> NCRISC writer)
    // handshake (up_go / up_done, monotonic) orders the two.
    Noc noc_up(1);
    const uint32_t up_tile_bytes = get_tile_size(cb_in1_up);
    Semaphore<> up_go_sem(up_go_sem_id);
    Semaphore<> up_done_sem(up_done_sem_id);
    // up_seq stays in lockstep with the reader ACROSS all experts.
    uint32_t up_seq = 0;

    // ======================= per-local-expert loop =======================
    // Drain every local expert's down-matmul output (and, for UP_SPLIT sender
    // cores, feed its `up` weights) into the shared output buffer at the
    // expert's region offset. The chunk-loop body below is unchanged from the
    // single-expert kernel — only the per-expert bindings differ.
    for (uint32_t local_expert_id = 0; local_expert_id < experts_per_chip; ++local_expert_id) {
        const auto up_acc = TensorAccessor(up_args, get_arg_val<uint32_t>(UP_RT + local_expert_id), up_tile_bytes);
        const uint32_t global_expert_id = idx_ptr[local_expert_id];
        const uint32_t count_value = counts_ptr[global_expert_id];
        const uint32_t count_tiles = (count_value + TILE_HEIGHT - 1) / TILE_HEIGHT;
        const uint32_t effective_chunks = (count_tiles + chunk_M_tiles - 1) / chunk_M_tiles;
        ASSERT(effective_chunks <= num_chunks);
        const uint32_t row_offset_tiles = start_ptr[global_expert_id] / TILE_HEIGHT;

        for (uint32_t chunk = 0; chunk < effective_chunks; ++chunk) {
            // ---- Phase 1/2 weight feed: writer reads `up` on NoC 1 (UP_SPLIT) ----
            // Streams `up` from DRAM concurrent with the reader's NoC-0 `gate` read.
            // Runs before the cb_out drain.
            if constexpr (writer_split_up) {
                // UP_SPLIT: only gy=0 in1-sender cores read `up`. Per K-block: wait
                // for the reader to reserve the slot (up_go), read this column's
                // `up` slice on NoC 1 into it, then signal up_done so the reader
                // mcasts on NoC 0. Only a NoC-1 DRAM read here (fabric-safe); the
                // reader owns cb_in1_up reserve/push.
                if (is_up_sender) {
                    // The CB write pointer is PER-RISC and the reader owns push, so
                    // the writer's get_write_ptr never advances. Replicate the
                    // reader's cadence: cb_in1_up is double-buffered, one push per
                    // K-block, so the live slot is base + (up_seq-1)%2 * slot.
                    constexpr uint32_t kUpNumSlots = 2;
                    CircularBuffer cb_in1_up_buf(cb_in1_up);
                    const uint32_t up_cb_base = cb_in1_up_buf.get_write_ptr();
                    const uint32_t up_slot_bytes = g_in1_block_num_tiles * up_tile_bytes;
                    for (uint32_t kb = 0; kb < num_blocks_gu; ++kb) {
                        ++up_seq;
                        up_go_sem.wait_min(up_seq);
                        uint32_t l1_w_up = up_cb_base + ((up_seq - 1) % kUpNumSlots) * up_slot_bytes;
                        for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
                            for (uint32_t n = 0; n < per_core_N_gu; ++n) {
                                const uint32_t row = kb * in0_block_w_gu + k;
                                const uint32_t col = my_nt_gu * per_core_N_gu + n;
                                if (col < N_gate_tiles_full) {
                                    const uint32_t tile_idx = row * N_gate_tiles_full + col;
                                    noc_up.async_read(
                                        up_acc,
                                        CoreLocalMem<uint32_t>(l1_w_up),
                                        up_tile_bytes,
                                        {.page_id = tile_idx},
                                        {});
                                } else {
                                    volatile tt_l1_ptr uint64_t* p =
                                        reinterpret_cast<volatile tt_l1_ptr uint64_t*>(l1_w_up);
                                    for (uint32_t i = 0; i < up_tile_bytes / 8; ++i) {
                                        p[i] = 0;
                                    }
                                }
                                l1_w_up += up_tile_bytes;
                            }
                        }
                        noc_up.async_read_barrier();
                        up_done_sem.set(up_seq);
                    }
                }
            }

            // ---- Drain cb_out (down matmul output) to DRAM ----
            const uint32_t row0 = chunk * chunk_M_tiles + my_mt * per_core_M;
            const uint32_t col0 = my_nt_d * per_core_N_d;
            for (uint32_t sb_m = 0; sb_m < d_in1_num_subblocks_M; ++sb_m) {
                for (uint32_t sb_n = 0; sb_n < d_in1_num_subblocks_N; ++sb_n) {
                    cb_out_buf.wait_front(d_out_subblock_num_tiles);
                    uint32_t subblock_tile_offset = 0;
                    for (uint32_t i = 0; i < d_out_subblock_h; ++i) {
                        for (uint32_t j = 0; j < d_out_subblock_w; ++j) {
                            const uint32_t row = row0 + sb_m * d_out_subblock_h + i;
                            const uint32_t col = col0 + sb_n * d_out_subblock_w + j;
                            // `row` indexes the FFN *input* (x) tile-rows; the
                            // destination tile-row adds the per-expert region
                            // offset (0 in non-direct mode).
                            const uint32_t dst_row = row_offset_tiles + row;
                            // Bounds that decide whether this is a real output tile
                            // for this expert:
                            //   * col < N_down_tiles_full: GRID_X=11 ceil_div
                            //     produces phantom output cols past actual N.
                            //   * row < M_tiles_full: ceil_div of M produces a
                            //     last-chunk tail past actual M when
                            //     M_tiles_full doesn't divide chunk_M_tiles.
                            //   * row < count_tiles: the last chunk's per_core_M
                            //     rows extend past count_tiles when count_tiles
                            //     is not chunk-aligned.
                            if (col < N_down_tiles_full && row < M_tiles_full && row < count_tiles) {
                                // The destination tile-row must stay inside the
                                // (possibly shared) output buffer. ttnn::insert
                                // asserted the whole-slice fit
                                // (start_tile_idx + num_tiles <= global_num_tiles);
                                // assert the per-tile equivalent so an over-capacity
                                // region offset fails loudly in watcher builds. The
                                // guard below keeps Release builds safe (skip the OOB
                                // write rather than corrupt DRAM, since ASSERT is a
                                // no-op there).
                                ASSERT(dst_row < dst_M_tiles);
                                if (dst_row < dst_M_tiles) {
                                    const uint32_t tile_idx = dst_row * N_down_tiles_full + col;
                                    noc.async_write(
                                        cb_out_buf,
                                        out_acc,
                                        out_tile_bytes,
                                        {.offset_bytes = subblock_tile_offset},
                                        {.page_id = tile_idx});
                                }
                            }
                            subblock_tile_offset += out_tile_bytes;
                        }
                    }
                    // Wait for the writes to LEAVE this core (departed sender);
                    // doesn't wait for the DRAM round-trip. Safe to reuse the L1
                    // slot now — the NoC has captured the data. ~10x faster than
                    // noc_async_write_barrier per subblock at small per_core_M.
                    noc.async_writes_flushed();
                    cb_out_buf.pop_front(d_out_subblock_num_tiles);
                }
            }
        }  // end chunk loop
    }  // end per-local-expert loop
    // Ensure all outstanding writes complete at the destination before the
    // kernel returns (the next dispatched op may read this output).
    noc.async_write_barrier();
    // UP_SPLIT issues only per-K-block-barriered NoC-1 `up` reads (no NoC-1
    // worker multicast and no NoC-1 atomics), so no extra NoC-1 drain is needed
    // here — which is exactly why it is safe beside the fabric CCL ops.
}
