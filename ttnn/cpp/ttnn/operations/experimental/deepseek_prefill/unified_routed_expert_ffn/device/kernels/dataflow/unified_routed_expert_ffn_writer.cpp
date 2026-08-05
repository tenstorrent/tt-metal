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
#include "../adaptive_chunk.hpp"

constexpr uint32_t TILE_HEIGHT = 32;

void kernel_main() {
    Noc noc;

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_mt = get_arg_val<uint32_t>(1);
    const uint32_t my_nt_d = get_arg_val<uint32_t>(2);
    // start (= expert_region_offsets) address. Only read when direct_write.
    const uint32_t start_addr = get_arg_val<uint32_t>(3);
    // UP_SPLIT up-weight read args: up tensor base, this core's N-column, and
    // whether this core is the gy=0 sender (only senders read `up`).
    const uint32_t up_addr = get_arg_val<uint32_t>(4);
    const uint32_t my_nt_gu = get_arg_val<uint32_t>(5);
    const bool is_up_sender = get_arg_val<uint32_t>(6) != 0;
    // UP_SPLIT local handshake sems (see reader): up_go = slot reserved,
    // up_done = up landed.
    const uint32_t up_go_sem_id = get_arg_val<uint32_t>(7);
    const uint32_t up_done_sem_id = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    // per_core_M_max: CB-sized max per-core M. The runtime per_core_M is picked
    // from the device count below; the down matmul packs the full max ring (so
    // cb_out carries per_core_M_max rows) and this writer emits only the first
    // (runtime) per_core_M of them.
    constexpr uint32_t per_core_M_max = get_compile_time_arg_val(2);
    constexpr uint32_t per_core_N_gu = get_compile_time_arg_val(3);
    constexpr uint32_t per_core_N_d = get_compile_time_arg_val(4);
    // DOWN_SPLIT (26..32): the writer reads the UPPER K-rows of each down block on
    // NoC 1 while the reader reads [0, down_split_k) on NoC 0.
    constexpr uint32_t cb_in1_down = get_compile_time_arg_val(26);
    constexpr uint32_t in0_block_w_d = get_compile_time_arg_val(27);
    constexpr uint32_t K_down_tiles = get_compile_time_arg_val(28);
    constexpr uint32_t num_blocks_d = get_compile_time_arg_val(29);
    constexpr uint32_t d_in1_block_num_tiles = get_compile_time_arg_val(30);
    constexpr uint32_t down_split_k = get_compile_time_arg_val(31);
    constexpr bool writer_split_down = get_compile_time_arg_val(32) != 0;
    constexpr uint32_t d_out_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t d_out_subblock_w = get_compile_time_arg_val(8);
    constexpr uint32_t N_gate_tiles_full = get_compile_time_arg_val(9);
    constexpr uint32_t N_down_tiles_full = get_compile_time_arg_val(10);
    constexpr uint32_t num_chunks_max = get_compile_time_arg_val(11);
    constexpr uint32_t chunk_M_max = get_compile_time_arg_val(12);
    // device-side count read.
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(13);
    constexpr uint32_t cb_idx_scratch = get_compile_time_arg_val(14);
    constexpr uint32_t local_expert_id = get_compile_time_arg_val(15);
    // M_tiles_full: total tile-row count of the FFN *input* (x). When the
    // kernel runs more chunks than strictly needed (because
    // M_tiles_full % chunk_M_tiles != 0), the last chunk has writer
    // destinations past M_tiles_full — we skip those source rows here.
    constexpr uint32_t M_tiles_full = get_compile_time_arg_val(16);
    // direct_write: 0 -> write to per-expert output at row 0 (standalone FFN).
    //               1 -> write into shared buffer at start[global_id]/TILE.
    constexpr uint32_t direct_write = get_compile_time_arg_val(17);
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
    // When the compute tail-skips the last down block's K padding, the down
    // matmul never reduces the N-OOB hidden columns, so the `up` read can skip
    // zero-filling them. Derived identically to the reader's down_k_tail_skip.
    constexpr bool down_k_tail_skip = get_compile_time_arg_val(24) != 0;
    // See the reader: under WEIGHTS_ND_SHARDED a whole K-row `up` slice is one
    // request. SHARD_GRID_N is the shard grid's N extent (= GRID_X).
    // ceil(N_tiles / per_core_N), not GRID_X; see the reader for the full rationale and
    // why gate/up and down need separate extents.
    constexpr uint32_t SHARD_GRID_N_GU = get_compile_time_arg_val(25);
    constexpr uint32_t SHARD_GRID_N_D = get_compile_time_arg_val(33);
    // IN1_WRITER_MCAST: the writer issues the gate/up weight multicast on its own NoC 1.
    constexpr uint32_t cb_in1_gate = get_compile_time_arg_val(34);
    constexpr bool writer_mcasts_in1 = get_compile_time_arg_val(35) != 0;

    constexpr uint32_t d_out_subblock_num_tiles = d_out_subblock_h * d_out_subblock_w;
    // Full compile-time M-subblock count of cb_out (the down matmul copies the
    // full max ring). The writer DRAINS all of them but only WRITES the first
    // (runtime) per_core_M rows (see the drain loop below).
    constexpr uint32_t d_in1_num_subblocks_M = per_core_M_max / d_out_subblock_h;
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
    // 7 DOWN_SPLIT compile args (26..32), the down shard extent (33) and the two
    // IN1_WRITER_MCAST args (34..35) precede the accessor stream.
    constexpr uint32_t GU_SHARD_KROWS = get_compile_time_arg_val(36);  // gate/up shard height
    constexpr uint32_t out_accessor_offset = 37;
    constexpr auto out_args = TensorAccessorArgs<out_accessor_offset>();
    const auto out_acc = TensorAccessor(out_args, output_addr, cb_out_buf.get_tile_size());

    constexpr uint32_t start_accessor_offset = out_args.next_compile_time_args_offset();
    constexpr auto start_args = TensorAccessorArgs<start_accessor_offset>();
    const auto start_acc = TensorAccessor(start_args, start_addr);

    constexpr uint32_t up_accessor_offset = start_args.next_compile_time_args_offset();
    constexpr auto up_args = TensorAccessorArgs<up_accessor_offset>();
    const auto up_acc = TensorAccessor(up_args, up_addr, get_tile_size(cb_in1_up));
    // DOWN_SPLIT: down accessor follows up in the compile-arg stream.
    constexpr uint32_t down_accessor_offset = up_args.next_compile_time_args_offset();
    constexpr auto down_args = TensorAccessorArgs<down_accessor_offset>();
    const uint32_t down_addr = get_arg_val<uint32_t>(9);
    const auto down_acc = TensorAccessor(down_args, down_addr, get_tile_size(cb_in1_down));

    const uint32_t out_tile_bytes = cb_out_buf.get_tile_size();

    // Wait for the reader's counts/idx push and compute effective_chunks =
    // ceil(count / chunk_M_tiles). The writer drains cb_out per chunk;
    // bounding the loop here is required because the reader and compute
    // bound theirs too — without this, the writer would wait forever on
    // cb_out for chunks the compute never pushes.
    cb_counts_scratch_buf.wait_front(1);
    cb_idx_scratch_buf.wait_front(1);
    const volatile tt_l1_ptr uint32_t* counts_ptr =
        reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(cb_counts_scratch_buf.get_read_ptr());
    const uint32_t idx_l1 = cb_idx_scratch_buf.get_read_ptr();
    const volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(idx_l1);
    const uint32_t global_expert_id = idx_ptr[local_expert_id];
    const uint32_t count_value = counts_ptr[global_expert_id];
    const uint32_t count_tiles = (count_value + TILE_HEIGHT - 1) / TILE_HEIGHT;
    // Runtime chunk layout from the actual count (same math as reader/compute so
    // the row mapping agrees). per_core_M is per-chunk (see the loop).
    const uint32_t effective_chunks_runtime = adaptive_chunk::num_chunks(count_tiles, chunk_M_max);
    const uint32_t effective_chunks =
        effective_chunks_runtime < num_chunks_max ? effective_chunks_runtime : num_chunks_max;

    // Destination tile-row offset for direct-write mode. In direct-write
    // mode the output buffer is a SHARED buffer and this expert's slice
    // begins at start[global_expert_id] (token row); convert to tile rows.
    // Mirrors ttnn::insert's writer: start_tile_row = start_value / TILE.
    uint32_t row_offset_tiles = 0;
    if constexpr (direct_write != 0) {
        const uint32_t start_l1 = cb_start_scratch_buf.get_write_ptr();
        const uint32_t start_page_size = start_acc.get_aligned_page_size();
        noc.async_read(start_acc, CoreLocalMem<uint32_t>(start_l1), start_page_size, {.page_id = 0}, {});
        noc.async_read_barrier();
        const volatile tt_l1_ptr uint32_t* start_ptr = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(start_l1);
        const uint32_t start_value = start_ptr[global_expert_id];
        row_offset_tiles = start_value / TILE_HEIGHT;
    }

    // ---- UP_SPLIT up-weight read setup ----
    // The writer reads `up` from DRAM on NoC 1 concurrent with the
    // reader's NoC-0 `gate` read, into the gy=0 sender's cb_in1_up slot; the
    // reader multicasts it on NoC 0. A local same-core (BRISC reader <-> NCRISC
    // writer) handshake orders the two: up_go (reader: slot reserved) and
    // up_done (writer: up landed in L1), monotonic counters.
    Noc noc_up(1);
    // IN1_WRITER_MCAST runtime args (10..18) and the gate/up multicast state. NoC 1
    // multicasts bottom-left -> top-right, so the rectangle corners are given in the
    // OPPOSITE order from a NoC-0 multicast (device.cpp:818 does the same swap host-side);
    // pass them unswapped and the NoC reads start>end as a torus wraparound, covers no
    // receiver, and the run deadlocks with no assert.
    const uint32_t in1_mc_nx_start = get_arg_val<uint32_t>(12);  // = NoC-0 frame's END x
    const uint32_t in1_mc_ny_start = get_arg_val<uint32_t>(13);  // = NoC-0 frame's END y
    const uint32_t in1_mc_nx_end = get_arg_val<uint32_t>(10);    // = NoC-0 frame's START x
    const uint32_t in1_mc_ny_end = get_arg_val<uint32_t>(11);    // = NoC-0 frame's START y
    const uint32_t in1_num_receivers = get_arg_val<uint32_t>(14);
    Semaphore<> in1_ready_sem(get_arg_val<uint32_t>(15));
    Semaphore<> in1_valid_sem(get_arg_val<uint32_t>(16));
    Semaphore<> mcast_go_sem(get_arg_val<uint32_t>(17));
    Semaphore<> mcast_done_sem(get_arg_val<uint32_t>(18));
    const uint32_t up_tile_bytes = get_tile_size(cb_in1_up);
    Semaphore<> up_go_sem(up_go_sem_id);
    Semaphore<> up_done_sem(up_done_sem_id);
    // CB slot cadence must be tracked PER CB with counters that run across chunks:
    // the reader pushes cb_in1_up once per gate/up block and cb_in1_down once per
    // down block, and neither count is even (14 and 11 for kimi), so the live slot
    // parity carries over between chunks. Deriving a slot from the shared up_seq
    // breaks as soon as there is more than one chunk (measured: isl >= 2048 failed).
    uint32_t up_blk = 0;
    uint32_t down_blk = 0;
    // IN1_WRITER_MCAST handshake counter -- gate/up blocks only, see the reader.
    uint32_t mc_seq = 0;
    uint32_t up_seq = 0;

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
                    uint32_t l1_w_up = up_cb_base + (up_blk % kUpNumSlots) * up_slot_bytes;
                    ++up_blk;
#ifdef WEIGHTS_ND_SHARDED
                    {
                        constexpr uint32_t up_slice_bytes = per_core_N_gu * 576;  // bfp4 tile
                        // Grouped like the reader's gate read -- see there for why.
                        constexpr uint32_t up_group_bytes = up_slice_bytes * GU_SHARD_KROWS;
                        for (uint32_t g = 0; g < in0_block_w_gu / GU_SHARD_KROWS; ++g) {
                            const uint32_t krow = kb * in0_block_w_gu + g * GU_SHARD_KROWS;
                            const uint64_t src = up_acc.get_shard_noc_addr(
                                (krow / GU_SHARD_KROWS) * SHARD_GRID_N_GU + my_nt_gu, 0, noc_up.get_noc_id());
                            noc_async_read(src, l1_w_up, up_group_bytes, noc_up.get_noc_id());
                            l1_w_up += up_group_bytes;
                        }
                    }
#else
                    for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
                        for (uint32_t n = 0; n < per_core_N_gu; ++n) {
                            const uint32_t row = kb * in0_block_w_gu + k;
                            const uint32_t col = my_nt_gu * per_core_N_gu + n;
                            if (col < N_gate_tiles_full) {
                                const uint32_t tile_idx = row * N_gate_tiles_full + col;
                                noc_up.async_read(
                                    up_acc, CoreLocalMem<uint32_t>(l1_w_up), up_tile_bytes, {.page_id = tile_idx}, {});
                            } else {
                                // N-OOB hidden padding column: garbage up output feeds the
                                // down matmul's K reduction, so keep it zero UNLESS the down
                                // tail-skips the last block's padding (down_k_tail_skip) —
                                // then this column is never reduced and the garbage is dropped.
                                if constexpr (!down_k_tail_skip) {
                                    volatile tt_l1_ptr uint64_t* p =
                                        reinterpret_cast<volatile tt_l1_ptr uint64_t*>(l1_w_up);
                                    for (uint32_t i = 0; i < up_tile_bytes / 8; ++i) {
                                        p[i] = 0;
                                    }
                                }
                            }
                            l1_w_up += up_tile_bytes;
                        }
                    }
#endif  // WEIGHTS_ND_SHARDED
                    noc_up.async_read_barrier();
                    up_done_sem.set(up_seq);

                    // IN1_WRITER_MCAST: this RISC owns NoC 1, so it runs the weight
                    // multicast the reader used to do on NoC 0. Wait for the reader's
                    // gate read (mcast_go), then for the receivers to free their slots
                    // (in1_ready), multicast gate+up, and signal mcast_done so the reader
                    // may eventually REUSE this slot. The reader does not block on
                    // mcast_done until it needs the slot again, so its next block's DRAM
                    // read overlaps this multicast -- which is the whole point.
                    if constexpr (writer_mcasts_in1) {
                        ++mc_seq;
                        mcast_go_sem.wait_min(mc_seq);
                        in1_ready_sem.wait(in1_num_receivers);
                        in1_ready_sem.set(0);
                        const uint32_t gate_tile_bytes_l = get_tile_size(cb_in1_gate);
                        CircularBuffer cb_in1_gate_buf(cb_in1_gate);
                        const uint32_t gate_slot_bytes = g_in1_block_num_tiles * gate_tile_bytes_l;
                        const uint32_t gate_block_start =
                            cb_in1_gate_buf.get_write_ptr() + ((up_blk - 1) % kUpNumSlots) * gate_slot_bytes;
                        // linked=true on both data multicasts and the sem, so the posted
                        // valid-sem write cannot overtake the data (same rationale as the
                        // reader's NoC-0 version it replaces).
                        noc_up.async_write_multicast(
                            CoreLocalMem<uint32_t>(gate_block_start),
                            MulticastEndpoint{},
                            gate_slot_bytes,
                            in1_num_receivers,
                            {.offset_bytes = 0},
                            {.noc_x_start = in1_mc_nx_start,
                             .noc_y_start = in1_mc_ny_start,
                             .noc_x_end = in1_mc_nx_end,
                             .noc_y_end = in1_mc_ny_end,
                             .addr = gate_block_start},
                            /*linked=*/true);
                        const uint32_t up_block_start = up_cb_base + ((up_blk - 1) % kUpNumSlots) * up_slot_bytes;
                        noc_up.async_write_multicast(
                            CoreLocalMem<uint32_t>(up_block_start),
                            MulticastEndpoint{},
                            up_slot_bytes,
                            in1_num_receivers,
                            {.offset_bytes = 0},
                            {.noc_x_start = in1_mc_nx_start,
                             .noc_y_start = in1_mc_ny_start,
                             .noc_x_end = in1_mc_nx_end,
                             .noc_y_end = in1_mc_ny_end,
                             .addr = up_block_start},
                            /*linked=*/true);
                        noc_up.async_writes_flushed();
                        in1_valid_sem.set(1);
                        in1_valid_sem.set_multicast<NocOptions::DEFAULT>(
                            noc_up, in1_mc_nx_start, in1_mc_ny_start, in1_mc_nx_end, in1_mc_ny_end, in1_num_receivers);
                        mcast_done_sem.set(mc_seq);
                    }
                }
            }
        }

        // ---- DOWN_SPLIT: read the UPPER K-rows of each down block on NoC 1 ----
        // Mirrors the UP_SPLIT block above. The reader owns cb_in1_down's
        // reserve/push and reads rows [0, down_split_k) on NoC 0; this RISC reads
        // [down_split_k, in0_block_w_d) concurrently. Per K-block: wait for the
        // reader's go, read, barrier, signal done so the reader can multicast.
        // The go/done counter continues the same up_seq the UP_SPLIT phase used —
        // both kernels advance it once per block, gated on the same sender core, so
        // they stay in lockstep across the phase boundary.
        if constexpr (writer_split_down) {
            if (is_up_sender) {
                // cb_in1_down is double-buffered with one push per down K-block, so
                // the live slot is down_blk % 2 — a counter that runs across chunks
                // (see up_blk/down_blk above for why kb % 2 is wrong).
                constexpr uint32_t kDownNumSlots = 2;
                CircularBuffer cb_in1_down_buf(cb_in1_down);
                const uint32_t down_cb_base = cb_in1_down_buf.get_write_ptr();
                const uint32_t down_tile_bytes = get_tile_size(cb_in1_down);
                const uint32_t down_slot_bytes = d_in1_block_num_tiles * down_tile_bytes;
                for (uint32_t kb = 0; kb < num_blocks_d; ++kb) {
                    ++up_seq;
                    up_go_sem.wait_min(up_seq);
                    uint32_t l1_w = down_cb_base + (down_blk % kDownNumSlots) * down_slot_bytes +
                                    down_split_k * per_core_N_d * down_tile_bytes;
                    ++down_blk;
#ifdef WEIGHTS_ND_SHARDED
                    {
                        const uint32_t down_slice_bytes = per_core_N_d * down_tile_bytes;
                        for (uint32_t k = down_split_k; k < in0_block_w_d; ++k) {
                            const uint32_t krow = kb * in0_block_w_d + k;
                            if (krow < K_down_tiles) {
                                const uint64_t src = down_acc.get_shard_noc_addr(
                                    krow * SHARD_GRID_N_D + my_nt_d, 0, noc_up.get_noc_id());
                                noc_async_read(src, l1_w, down_slice_bytes, noc_up.get_noc_id());
                            } else if constexpr (!down_k_tail_skip) {
                                volatile tt_l1_ptr uint64_t* p = reinterpret_cast<volatile tt_l1_ptr uint64_t*>(l1_w);
                                for (uint32_t b = 0; b < down_slice_bytes / 8; ++b) {
                                    p[b] = 0;
                                }
                            }
                            l1_w += down_slice_bytes;
                        }
                    }
#else
                    for (uint32_t k = down_split_k; k < in0_block_w_d; ++k) {
                        for (uint32_t n = 0; n < per_core_N_d; ++n) {
                            const uint32_t row = kb * in0_block_w_d + k;
                            const uint32_t col = my_nt_d * per_core_N_d + n;
                            if (row < K_down_tiles && col < N_down_tiles_full) {
                                const uint32_t tile_idx = row * N_down_tiles_full + col;
                                noc_up.async_read(
                                    down_acc, CoreLocalMem<uint32_t>(l1_w), down_tile_bytes, {.page_id = tile_idx}, {});
                            } else if (row >= K_down_tiles) {
                                // K-OOB: matched to the reader's identical zero-fill.
                                if constexpr (!down_k_tail_skip) {
                                    volatile tt_l1_ptr uint64_t* p =
                                        reinterpret_cast<volatile tt_l1_ptr uint64_t*>(l1_w);
                                    for (uint32_t i = 0; i < down_tile_bytes / 8; ++i) {
                                        p[i] = 0;
                                    }
                                }
                            }
                            l1_w += down_tile_bytes;
                        }
                    }
#endif  // WEIGHTS_ND_SHARDED
                    noc_up.async_read_barrier();
                    up_done_sem.set(up_seq);
                }
            }
        }

        // ---- Drain cb_out (down matmul output) to DRAM ----
        // Per-chunk per_core_M (per_core_M_max for full chunks, a smaller divisor
        // for the tail); chunk starts are uniform at chunk*chunk_M_max. Contiguous
        // row map: this core owns rows [row0, row0 + per_core_M).
        const uint32_t per_core_M = adaptive_chunk::per_core_M_for_chunk(chunk, count_tiles, chunk_M_max);
        const uint32_t row0 = chunk * chunk_M_max + my_mt * per_core_M;
        const uint32_t col0 = my_nt_d * per_core_N_d;
        // The DOWN matmul cycles the full compile-time-MAX ring through its
        // PARTIALS CB (L1_ACC needs that to wrap onto block 0's slots), but it
        // only copies the first per_core_M (runtime) M-rows out to cb_out — the
        // rest are ring padding it drains and discards. So bound this drain by the
        // same runtime per_core_M the compute kernel used; both derive it from the
        // device-side count, so they cannot disagree. (d_out_subblock_h is 1, the
        // same assumption matmul_phase makes when it compares m_subblocks against
        // its subblock-row count.)
        const uint32_t sb_m_bound = per_core_M / d_out_subblock_h;
        for (uint32_t sb_m = 0; sb_m < sb_m_bound; ++sb_m) {
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
                        //   * sb_m < per_core_M: cb_out carries per_core_M_max
                        //     rows (full ring); rows past the runtime per_core_M
                        //     are zeros that belong to other cores — never write.
                        if (sb_m < per_core_M && col < N_down_tiles_full && row < M_tiles_full && row < count_tiles) {
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
    }
    // Ensure all outstanding writes complete at the destination before the
    // kernel returns (the next dispatched op may read this output).
    noc.async_write_barrier();
    // UP_SPLIT issues only per-K-block-barriered NoC-1 `up` reads (no NoC-1
    // worker multicast and no NoC-1 atomics), so no extra NoC-1 drain is needed
    // here — which is exactly why it is safe beside the fabric CCL ops.
}
