// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for the GROUPED unified_routed_expert_ffn program factory.
//
// Derived from unified_routed_expert_ffn_writer.cpp: row groups of group_rows (R)
// M-rows, device-side work-item plan (group_assign::build_plan: expert chunk ranges
// assigned to groups), my_mt is the row within the group, chunk_M_max =
// per_core_M_max * R, and column ownership may be strided (col_strided, see
// group_assign::global_col). The UP_SPLIT reader/writer sequence is kept.
//
//
// Two responsibilities, both handled by this writer kernel:
//
// 1. Output drain + placement. Pop `cb_out` (the down matmul's per-core
//    final block, packed one subblock at a time) and write tiles to the
//    DRAM-interleaved output tensor at this core's (mt, nt_d) tile region,
//    looped over `effective_chunks` chunks (computed from the device-side
//    counts/idx scratch CBs). Activated tiles are distributed across the
//    M-row by the READER via L1 multicast — no DRAM scratch round-trip or
//    cross-core barrier. Every expert's output is written directly into the
//    shared destination buffer at that expert's region offset: the kernel
//    reads start[global_expert_id] from `start` (= expert_region_offsets)
//    device-side and adds (start / TILE_HEIGHT) tile-rows to every output
//    row — fusing what ttnn::insert would otherwise do (no temp-buffer DRAM
//    round-trip). expert_region_offsets is a mandatory op input, so there is
//    no per-expert-output-tensor mode.
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
#include "../group_assign.hpp"
#include "../fast_interleaved.hpp"

constexpr uint32_t TILE_HEIGHT = 32;

void kernel_main() {
    Noc noc;

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_mt = get_arg_val<uint32_t>(1);
    const uint32_t my_nt_d = get_arg_val<uint32_t>(2);
    const uint32_t start_addr = get_arg_val<uint32_t>(3);
    // UP_SPLIT up-weight read args. The per-expert `up` addresses live in the
    // runtime-arg array starting at UP_RT.
    const uint32_t my_nt_gu = get_arg_val<uint32_t>(4);
    const bool is_up_sender = get_arg_val<uint32_t>(5) != 0;
    // UP_SPLIT local handshake sems (see reader): up_go = slot reserved,
    // up_done = up landed.
    const uint32_t up_go_sem_id = get_arg_val<uint32_t>(6);
    const uint32_t up_done_sem_id = get_arg_val<uint32_t>(7);
    // Grouped-path args: row group id at 8; DOWN_SPLIT local handshake sems at 9/10;
    // per-expert `up` addresses follow at UP_RT, then per-expert `down` addresses at DOWN_RT.
    const uint32_t my_group = get_arg_val<uint32_t>(8);
    const uint32_t down_go_sem_id = get_arg_val<uint32_t>(9);
    const uint32_t down_done_sem_id = get_arg_val<uint32_t>(10);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    // per_core_M_max: CB-sized max per-core M. The runtime per_core_M is picked
    // from the device count below; the down matmul packs the full max ring (so
    // cb_out carries per_core_M_max rows) and this writer emits only the first
    // (runtime) per_core_M of them.
    constexpr uint32_t per_core_M_max = get_compile_time_arg_val(1);
    constexpr uint32_t per_core_N_gu = get_compile_time_arg_val(2);
    constexpr uint32_t per_core_N_d = get_compile_time_arg_val(3);
    constexpr uint32_t d_out_subblock_h = get_compile_time_arg_val(4);
    constexpr uint32_t d_out_subblock_w = get_compile_time_arg_val(5);
    constexpr uint32_t N_gate_tiles_full = get_compile_time_arg_val(6);
    constexpr uint32_t N_down_tiles_full = get_compile_time_arg_val(7);
    constexpr uint32_t num_chunks_max = get_compile_time_arg_val(8);
    constexpr uint32_t chunk_M_max = get_compile_time_arg_val(9);
    // device-side count read.
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(10);
    constexpr uint32_t cb_idx_scratch = get_compile_time_arg_val(11);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(12);
    // M_tiles_full: total tile-row count of the FFN *input* (x). When the
    // kernel runs more chunks than strictly needed (because
    // M_tiles_full % chunk_M_tiles != 0), the last chunk has writer
    // destinations past M_tiles_full — we skip those source rows here.
    constexpr uint32_t M_tiles_full = get_compile_time_arg_val(13);
    // dst_M_tiles: tile-row count of the *destination* (shared) buffer, used to
    // bound destination writes.
    constexpr uint32_t dst_M_tiles = get_compile_time_arg_val(14);
    constexpr uint32_t cb_start_scratch = get_compile_time_arg_val(15);
    // UP_SPLIT up-weight read (see header): the writer reads `up` on NoC 1 into
    // the gy=0 sender's cb_in1_up slot; the reader mcasts it on NoC 0.
    // writer_split_up gates it (1 = UP_SPLIT, 0 = LEGACY: reader owns `up`).
    constexpr uint32_t cb_in1_up = get_compile_time_arg_val(16);
    constexpr uint32_t in0_block_w_gu = get_compile_time_arg_val(17);
    constexpr uint32_t K_gate_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t writer_split_up = get_compile_time_arg_val(19);
    // Grouped-path compile-time args.
    constexpr uint32_t num_row_groups = get_compile_time_arg_val(20);
    constexpr uint32_t group_rows = get_compile_time_arg_val(21);  // R (= adaptive grid_y)
    constexpr uint32_t col_strided = get_compile_time_arg_val(22);
    constexpr uint32_t lpt_fixed_cost_tiles = get_compile_time_arg_val(23);
    constexpr uint32_t grid_x = get_compile_time_arg_val(24);
    // DOWN_SPLIT (see reader): writer reads the down-weight K-blocks on NoC 1.
    constexpr uint32_t cb_in1_down = get_compile_time_arg_val(25);
    constexpr uint32_t in0_block_w_d = get_compile_time_arg_val(26);
    constexpr uint32_t K_down_tiles = get_compile_time_arg_val(27);
    constexpr uint32_t K_down_tiles_padded = get_compile_time_arg_val(28);
    constexpr uint32_t down_split = get_compile_time_arg_val(29);
    // Blocks per weight CB: the reader's push cadence cycles through this many slots,
    // so the writer must fill slot (seq-1) % weight_cb_depth (NOT a hard-coded 2).
    constexpr uint32_t weight_cb_depth = get_compile_time_arg_val(30);
    constexpr uint32_t kNumDramBanks = get_compile_time_arg_val(31);
    constexpr uint32_t num_blocks_d = K_down_tiles_padded / in0_block_w_d;
    constexpr uint32_t d_in1_block_num_tiles = per_core_N_d * in0_block_w_d;

    constexpr uint32_t d_out_subblock_num_tiles = d_out_subblock_h * d_out_subblock_w;
    // Full compile-time M-subblock count of cb_out. The writer DRAINS all of them
    // but only WRITES the first (runtime) per_core_M rows (see the drain loop below).
    constexpr uint32_t d_in1_num_subblocks_M = per_core_M_max / d_out_subblock_h;
    constexpr uint32_t d_in1_num_subblocks_N = per_core_N_d / d_out_subblock_w;
    constexpr uint32_t num_blocks_gu = K_gate_tiles / in0_block_w_gu;
    constexpr uint32_t g_in1_block_num_tiles = per_core_N_gu * in0_block_w_gu;

    CircularBuffer cb_out_buf(cb_out);
    CircularBuffer cb_counts_scratch_buf(cb_counts_scratch);
    CircularBuffer cb_idx_scratch_buf(cb_idx_scratch);
    CircularBuffer cb_start_scratch_buf(cb_start_scratch);

    // Accessor compile-arg stream order (host appends in this exact order):
    // out, then start, then up (UP_SPLIT). The accessors are constructed
    // unconditionally; up_acc is used only when writer_split_up.
    constexpr uint32_t out_accessor_offset = 32;
    constexpr auto out_args = TensorAccessorArgs<out_accessor_offset>();
    const auto out_acc = TensorAccessor(out_args, output_addr, cb_out_buf.get_tile_size());

    constexpr uint32_t start_accessor_offset = out_args.next_compile_time_args_offset();
    constexpr auto start_args = TensorAccessorArgs<start_accessor_offset>();
    const auto start_acc = TensorAccessor(start_args, start_addr);

    constexpr uint32_t up_accessor_offset = start_args.next_compile_time_args_offset();
    constexpr auto up_args = TensorAccessorArgs<up_accessor_offset>();
    // Per-expert `up` base addresses live in a runtime-arg array after the fixed
    // writer args; UP_RT is its first index. up_addr(e) = arg[UP_RT + e].
    constexpr uint32_t UP_RT = 11;
    constexpr uint32_t DOWN_RT = UP_RT + experts_per_chip;
    // `down` accessor follows `up` in the compile-arg stream (host appends it last).
    constexpr uint32_t down_accessor_offset = up_args.next_compile_time_args_offset();
    constexpr auto down_args = TensorAccessorArgs<down_accessor_offset>();

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

    // Device-side expert -> row-group assignment (identical in reader/compute/writer).
    group_assign::Plan plan;
    group_assign::build_plan(
        counts_ptr,
        idx_ptr,
        experts_per_chip,
        num_row_groups,
        chunk_M_max,
        num_chunks_max,
        M_tiles_full,
        lpt_fixed_cost_tiles,
        plan);

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
    // DOWN_SPLIT counter + sems (separate cadence from up_seq).
    Semaphore<> down_go_sem(down_go_sem_id);
    Semaphore<> down_done_sem(down_done_sem_id);
    uint32_t down_seq = 0;
    const uint32_t down_tile_bytes = get_tile_size(cb_in1_down);

    // ======================= per-local-expert loop =======================
    // Drain every local expert's down-matmul output (and, for UP_SPLIT sender
    // cores, feed its `up` weights) into the shared output buffer at the
    // expert's region offset. The chunk-loop body below is unchanged from the
    // single-expert kernel — only the per-expert bindings differ.
    for (uint32_t item = 0; item < plan.n_items; ++item) {
        // Items of other row groups are skipped entirely (the other kernels skip identically).
        if (plan.group[item] != my_group) {
            continue;
        }
        const uint32_t local_expert_id = plan.expert[item];
        const uint32_t item_chunk_b = plan.chunk_b[item];
        const uint32_t item_chunk_e = plan.chunk_e[item];
        const auto up_acc = TensorAccessor(up_args, get_arg_val<uint32_t>(UP_RT + local_expert_id), up_tile_bytes);
        const auto down_acc =
            TensorAccessor(down_args, get_arg_val<uint32_t>(DOWN_RT + local_expert_id), down_tile_bytes);
        (void)up_acc;
        (void)down_acc;
        // Fast interleaved addressing on NoC 1 (see fast_interleaved.hpp).
        fast_il::Bases<kNumDramBanks> up_bases;
        up_bases.init(get_arg_val<uint32_t>(UP_RT + local_expert_id), /*noc=*/1);
        fast_il::Bases<kNumDramBanks> down_bases;
        down_bases.init(get_arg_val<uint32_t>(DOWN_RT + local_expert_id), /*noc=*/1);
        const uint32_t global_expert_id = idx_ptr[local_expert_id];
        const uint32_t count_value = counts_ptr[global_expert_id];
        // Same capacity clamp the reader and compute apply to the device-produced
        // count (see adaptive_chunk::clamp_count_tiles): arithmetic, so it bounds
        // the chunk loop in Release too, where ASSERT is a no-op.
        const uint32_t count_tiles_raw = (count_value + TILE_HEIGHT - 1) / TILE_HEIGHT;
        const uint32_t count_tiles =
            adaptive_chunk::clamp_count_tiles(count_tiles_raw, chunk_M_max, num_chunks_max, M_tiles_full);
        ASSERT(count_tiles == count_tiles_raw);
        // Runtime chunk layout from THIS expert's count — the same math the reader
        // and compute kernels run, so all three agree on the row mapping (a
        // mismatch would emit this expert's rows at the wrong token offsets).
        const uint32_t effective_chunks = adaptive_chunk::num_chunks(count_tiles, chunk_M_max);
        (void)effective_chunks;  // this item covers chunks [item_chunk_b, item_chunk_e) of the expert
        const uint32_t row_offset_tiles = start_ptr[global_expert_id] / TILE_HEIGHT;

        for (uint32_t chunk = item_chunk_b; chunk < item_chunk_e; ++chunk) {
            // ---- Phase 1/2 weight feed: writer reads `up` on NoC 1 (UP_SPLIT) ----
            // Streams `up` from DRAM concurrent with the reader's NoC-0 `gate` read.
            // Runs before the cb_out drain.
            if constexpr (writer_split_up) {
                // UP_SPLIT: only gy=0 in1-sender cores read `up`. Per K-block: wait
                // for the reader to reserve the slot (up_go), read this column's
                // `up` slice on NoC 1 into it, then signal up_done so the reader
                // mcasts on NoC 0. Only a NoC-1 DRAM read here (fabric-safe); the
                // reader owns cb_in1_up reserve/push.
                // Split-read (group_rows > 1): this core reads only tile-rows [k_b, k_e) of
                // each `up` block; the other rows arrive by multicast from the group peers.
                uint32_t up_k_b = 0;
                uint32_t up_k_e = in0_block_w_gu;
                if constexpr (group_rows > 1) {
                    group_assign::slice_rows(in0_block_w_gu, group_rows, my_mt, up_k_b, up_k_e);
                }
                if (is_up_sender) {
                    // The CB write pointer is PER-RISC and the reader owns push, so
                    // the writer's get_write_ptr never advances. Replicate the
                    // reader's cadence: cb_in1_up is double-buffered, one push per
                    // K-block, so the live slot is base + (up_seq-1)%2 * slot.
                    constexpr uint32_t kUpNumSlots = weight_cb_depth;
                    CircularBuffer cb_in1_up_buf(cb_in1_up);
                    const uint32_t up_cb_base = cb_in1_up_buf.get_write_ptr();
                    const uint32_t up_slot_bytes = g_in1_block_num_tiles * up_tile_bytes;
                    for (uint32_t kb = 0; kb < num_blocks_gu; ++kb) {
                        ++up_seq;
                        up_go_sem.wait_min(up_seq);
                        uint32_t l1_w_up = up_cb_base + ((up_seq - 1) % kUpNumSlots) * up_slot_bytes +
                                           up_k_b * per_core_N_gu * up_tile_bytes;
                        if constexpr (col_strided) {
                            // Band mode: one contiguous run per slice (see reader).
                            constexpr uint32_t run_tiles = N_gate_tiles_full / grid_x;
                            const uint64_t src =
                                up_bases.b[my_nt_gu] +
                                static_cast<uint64_t>((kb * in0_block_w_gu + up_k_b) * run_tiles) * up_tile_bytes;
                            noc_async_read(src, l1_w_up, (up_k_e - up_k_b) * run_tiles * up_tile_bytes, /*noc=*/1);
                        } else {
                            for (uint32_t k = up_k_b; k < up_k_e; ++k) {
                                const uint32_t row = kb * in0_block_w_gu + k;
                                const uint32_t col0 = my_nt_gu * per_core_N_gu;
                                fast_il::Cursor<kNumDramBanks> uc;
                                uc.start(row * N_gate_tiles_full + col0);
                                for (uint32_t n = 0; n < per_core_N_gu; ++n) {
                                    // N-OOB hidden padding column left UNWRITTEN (never reduced).
                                    if (col0 + n < N_gate_tiles_full) {
                                        noc_async_read(
                                            uc.addr(up_bases, up_tile_bytes), l1_w_up, up_tile_bytes, /*noc=*/1);
                                    }
                                    uc.next();
                                    l1_w_up += up_tile_bytes;
                                }
                            }
                        }
                        noc_up.async_read_barrier();
                        up_done_sem.set(up_seq);
                    }
                }
            }

            // ---- Phase 4 weight feed: writer reads `down` on NoC 1 (DOWN_SPLIT) ----
            // Same protocol as UP_SPLIT: per down K-block wait for the reader's slot
            // reservation (down_go), read this column's down slice into the slot, then
            // signal down_done so the reader mcasts / pushes it. Runs before the cb_out
            // drain: the down matmul cannot produce output before its weights arrive,
            // and the reader reserves slot kb only after compute consumed kb-2, so the
            // two orders never wait on each other.
            if constexpr (down_split) {
                if (is_up_sender) {
                    constexpr uint32_t kDownNumSlots = weight_cb_depth;
                    CircularBuffer cb_in1_down_buf(cb_in1_down);
                    const uint32_t down_cb_base = cb_in1_down_buf.get_write_ptr();
                    const uint32_t down_slot_bytes = d_in1_block_num_tiles * down_tile_bytes;
                    for (uint32_t kb = 0; kb < num_blocks_d; ++kb) {
                        ++down_seq;
                        down_go_sem.wait_min(down_seq);
                        // My slice of this block is tile-rows [k_b, k_e) (split-read); the reader
                        // reads [k_b, k_mid) on NoC 0 and the writer [k_mid, k_e) on NoC 1.
                        uint32_t k_b = 0;
                        uint32_t k_e = in0_block_w_d;
                        if constexpr (group_rows > 1) {
                            group_assign::slice_rows(in0_block_w_d, group_rows, my_mt, k_b, k_e);
                        }
                        const uint32_t k_mid = k_b + (k_e - k_b) / 2;
                        uint32_t l1_w = down_cb_base + ((down_seq - 1) % kDownNumSlots) * down_slot_bytes +
                                        k_mid * per_core_N_d * down_tile_bytes;
                        for (uint32_t k = k_mid; k < k_e; ++k) {
                            // K-block kb = the activated N-slice of column core kb, so the
                            // down-weight ROWS follow that core's column map (see reader).
                            const uint32_t row = group_assign::global_col(kb, k, in0_block_w_d, grid_x, col_strided);
                            if constexpr (col_strided) {
                                // Band mode: this row's tiles {i*grid_x + gx} are one contiguous
                                // run in bank gx at offset row*(N/grid_x).
                                constexpr uint32_t run_tiles = N_down_tiles_full / grid_x;
                                if (row < K_down_tiles) {
                                    const uint64_t src = down_bases.b[my_nt_d] +
                                                         static_cast<uint64_t>(row * run_tiles) * down_tile_bytes;
                                    noc_async_read(src, l1_w, run_tiles * down_tile_bytes, /*noc=*/1);
                                }
                                l1_w += per_core_N_d * down_tile_bytes;
                            } else {
                                const uint32_t col0 = my_nt_d * per_core_N_d;
                                fast_il::Cursor<kNumDramBanks> dc;
                                dc.start(row * N_down_tiles_full + col0);
                                for (uint32_t n = 0; n < per_core_N_d; ++n) {
                                    if (row < K_down_tiles && col0 + n < N_down_tiles_full) {
                                        noc_async_read(
                                            dc.addr(down_bases, down_tile_bytes), l1_w, down_tile_bytes, /*noc=*/1);
                                    }
                                    dc.next();
                                    l1_w += down_tile_bytes;
                                }
                            }
                        }
                        noc_up.async_read_barrier();
                        down_done_sem.set(down_seq);
                    }
                }
            }

            // ---- Drain cb_out (down matmul output) to DRAM ----
            // Per-chunk per_core_M (per_core_M_max for full chunks, a smaller
            // divisor for the tail); chunk starts are uniform at chunk*chunk_M_max.
            // Contiguous row map: this core owns rows [row0, row0 + per_core_M).
            const uint32_t per_core_M =
                adaptive_chunk::per_core_M_for_chunk(chunk, count_tiles, chunk_M_max, group_rows);
            const uint32_t row0 = chunk * chunk_M_max + my_mt * per_core_M;
            (void)0;  // column mapping below via group_assign::global_col (contiguous or strided)
            // The DOWN matmul packs and pushes the FULL compile-time-MAX ring (its
            // L1_ACC needs full-ring cycling), so DRAIN all d_in1_num_subblocks_M
            // rows to keep cb_out balanced — but only WRITE the first per_core_M
            // (runtime) rows; the rest are MAC-skipped zeros that map onto other
            // cores' rows and must not be emitted (the sb_m < per_core_M guard below).
            const uint32_t sb_m_bound = d_in1_num_subblocks_M;
            for (uint32_t sb_m = 0; sb_m < sb_m_bound; ++sb_m) {
                for (uint32_t sb_n = 0; sb_n < d_in1_num_subblocks_N; ++sb_n) {
                    cb_out_buf.wait_front(d_out_subblock_num_tiles);
                    uint32_t subblock_tile_offset = 0;
                    for (uint32_t i = 0; i < d_out_subblock_h; ++i) {
                        for (uint32_t j = 0; j < d_out_subblock_w; ++j) {
                            const uint32_t row = row0 + sb_m * d_out_subblock_h + i;
                            const uint32_t col = group_assign::global_col(
                                my_nt_d, sb_n * d_out_subblock_w + j, per_core_N_d, grid_x, col_strided);
                            // `row` indexes the FFN *input* (x) tile-rows; the
                            // destination tile-row adds the per-expert region
                            // offset.
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
                            if (sb_m < per_core_M && col < N_down_tiles_full && row < M_tiles_full &&
                                row < count_tiles) {
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
    }  // end work-item loop
    // Ensure all outstanding writes complete at the destination before the
    // kernel returns (the next dispatched op may read this output).
    noc.async_write_barrier();
    // UP_SPLIT issues only per-K-block-barriered NoC-1 `up` reads (no NoC-1
    // worker multicast and no NoC-1 atomics), so no extra NoC-1 drain is needed
    // here — which is exactly why it is safe beside the fabric CCL ops.
}
