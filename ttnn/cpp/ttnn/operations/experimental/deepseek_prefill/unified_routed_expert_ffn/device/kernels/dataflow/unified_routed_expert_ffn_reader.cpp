// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for unified_routed_expert_ffn.
//
// Per-core responsibilities, sequenced over `effective_chunks` chunks
// (effective_chunks = ceil(this expert's token count / chunk_M_tiles)):
//   - Read counts/idx_table scratch once at kernel start to discover this
//     expert's active token count. x tile reads start at row 0 unless
//     read_x_at_offset is set, in which case x is a shared buffer and this
//     expert's rows begin at start[global_id] (fusing what ttnn::extract did);
//     the reader adds (start / TILE) * K_gate_tiles to every x page index.
//   - Phase 1 (gate matmul, fused with phase 2): per K-block, sender at
//     gx=0 NoC-mcasts the x K-block to its M-row receivers (in0 mcast);
//     sender at gy=0 NoC-mcasts the gate+up K-block to its N-col
//     receivers (in1 mcast). Handshakes are per-K-block ready/valid
//     sems (in0_*_sem, in1_*_sem).
//   - Phase 4 (down matmul): per K-block kb, exactly one core in the
//     M-row (gx==kb) is the "activated sender" — it L1-mcasts its
//     cb_activated tiles to all M-row cores' cb_in0_down_full (with
//     loopback) using act_{ready,valid}_sem. The in1_down sender at
//     gy=0 mcasts the down K-block weight in parallel on the other NoC.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/debug/dprint.h"  // TEMP: config dump

void kernel_main() {
    // -------------------------- runtime args ------------------------------
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t gate_addr = get_arg_val<uint32_t>(1);
    const uint32_t up_addr = get_arg_val<uint32_t>(2);
    const uint32_t down_addr = get_arg_val<uint32_t>(3);
    const uint32_t counts_addr = get_arg_val<uint32_t>(4);
    const uint32_t idx_table_addr = get_arg_val<uint32_t>(5);

    const uint32_t my_mt = get_arg_val<uint32_t>(6);
    const uint32_t my_nt_gu = get_arg_val<uint32_t>(7);
    const uint32_t my_nt_d = get_arg_val<uint32_t>(8);

    // Weight-multicast runtime args (indices 9..18).
    const uint32_t is_in1_sender_u32 = get_arg_val<uint32_t>(9);
    const bool is_in1_sender = is_in1_sender_u32 != 0;
    const uint32_t in1_ready_sem_id = get_arg_val<uint32_t>(10);
    const uint32_t in1_valid_sem_id = get_arg_val<uint32_t>(11);
    const uint32_t in1_num_receivers = get_arg_val<uint32_t>(12);
    const uint32_t in1_mcast_nx_start = get_arg_val<uint32_t>(13);
    const uint32_t in1_mcast_ny_start = get_arg_val<uint32_t>(14);
    const uint32_t in1_mcast_nx_end = get_arg_val<uint32_t>(15);
    const uint32_t in1_mcast_ny_end = get_arg_val<uint32_t>(16);
    const uint32_t in1_sender_nx = get_arg_val<uint32_t>(17);
    const uint32_t in1_sender_ny = get_arg_val<uint32_t>(18);

    // x (in0) multicast runtime args (indices 19..28).
    const uint32_t is_in0_sender_u32 = get_arg_val<uint32_t>(19);
    const bool is_in0_sender = is_in0_sender_u32 != 0;
    const uint32_t in0_ready_sem_id = get_arg_val<uint32_t>(20);
    const uint32_t in0_valid_sem_id = get_arg_val<uint32_t>(21);
    const uint32_t in0_num_receivers = get_arg_val<uint32_t>(22);
    const uint32_t in0_mcast_nx_start = get_arg_val<uint32_t>(23);
    const uint32_t in0_mcast_ny_start = get_arg_val<uint32_t>(24);
    const uint32_t in0_mcast_nx_end = get_arg_val<uint32_t>(25);
    const uint32_t in0_mcast_ny_end = get_arg_val<uint32_t>(26);
    const uint32_t in0_sender_nx = get_arg_val<uint32_t>(27);
    const uint32_t in0_sender_ny = get_arg_val<uint32_t>(28);

    // Activated L1 mcast sems. Sender (gx == kb at phase-4 K-block kb) waits
    // on its act_ready_sem for GRID_X - 1 incs from the receivers; then
    // mcasts cb_activated -> all M-row cores' cb_in0_down_full L1; then
    // mcasts act_valid_sem to release receivers.
    const uint32_t act_ready_sem_id = get_arg_val<uint32_t>(29);
    const uint32_t act_valid_sem_id = get_arg_val<uint32_t>(30);

    // UP_SPLIT local handshake (reader <-> writer): up_go = slot reserved,
    // up_done = up block landed in L1. Monotonic; gy=0 in1-sender cores only.
    const uint32_t up_go_sem_id = get_arg_val<uint32_t>(31);
    const uint32_t up_done_sem_id = get_arg_val<uint32_t>(32);
    Semaphore<> up_go_sem(up_go_sem_id);
    Semaphore<> up_done_sem(up_done_sem_id);

    // X_SPLIT handshake (reader <-> writer, in0-sender cores): x_go = x slot
    // reserved, x_done = the writer's half of this K-block's x landed in cb_x_rm.
    // Appended after the M-row NoC table (2*GRID_X, compile arg 21) AND start_addr,
    // so at 33 + 2*GRID_X + 1 (x_go) and + 2 (x_done).
    const uint32_t x_go_sem_id = get_arg_val<uint32_t>(34 + 2 * get_compile_time_arg_val(21));
    const uint32_t x_done_sem_id = get_arg_val<uint32_t>(35 + 2 * get_compile_time_arg_val(21));
    Semaphore<> x_go_sem(x_go_sem_id);
    Semaphore<> x_done_sem(x_done_sem_id);

    // M-row NoC coord table: GRID_X (x, y) pairs starting at runtime arg 33.
    // Used to resolve the sender's NoC addr per phase-4 K-block kb (= gx).
    constexpr uint32_t M_ROW_NOC_RT_OFFSET = 33;

    // -------------------------- compile-time args -------------------------
    constexpr uint32_t cb_in0_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in1_gate = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in1_up = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in1_down = get_compile_time_arg_val(3);
    constexpr uint32_t cb_in0_down_full = get_compile_time_arg_val(4);
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(5);
    constexpr uint32_t cb_idx_scratch = get_compile_time_arg_val(6);

    constexpr uint32_t local_expert_id = get_compile_time_arg_val(7);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(8);
    constexpr uint32_t per_core_N_gu = get_compile_time_arg_val(9);
    constexpr uint32_t per_core_N_d = get_compile_time_arg_val(10);
    constexpr uint32_t K_gate_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t K_down_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t in0_block_w_gu = get_compile_time_arg_val(13);
    constexpr uint32_t in0_block_w_d = get_compile_time_arg_val(14);
    constexpr uint32_t N_gate_tiles_full = get_compile_time_arg_val(15);
    constexpr uint32_t N_down_tiles_full = get_compile_time_arg_val(16);
    constexpr uint32_t M_tiles_full = get_compile_time_arg_val(17);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(18);
    constexpr uint32_t chunk_M_tiles = get_compile_time_arg_val(19);
    constexpr uint32_t cb_activated = get_compile_time_arg_val(20);
    constexpr uint32_t GRID_X_NOC = get_compile_time_arg_val(21);  // M-row mcast group size
    constexpr uint32_t K_down_tiles_padded = get_compile_time_arg_val(22);
    // `up` read mode: reader_reads_up = reader does the DRAM read (LEGACY);
    // reader_mcasts_up = reader NoC-0 mcasts up (UP_SPLIT: writer reads it on
    // NoC 1 into cb_in1_up); both 0 = UP_WRITER_MCAST, reader skips up.
    constexpr uint32_t reader_reads_up = get_compile_time_arg_val(23);
    constexpr uint32_t reader_mcasts_up = get_compile_time_arg_val(24);
    // read_x_at_offset: x is a shared buffer; offset x reads by this expert's
    // region start (start[global_id]/TILE tile-rows). 0 => x is per-expert,
    // reads start at row 0. cb_start_scratch holds the fetched `start` page.
    constexpr uint32_t read_x_at_offset = get_compile_time_arg_val(25);
    constexpr uint32_t cb_start_scratch = get_compile_time_arg_val(26);
    // x_is_row_major: x is ROW_MAJOR bf16 — stream sticks into cb_x_rm for the
    // compute kernel to tilize. 0 => x is TILE bf8_b, read directly.
    constexpr uint32_t x_is_row_major = get_compile_time_arg_val(27);
    constexpr uint32_t cb_x_rm = get_compile_time_arg_val(28);
    // Tile height: rows (token-row sticks) per tile-row. Used to size row-major
    // reads and to convert token counts to tile-rows.
    constexpr uint32_t TILE_HEIGHT = get_compile_time_arg_val(29);
    // Byte size of one row-major x element: x is bf16 in the row-major path.
    constexpr uint32_t X_RM_ELEM_BYTES = get_compile_time_arg_val(30);
    // UP_SPLIT iff the reader multicasts up but does not read it from DRAM.
    constexpr bool up_split = (reader_mcasts_up != 0) && (reader_reads_up == 0);

    constexpr uint32_t g_in0_block_num_tiles = per_core_M * in0_block_w_gu;
    constexpr uint32_t g_in1_block_num_tiles = per_core_N_gu * in0_block_w_gu;
    constexpr uint32_t d_in0_block_num_tiles = per_core_M * in0_block_w_d;
    constexpr uint32_t d_in1_block_num_tiles = per_core_N_d * in0_block_w_d;
    constexpr uint32_t num_blocks_gu = K_gate_tiles / in0_block_w_gu;
    constexpr uint32_t num_blocks_d = K_down_tiles_padded / in0_block_w_d;

    constexpr uint32_t x_accessor_offset = 31;
    constexpr auto x_args = TensorAccessorArgs<x_accessor_offset>();
    const auto x_acc = TensorAccessor(x_args, x_addr, get_tile_size(cb_in0_x));
    // Row-major x accessor (x_is_row_major): x is a ROW_MAJOR bf16 buffer whose
    // page is one token-row stick = emb elements = K_gate_tiles*TILE_HEIGHT bf16
    // (X_RM_ELEM_BYTES each). Partial-stick reads index page_id=token-row,
    // offset_bytes=column window. Only used in the row-major path; the tile-page
    // x_acc above serves the TILE path.
    constexpr uint32_t x_rm_stick_bytes = K_gate_tiles * TILE_HEIGHT * X_RM_ELEM_BYTES;
    const auto x_acc_rm = TensorAccessor(x_args, x_addr, x_rm_stick_bytes);

    constexpr uint32_t gate_accessor_offset = x_args.next_compile_time_args_offset();
    constexpr auto gate_args = TensorAccessorArgs<gate_accessor_offset>();
    const auto gate_acc = TensorAccessor(gate_args, gate_addr, get_tile_size(cb_in1_gate));

    constexpr uint32_t up_accessor_offset = gate_args.next_compile_time_args_offset();
    constexpr auto up_args = TensorAccessorArgs<up_accessor_offset>();
    const auto up_acc = TensorAccessor(up_args, up_addr, get_tile_size(cb_in1_up));

    constexpr uint32_t down_accessor_offset = up_args.next_compile_time_args_offset();
    constexpr auto down_args = TensorAccessorArgs<down_accessor_offset>();
    const auto down_acc = TensorAccessor(down_args, down_addr, get_tile_size(cb_in1_down));

    constexpr uint32_t counts_accessor_offset = down_args.next_compile_time_args_offset();
    constexpr auto counts_args = TensorAccessorArgs<counts_accessor_offset>();
    const auto counts_acc = TensorAccessor(counts_args, counts_addr);

    constexpr uint32_t idx_accessor_offset = counts_args.next_compile_time_args_offset();
    constexpr auto idx_args = TensorAccessorArgs<idx_accessor_offset>();
    const auto idx_acc = TensorAccessor(idx_args, idx_table_addr);

    // `start` (= expert_region_offsets) accessor. Appended last in the reader's
    // accessor stream; read only when read_x_at_offset. start_addr is the final
    // runtime arg, after the GRID_X-pair M-row NoC table at M_ROW_NOC_RT_OFFSET.
    const uint32_t start_addr = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * GRID_X_NOC);
    constexpr uint32_t start_accessor_offset = idx_args.next_compile_time_args_offset();
    constexpr auto start_args = TensorAccessorArgs<start_accessor_offset>();
    const auto start_acc = TensorAccessor(start_args, start_addr);

    // D2.0 NoC handles. `noc` uses default noc_index for mcasts/sem ops.
    // `noc_read` forces NoC 0 for DRAM weight/page reads — the kernel issues
    // those concurrently with mcast traffic on the kernel's default NoC for
    // dual-NoC parallelism (see in1_down + activated mcast in phase 4).
    Noc noc;
    Noc noc_read(0);

    // D2.0 CircularBuffer wrappers for method-form access. Compile-time CB
    // indices are unchanged.
    CircularBuffer cb_in0_x_obj(cb_in0_x);
    CircularBuffer cb_in1_gate_obj(cb_in1_gate);
    CircularBuffer cb_in1_up_obj(cb_in1_up);
    CircularBuffer cb_in1_down_obj(cb_in1_down);
    CircularBuffer cb_in0_down_full_obj(cb_in0_down_full);
    CircularBuffer cb_counts_scratch_obj(cb_counts_scratch);
    CircularBuffer cb_idx_scratch_obj(cb_idx_scratch);
    CircularBuffer cb_start_scratch_obj(cb_start_scratch);
    CircularBuffer cb_x_rm_obj(cb_x_rm);
    CircularBuffer cb_activated_obj(cb_activated);

    // x staging CB for the in0 path. Row-major: the reader fills + mcasts
    // cb_x_rm (bf16 row-major) and the compute kernel tilizes it into cb_in0_x.
    // TILE: the reader fills + mcasts cb_in0_x directly. reserve / mcast /
    // push all operate on x_stage_obj; only the DRAM read loop differs
    // (partial-stick vs tile-page). In row-major mode the reader never touches
    // cb_in0_x — compute produces it.
    constexpr uint32_t x_stage_cb = (x_is_row_major != 0) ? cb_x_rm : cb_in0_x;
    CircularBuffer x_stage_obj(x_stage_cb);
    const uint32_t x_stage_tile_bytes = x_stage_obj.get_tile_size();

    // D2.0 Semaphore wrappers.
    Semaphore<> in1_ready_sem(in1_ready_sem_id);
    Semaphore<> in1_valid_sem(in1_valid_sem_id);
    Semaphore<> in0_ready_sem(in0_ready_sem_id);
    Semaphore<> in0_valid_sem(in0_valid_sem_id);
    Semaphore<> act_ready_sem(act_ready_sem_id);
    Semaphore<> act_valid_sem(act_valid_sem_id);

    // Look up active token count for this expert from device-side buffers.
    // Reserve+read+push so the compute kernel (TRISC) and writer kernel
    // (NCRISC) can cb_wait_front on these CBs and read the same L1 data.
    //
    // Each scratch CB is a single page sized (host-side) to hold up to
    // MAX_GLOBAL_EXPERTS UINT32 entries, so `1` here is the whole buffer and a
    // single async_read lands the entire counts / idx vector. The
    // later counts_ptr[global_expert_id] / idx_ptr[local_expert_id] indexing
    // therefore stays in-bounds for any model up to MAX_GLOBAL_EXPERTS experts
    // (DeepSeek V3 256, Kimi 384, ... up to 1024).
    cb_counts_scratch_obj.reserve_back(1);
    cb_idx_scratch_obj.reserve_back(1);
    const uint32_t counts_l1 = cb_counts_scratch_obj.get_write_ptr();
    const uint32_t idx_l1 = cb_idx_scratch_obj.get_write_ptr();
    const uint32_t counts_page_size = counts_acc.get_aligned_page_size();
    const uint32_t idx_page_size = idx_acc.get_aligned_page_size();
    noc_read.async_read(counts_acc, CoreLocalMem<uint32_t>(counts_l1), counts_page_size, {.page_id = 0}, {});
    noc_read.async_read(idx_acc, CoreLocalMem<uint32_t>(idx_l1), idx_page_size, {.page_id = 0}, {});
    noc_read.async_read_barrier();
    cb_counts_scratch_obj.push_back(1);
    cb_idx_scratch_obj.push_back(1);

    const volatile tt_l1_ptr uint32_t* counts_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_l1);
    const volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    const uint32_t global_expert_id = idx_ptr[local_expert_id];
    const uint32_t count_value = counts_ptr[global_expert_id];
    // count_value is in TOKEN rows. Convert to tile rows (ceil) then to chunks.
    // For count=0 the loop is empty (no chunks processed). For count > 0 we
    // process ceil(count_tiles / chunk_M_tiles) chunks; the remaining chunks
    // (if any) are skipped — no DRAM reads, no mcasts, no compute.
    const uint32_t count_tiles = (count_value + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t effective_chunks_runtime = (count_tiles + chunk_M_tiles - 1) / chunk_M_tiles;
    // Clamp to compile-time num_chunks just in case (defensive against bad input).
    const uint32_t effective_chunks = effective_chunks_runtime < num_chunks ? effective_chunks_runtime : num_chunks;

    // TEMP config dump (one core): understand RM vs TILE blocking difference.
    if (is_in0_sender && my_mt == 0 && my_nt_gu == 0) {
        DPRINT(
            "CFG rm="
            "{}"
            " ibw_gu={} per_core_M={} chunk_M={} num_chunks={} eff={} K_gate={}\n",
            (uint32_t)x_is_row_major,
            in0_block_w_gu,
            per_core_M,
            chunk_M_tiles,
            num_chunks,
            effective_chunks,
            K_gate_tiles);
    }

    // x-read row offset. Zero unless read_x_at_offset: then x is a shared buffer
    // and this expert's rows begin at start[global_id]. Fetch the start page and
    // convert the token row to a tile-page offset (row_tile * K_gate_tiles), the
    // exact source rebase ttnn::extract used to perform. Must agree with the
    // writer's row_offset_tiles so x is read and y is written to the same region.
    uint32_t x_start_tile_idx = 0;
    // Row-major stick (token-row) base for this expert's region. TILE mode uses
    // x_start_tile_idx (tile-page offset); row-major mode uses x_start_stick to
    // offset the token-row stick page. Both derive from start[global_id] (a
    // tile-aligned token row) and must agree with the writer's row_offset_tiles.
    uint32_t x_start_stick = 0;
    if constexpr (read_x_at_offset != 0) {
        cb_start_scratch_obj.reserve_back(1);
        const uint32_t start_l1 = cb_start_scratch_obj.get_write_ptr();
        noc_read.async_read(
            start_acc, CoreLocalMem<uint32_t>(start_l1), start_acc.get_aligned_page_size(), {.page_id = 0}, {});
        noc_read.async_read_barrier();
        cb_start_scratch_obj.push_back(1);
        const volatile tt_l1_ptr uint32_t* start_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(start_l1);
        const uint32_t start_value = start_ptr[global_expert_id];
        x_start_tile_idx = (start_value / TILE_HEIGHT) * K_gate_tiles;
        x_start_stick = start_value;
    }

    const uint32_t x_tile_bytes = get_tile_size(cb_in0_x);
    const uint32_t gate_tile_bytes = get_tile_size(cb_in1_gate);
    const uint32_t up_tile_bytes = get_tile_size(cb_in1_up);
    const uint32_t down_tile_bytes = get_tile_size(cb_in1_down);

    // Weight-multicast helper. For each in1 K-block:
    //   * Sender (gy=0): wait for all GRID_Y-1 receivers to inc the local
    //     ready_sem. Reset ready_sem. Read in1 from DRAM into local cb_in1.
    //     Multicast the L1 region to receivers. Multicast valid_sem=1.
    //   * Receiver: reserve cb space. Reset local valid_sem=0. Increment
    //     sender's ready_sem at sender's NoC coord. Wait local valid_sem=1.
    //
    // Both sender and receiver finish with the K-block of in1 in their own
    // cb_in1 L1, ready for cb_push_back/compute.
    constexpr uint32_t IN1_VALID = 1;
    constexpr uint32_t IN0_VALID = 1;

    // UP_SPLIT handshake counter, kept in lockstep with the writer's.
    uint32_t up_seq = 0;

    // X_SPLIT: on the row-major path, hand the first x_split_sticks token-row
    // sticks of each K-block's x DRAM read to the writer (NoC 1); the reader reads
    // the rest on NoC 0. Stick-granularity (not row) so the split is a true 50/50
    // of the x bytes, balancing the NoC-0-heavy x+gate against the slack NoC 1.
    uint32_t x_seq = 0;
    constexpr bool x_split = (x_is_row_major != 0) && (per_core_M >= 2);
    constexpr uint32_t x_total_sticks = per_core_M * TILE_HEIGHT;
    constexpr uint32_t x_split_sticks = x_split ? (x_total_sticks / 2) : 0;

    // Bound the chunk loop by effective_chunks (= ceil_div(count, chunk_M_tiles))
    // so this expert only does work proportional to its actual token count,
    // not the max-tokens-padded shape of the input. Eliminates the host-side
    // count read that previously had to narrow the input tensor.
    for (uint32_t chunk = 0; chunk < effective_chunks; ++chunk) {
        const uint32_t this_core_first_row = chunk * chunk_M_tiles + my_mt * per_core_M;

        // -------- PHASES 1+2 fused — push x ONCE per K-block, then gate then up.
        //
        // Per-K-block restructure for parallelism:
        //   Previously each K-block ran in0 mcast section FULLY then in1 mcast
        //   section FULLY in series on every core. That meant the kernel walked
        //   through every K-block as ~30µs(in0) + ~30µs(in1) = 60µs.
        //   Now we ack BOTH ready semaphores up-front (receivers signal both
        //   senders before either sender starts work), then both senders run
        //   their DRAM-read + NoC-mcast in parallel (in0 sender on M-row,
        //   in1 sender on N-col are disjoint sets of cores except (0,0)).
        //   Receivers wait for BOTH valid semaphores at the end. Halves the
        //   per-K-block elapsed time at small per_core_M where mcast/handshake
        //   overhead dominates compute.
        for (uint32_t kb = 0; kb < num_blocks_gu; ++kb) {
            x_stage_obj.reserve_back(g_in0_block_num_tiles);
            cb_in1_gate_obj.reserve_back(g_in1_block_num_tiles);
            if constexpr (reader_mcasts_up) {
                cb_in1_up_obj.reserve_back(g_in1_block_num_tiles);
            }

            // UP_SPLIT: slot reserved -> release writer to read `up` on NoC 1,
            // concurrent with the reader's NoC-0 `gate` read below.
            if constexpr (up_split) {
                if (is_in1_sender) {
                    ++up_seq;
                    up_go_sem.set(up_seq);
                }
            }

            // X_SPLIT: slot reserved above -> release the writer to read its
            // M-rows of x on NoC 1, concurrent with the reader's NoC-0 reads.
            if constexpr (x_split) {
                if (is_in0_sender) {
                    ++x_seq;
                    x_go_sem.set(x_seq);
                }
            }

            // Step 1: receivers ack BOTH senders upfront so both senders can
            // proceed in parallel. The senders are usually disjoint sets of
            // cores; the only core that's both senders is (0,0) which doesn't
            // self-inc (it's its own sender for both).
            if (!is_in0_sender) {
                in0_valid_sem.set(0);
                in0_ready_sem.up(noc, in0_sender_nx, in0_sender_ny, 1);
            }
            if (!is_in1_sender) {
                in1_valid_sem.set(0);
                in1_ready_sem.up(noc, in1_sender_nx, in1_sender_ny, 1);
            }

            // Step 2: senders run their work. in0 sender path and in1 sender
            // path can each start as soon as their ready sem is satisfied —
            // for the common case where a core is one type of sender, the
            // work begins immediately. For core (0,0) (both senders), in0
            // runs first then in1, ~60µs sequentially — same as before.
            if (is_in0_sender) {
                {
                    DeviceZoneScopedN("IN0-READY-WAITING");
                    in0_ready_sem.wait(in0_num_receivers);
                    in0_ready_sem.set(0);
                }

                uint32_t l1_x = x_stage_obj.get_write_ptr();
                const uint32_t block_start = l1_x;
                // Rows past count_tiles are invalid. Skip the read; their L1 keeps
                // stale content. Harmless: the writer only stores output tiles with
                // row < count_tiles and the matmul is per-row independent, so the
                // garbage stays confined to output rows that are never written.
                {  // TEMP profiling: time the x DRAM read (row-major strided vs tiled)
                    DeviceZoneScopedN("X-READ");
                    if constexpr (x_is_row_major != 0) {
                        // Row-major: read this K-block's column window (in0_block_w_gu
                        // tiles wide) from each of the TILE_HEIGHT token-row sticks of
                        // every tile-row, laid contiguously so each tile-row forms one
                        // TILE_HEIGHT x (in0_block_w_gu*32) strip for tilize_block.
                        // page_id is the token-row stick; offset_bytes is the K-block
                        // column window within the emb-wide stick.
                        constexpr uint32_t rm_kblock_bytes = in0_block_w_gu * TILE_HEIGHT * X_RM_ELEM_BYTES;
                        const uint32_t col_off_bytes = kb * rm_kblock_bytes;
                        // X_SPLIT: the writer reads sticks [0, x_split_sticks) on NoC 1
                        // into this same L1 block; the reader reads sticks
                        // [x_split_sticks, x_total_sticks) on NoC 0. Index by absolute
                        // stick so both sides target the same contiguous L1 slots; the
                        // per-stick row (s / TILE_HEIGHT) drives the invalid-row skip.
                        for (uint32_t s = x_split_sticks; s < x_total_sticks; ++s) {
                            const uint32_t m = s / TILE_HEIGHT;
                            const uint32_t r = s % TILE_HEIGHT;
                            const uint32_t tile_row = this_core_first_row + m;
                            if (tile_row < count_tiles) {
                                // x_start_stick offsets into this expert's region of the
                                // shared row-major buffer (0 for a standalone buffer).
                                const uint32_t stick = x_start_stick + tile_row * TILE_HEIGHT + r;
                                noc_read.async_read(
                                    x_acc_rm,
                                    CoreLocalMem<uint32_t>(block_start + s * rm_kblock_bytes),
                                    rm_kblock_bytes,
                                    {.page_id = stick, .offset_bytes = col_off_bytes},
                                    {});
                            }
                            // Invalid row: skip; the L1 slot at s*rm_kblock_bytes keeps stale content.
                        }
                    } else {
                        for (uint32_t m = 0; m < per_core_M; ++m) {
                            const uint32_t row = this_core_first_row + m;
                            const bool row_valid = row < count_tiles;
                            if (row_valid) {
                                for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
                                    const uint32_t col = kb * in0_block_w_gu + k;
                                    // x_start_tile_idx offsets into this expert's region
                                    // of a shared buffer (0 when x is per-expert).
                                    const uint32_t tile_idx = x_start_tile_idx + row * K_gate_tiles + col;
                                    noc_read.async_read(
                                        x_acc, CoreLocalMem<uint32_t>(l1_x), x_tile_bytes, {.page_id = tile_idx}, {});
                                    l1_x += x_tile_bytes;
                                }
                            } else {
                                // Invalid row: skip the read, L1 keeps stale content. Just advance.
                                l1_x += in0_block_w_gu * x_tile_bytes;
                            }
                        }
                    }
                    noc_read.async_read_barrier();
                }  // end TEMP XREAD profiling zone

                // X_SPLIT: the writer's sticks [0, x_split_sticks) must have landed
                // in cb_x_rm on NoC 1 before this core multicasts the full block.
                if constexpr (x_split) {
                    x_done_sem.wait_min(x_seq);
                }

                {
                    DeviceZoneScopedN("IN0-MULTICAST");

                    const uint32_t block_bytes = g_in0_block_num_tiles * x_stage_tile_bytes;
                    // linked=true keeps the multicast path RESERVED so the in0_valid
                    // sem multicast below travels the SAME path and is delivered
                    // AFTER the data at every receiver. With linked=false the path is
                    // released and the (posted) valid-sem multicast can overtake the
                    // bulk data multicast at a receiver under NoC contention (heavy
                    // fabric load) -> the receiver observes in0_valid, pushes
                    // cb_in0_x, and compute reads STALE x from L1 -> wrong gate/up
                    // matmul output for that core (rare, timing-dependent). A write
                    // barrier does NOT fix this on Blackhole (multicast writes are
                    // posted; no completion ack). Mirrors the phase-4 activated mcast.
                    noc.async_write_multicast(
                        CoreLocalMem<uint32_t>(block_start),
                        MulticastEndpoint{},
                        block_bytes,
                        in0_num_receivers,
                        {.offset_bytes = 0},
                        {.noc_x_start = in0_mcast_nx_start,
                         .noc_y_start = in0_mcast_ny_start,
                         .noc_x_end = in0_mcast_nx_end,
                         .noc_y_end = in0_mcast_ny_end,
                         .addr = block_start},
                        /*linked=*/true);
                    x_stage_obj.push_back(g_in0_block_num_tiles);

                    noc.async_writes_flushed();
                    in0_valid_sem.set(IN0_VALID);
                    in0_valid_sem.set_multicast<NocOptions::DEFAULT>(
                        noc,
                        in0_mcast_nx_start,
                        in0_mcast_ny_start,
                        in0_mcast_nx_end,
                        in0_mcast_ny_end,
                        in0_num_receivers);
                }
            }

            if (is_in1_sender) {
                {
                    DeviceZoneScopedN("IN1-READY-WAITING");
                    in1_ready_sem.wait(in1_num_receivers);
                    in1_ready_sem.set(0);
                }

                // Sender-scope L1 start for the gate block, so the GATE-MULTICAST
                // below can address it (mirrors up_block_start). No cb push touches
                // cb_in1_gate between here and the mcast, so the ptr is stable.
                const uint32_t gate_block_start = cb_in1_gate_obj.get_write_ptr();
                {
                    DeviceZoneScopedN("GATE-READ-ISSUE");
                    // DRAM read gate region first.
                    uint32_t l1_w_gate = gate_block_start;
                    for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
                        for (uint32_t n = 0; n < per_core_N_gu; ++n) {
                            const uint32_t row = kb * in0_block_w_gu + k;
                            const uint32_t col = my_nt_gu * per_core_N_gu + n;
                            if (col < N_gate_tiles_full) {
                                const uint32_t tile_idx = row * N_gate_tiles_full + col;
                                noc_read.async_read(
                                    gate_acc,
                                    CoreLocalMem<uint32_t>(l1_w_gate),
                                    gate_tile_bytes,
                                    {.page_id = tile_idx},
                                    {});
                            } else {
                                volatile tt_l1_ptr uint64_t* p =
                                    reinterpret_cast<volatile tt_l1_ptr uint64_t*>(l1_w_gate);
                                for (uint32_t i = 0; i < gate_tile_bytes / 8; ++i) {
                                    p[i] = 0;
                                }
                            }
                            l1_w_gate += gate_tile_bytes;
                        }
                    }
                }
                // `up` slot. LEGACY: reader reads it on NoC 0. UP_SPLIT: writer
                // already read it on NoC 1; reader just takes the L1 start and
                // waits on up_done (below) before mcasting.
                uint32_t up_block_start = 0;
                if constexpr (reader_mcasts_up) {
                    up_block_start = cb_in1_up_obj.get_write_ptr();
                }
                if constexpr (reader_reads_up) {
                    uint32_t l1_w_up = up_block_start;
                    for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
                        for (uint32_t n = 0; n < per_core_N_gu; ++n) {
                            const uint32_t row = kb * in0_block_w_gu + k;
                            const uint32_t col = my_nt_gu * per_core_N_gu + n;
                            if (col < N_gate_tiles_full) {
                                const uint32_t tile_idx = row * N_gate_tiles_full + col;
                                noc_read.async_read(
                                    up_acc, CoreLocalMem<uint32_t>(l1_w_up), up_tile_bytes, {.page_id = tile_idx}, {});
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
                }
                noc_read.async_read_barrier();

                // GRID_Y == 1: no column receivers — skip mcast/valid-sem; the
                // locally-read weights go straight to compute via cb_push_back.
                // GATE first (needs no `up`), so the reader issues the gate mcast
                // immediately and the up_done wait below overlaps the writer's
                // NoC-1 up read.
                if (in1_num_receivers > 0) {
                    const uint32_t gate_block_bytes = g_in1_block_num_tiles * gate_tile_bytes;
                    // The LAST in1 data multicast before the in1_valid sem must
                    // be linked=true so the (posted) valid-sem multicast travels
                    // the SAME reserved path and lands AFTER the data at every
                    // receiver. Otherwise, under NoC contention (heavy fabric
                    // load), the valid sem can overtake the weight data -> the
                    // receiver pushes cb_in1_{gate,up} and compute reads STALE
                    // weights -> wrong matmul output (rare, timing-dependent;
                    // a flush/barrier does not fix posted multicast writes on
                    // Blackhole). gate links into up (issued next, after the
                    // up_done wait) and up holds the path for the sem.
                    {
                        DeviceZoneScopedN("GATE-MULTICAST");

                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(gate_block_start),
                            MulticastEndpoint{},
                            gate_block_bytes,
                            in1_num_receivers,
                            {.offset_bytes = 0},
                            {.noc_x_start = in1_mcast_nx_start,
                             .noc_y_start = in1_mcast_ny_start,
                             .noc_x_end = in1_mcast_nx_end,
                             .noc_y_end = in1_mcast_ny_end,
                             .addr = gate_block_start},
                            /*linked=*/true);
                    }
                }

                // UP_SPLIT: gate is already issued on NoC 0; NOW wait for the
                // writer's NoC-1 up read to land before touching cb_in1_up. After
                // the gate mcast so the wait overlaps the writer's up read;
                // unconditional (outside the receivers guard) so GRID_Y==1 — which
                // pushes cb_in1_up straight to compute below without an mcast —
                // still gates on up_done. A semaphore poll issues no NoC op, so the
                // gate mcast's linked path stays reserved for the up mcast.
                if constexpr (up_split) {
                    DeviceZoneScopedN("UP-DONE-WAITING");
                    up_done_sem.wait_min(up_seq);
                }

                if (in1_num_receivers > 0) {
                    if constexpr (reader_mcasts_up) {
                        DeviceZoneScopedN("UP-MULTICAST");
                        const uint32_t up_block_bytes = g_in1_block_num_tiles * up_tile_bytes;
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(up_block_start),
                            MulticastEndpoint{},
                            up_block_bytes,
                            in1_num_receivers,
                            {.offset_bytes = 0},
                            {.noc_x_start = in1_mcast_nx_start,
                             .noc_y_start = in1_mcast_ny_start,
                             .noc_x_end = in1_mcast_nx_end,
                             .noc_y_end = in1_mcast_ny_end,
                             .addr = up_block_start},
                            /*linked=*/true);
                    }
                }

                cb_in1_gate_obj.push_back(g_in1_block_num_tiles);
                if constexpr (reader_mcasts_up) {
                    cb_in1_up_obj.push_back(g_in1_block_num_tiles);
                }

                if (in1_num_receivers > 0) {
                    noc.async_writes_flushed();

                    in1_valid_sem.set(IN1_VALID);
                    in1_valid_sem.set_multicast<NocOptions::DEFAULT>(
                        noc, in1_mcast_nx_start, in1_mcast_ny_start, in1_mcast_nx_end, in1_mcast_ny_end, in1_num_receivers);
                }
            }

            // Step 3: receivers wait for both valid semaphores and push.
            if (!is_in0_sender) {
                DeviceZoneScopedN("IN0-VALID-WAITING");
                in0_valid_sem.wait(IN0_VALID);
                x_stage_obj.push_back(g_in0_block_num_tiles);
            }
            if (!is_in1_sender) {
                DeviceZoneScopedN("IN1-VALID-WAITING");
                in1_valid_sem.wait(IN1_VALID);
                cb_in1_gate_obj.push_back(g_in1_block_num_tiles);
                if constexpr (reader_mcasts_up) {
                    cb_in1_up_obj.push_back(g_in1_block_num_tiles);
                }
            }
        }

        // -------- PHASE 4 — down matmul feed via L1 mcast of activated + down weight mcast --
        //
        // For each K-block kb (0..num_blocks_d-1=7), exactly one core in this
        // M-row is the "activated sender": the core at gx == kb. Its
        // cb_activated holds the per_core_M x per_core_N_gu tiles whose
        // hidden-column range matches K-block kb. Sender mcasts those tiles
        // (with loopback to its own L1) to every M-row core's cb_in0_down_full.
        //
        // We compute the mcast destination rectangle from the M-row NoC table
        // once: corners are mrow[0] (top-left) and mrow[GRID_X-1] (bottom-right).
        // Sender NoC addr for the per-K-block ready-sem inc is looked up by
        // index kb from the same table.
        // GRID_X_NOC comes from compile-time arg now (= GRID_X in program factory).
        const uint32_t mrow_first_nx = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 0);
        const uint32_t mrow_first_ny = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 1);
        const uint32_t mrow_last_nx = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * (GRID_X_NOC - 1) + 0);
        const uint32_t mrow_last_ny = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * (GRID_X_NOC - 1) + 1);
        constexpr uint32_t ACT_VALID = 1;
        const uint32_t intermed_tile_bytes = cb_in0_down_full_obj.get_tile_size();

        for (uint32_t kb = 0; kb < num_blocks_d; ++kb) {
            const bool is_act_sender = (my_nt_d == kb);

            cb_in1_down_obj.reserve_back(d_in1_block_num_tiles);
            cb_in0_down_full_obj.reserve_back(d_in0_block_num_tiles);

            // Step 1: receivers ack BOTH senders (in1_down and act) at the
            // top of the K-block iter. The in1_down ack lets the in1_down
            // sender immediately start DRAM reads; the act ack lets the act
            // sender start mcasting as soon as compute pushes cb_activated.
            // Without the early act ack the sender would only see receivers
            // after the in1_down section finishes, serializing the two paths.
            if (!is_in1_sender) {
                in1_valid_sem.set(0);
                in1_ready_sem.up(noc, in1_sender_nx, in1_sender_ny, 1);
            }
            if (!is_act_sender) {
                act_valid_sem.set(0);
                const uint32_t sender_nx = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * kb + 0);
                const uint32_t sender_ny = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * kb + 1);
                act_ready_sem.up(noc, sender_nx, sender_ny, 1);
            }

            // Step 2: in1_down sender kicks off DRAM reads (NoC 0) without
            // barriering — reads run concurrently with the activated mcast
            // below on NoC 1.
            uint32_t in1_block_start = 0;
            if (is_in1_sender) {
                {
                    DeviceZoneScopedN("IN1-DOWN-READY-WAITING");
                    in1_ready_sem.wait(in1_num_receivers);
                    in1_ready_sem.set(0);
                }
                uint32_t l1_w = cb_in1_down_obj.get_write_ptr();
                in1_block_start = l1_w;
                {
                    DeviceZoneScopedN("DOWN-READ-ISSUE");
                    for (uint32_t k = 0; k < in0_block_w_d; ++k) {
                        for (uint32_t n = 0; n < per_core_N_d; ++n) {
                            const uint32_t row = kb * in0_block_w_d + k;
                            const uint32_t col = my_nt_d * per_core_N_d + n;
                            if (row < K_down_tiles && col < N_down_tiles_full) {
                                const uint32_t tile_idx = row * N_down_tiles_full + col;
                                noc_read.async_read(
                                    down_acc, CoreLocalMem<uint32_t>(l1_w), down_tile_bytes, {.page_id = tile_idx}, {});
                            } else {
                                volatile tt_l1_ptr uint64_t* p = reinterpret_cast<volatile tt_l1_ptr uint64_t*>(l1_w);
                                for (uint32_t i = 0; i < down_tile_bytes / 8; ++i) {
                                    p[i] = 0;
                                }
                            }
                            l1_w += down_tile_bytes;
                        }
                    }
                }
            }

            // Step 3: activated L1 mcast (sender for this K-block = gx==kb).
            // act_sender starts as soon as compute pushes cb_activated AND
            // the ready acks are in (already done in step 1).
            if (is_act_sender) {
                cb_activated_obj.wait_front(d_in0_block_num_tiles);
                act_ready_sem.wait(GRID_X_NOC - 1);
                act_ready_sem.set(0);

                const uint32_t src_l1 = cb_activated_obj.get_read_ptr();
                const uint32_t dst_l1 = cb_in0_down_full_obj.get_write_ptr();
                const uint32_t mcast_bytes = d_in0_block_num_tiles * intermed_tile_bytes;
                // linked=true keeps the multicast path RESERVED so the
                // valid-semaphore multicast below travels the SAME path and is
                // delivered AFTER the data at every receiver. With linked=false
                // the path is released and the (posted) valid-sem multicast can
                // overtake the bulk data multicast at a receiver -> the receiver
                // observes act_valid, pushes cb_in0_down_full, and compute reads
                // stale L1 -> that core's whole down-matmul output block is wrong
                // (run-to-run nondeterministic). A write barrier does NOT fix this
                // on Blackhole (multicast writes are posted; no completion ack to
                // wait on) — only path-linking orders the sem behind the data.
                // Mirrors the canonical matmul in0 sender
                // (reader_bmm_tile_layout_in0_sender_padding.cpp).
                {
                    DeviceZoneScopedN("ACTIVATION-MULTICAST");
                    noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                        CoreLocalMem<uint32_t>(src_l1),
                        MulticastEndpoint{},
                        mcast_bytes,
                        GRID_X_NOC,
                        {.offset_bytes = 0},
                        {.noc_x_start = mrow_first_nx,
                         .noc_y_start = mrow_first_ny,
                         .noc_x_end = mrow_last_nx,
                         .noc_y_end = mrow_last_ny,
                         .addr = dst_l1},
                        /*linked=*/true);
                    noc.async_writes_flushed();

                    act_valid_sem.set(ACT_VALID);
                    act_valid_sem.set_multicast<NocOptions::MCAST_INCL_SRC>(
                        noc, mrow_first_nx, mrow_first_ny, mrow_last_nx, mrow_last_ny, GRID_X_NOC);

                    cb_activated_obj.pop_front(d_in0_block_num_tiles);
                }
            }

            // Step 4: in1_down sender finishes — barrier on DRAM reads (NoC 0,
            // in flight during step 3 activated mcast on NoC 1), then mcast.
            if (is_in1_sender) {
                noc_read.async_read_barrier();
                // GRID_Y == 1: no column receivers — skip mcast/valid-sem; this
                // core consumes the locally-read down weight directly.
                if (in1_num_receivers > 0) {
                    DeviceZoneScopedN("DOWN-MULTICAST");
                    const uint32_t block_bytes = d_in1_block_num_tiles * down_tile_bytes;
                    // linked=true so the in1_valid-sem multicast is ordered behind
                    // the weight data on the same reserved path (see the activated
                    // mcast above for the full rationale).
                    noc.async_write_multicast(
                        CoreLocalMem<uint32_t>(in1_block_start),
                        MulticastEndpoint{},
                        block_bytes,
                        in1_num_receivers,
                        {.offset_bytes = 0},
                        {.noc_x_start = in1_mcast_nx_start,
                         .noc_y_start = in1_mcast_ny_start,
                         .noc_x_end = in1_mcast_nx_end,
                         .noc_y_end = in1_mcast_ny_end,
                         .addr = in1_block_start},
                        /*linked=*/true);
                    noc.async_writes_flushed();

                    in1_valid_sem.set(IN1_VALID);
                    in1_valid_sem.set_multicast<NocOptions::DEFAULT>(
                        noc, in1_mcast_nx_start, in1_mcast_ny_start, in1_mcast_nx_end, in1_mcast_ny_end, in1_num_receivers);
                }
            }

            // Step 5: receivers wait for both valid sems and push.
            if (!is_act_sender) {
                DeviceZoneScopedN("ACTIVATION-WAITING");
                act_valid_sem.wait(ACT_VALID);
            }
            cb_in0_down_full_obj.push_back(d_in0_block_num_tiles);

            if (!is_in1_sender) {
                DeviceZoneScopedN("DOWN-WAITING");
                in1_valid_sem.wait(IN1_VALID);
            }
            cb_in1_down_obj.push_back(d_in1_block_num_tiles);
        }
    }  // end chunk loop

    // The last in-flight Semaphore<>::set_multicast (act_valid / in1_valid)
    // is a posted atomic; without an explicit barrier it can still be in
    // flight at kernel exit, leading to timing-dependent corruption.
    noc.async_atomic_barrier();
}
