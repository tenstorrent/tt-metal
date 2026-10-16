// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for unified_routed_expert_ffn.
//
// Per-core responsibilities, sequenced over `effective_chunks` chunks
// (effective_chunks = ceil(this expert's token count / chunk_M_tiles)):
//   - Read counts/idx_table scratch once at kernel start to discover this
//     expert's active token count. x is the shared dispatched buffer and this
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
#include "api/debug/assert.h"
#include "../adaptive_chunk.hpp"
#include "../weight_runs.hpp"

#ifdef RE_PACE_CB
// Cross-group pacing of the weight streams (see kernel_main below plan_launch). start() is the gate
// every weight sender passes at the start of each of its blocks.
struct Pacer {
    uint64_t board;    // NoC address of the progress board (one counter per group) on core (0,0)
    uint32_t snap_l1;  // this core's 32-byte snapshot of the board
    uint32_t groups;   // groups of the paced phase; 0 while this phase runs unpaced
    uint32_t me;       // this core's group
    bool pub;          // this core publishes its group's progress (its column-0 weight sender)

    // May this sender start its next block of `units` weight tiles? Only while its group is at most
    // kSlack weight tiles ahead of every group of the phase (a finished group's counter carries
    // kDone, so it never holds anyone). Blocking: poll the board (at most 64 reads, so a stale board
    // can only slow a block, never hang). Non-blocking (the look-aheads): judge the last snapshot and
    // hold without NoC traffic. Once going, the publisher adds `units` to its counter, and the
    // snapshot is refreshed by a read that lands under the block's own read barrier. Never called
    // inside an open linked multicast.
    // Add `units` to this group's counter (publisher only).
    __attribute__((noinline)) void publish(uint32_t units) {
        if (pub) {
            noc_semaphore_inc(board + 4 * me, units);
        }
    }
    template <uint32_t kSlack>
    __attribute__((noinline)) bool start(uint32_t units, bool blocking) {
        if (groups == 0) {
            return true;
        }
        volatile tt_l1_ptr uint32_t* snap = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(snap_l1);
        for (uint32_t poll = 0;; ++poll) {
            const uint32_t mine = snap[me];
            uint32_t h = 0;
            while (h < groups && mine <= snap[h] + kSlack) {
                ++h;
            }
            const bool go = h == groups || poll >= 64;
            if (!go && !blocking) {
                return false;
            }
            if (go) {
                publish(units);
            }
            noc_async_read<32>(board, snap_l1, 32);
            if (go) {
                return true;
            }
            noc_async_read_barrier();
        }
    }
};
constexpr uint32_t kPaceDone = 1u << 30;
#endif

void kernel_main() {
    // -------------------------- runtime args ------------------------------
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t counts_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t idx_table_addr = get_common_arg_val<uint32_t>(2);

    // Grid M-row of this core. The row WITHIN its expert group (my_mt) and the group's
    // weight-multicast topology are picked at runtime once the counts are known (see
    // adaptive_chunk::plan_launch); args 3 and 6..12 hold the whole-column topology and
    // are superseded by the per-layout sets at GROUP_TOPO_RT.
    const uint32_t my_gy = get_arg_val<uint32_t>(0);
    const uint32_t my_nt_gu = get_arg_val<uint32_t>(1);
    const uint32_t my_nt_d = get_arg_val<uint32_t>(2);

    const uint32_t in1_ready_sem_id = get_arg_val<uint32_t>(4);
    const uint32_t in1_valid_sem_id = get_arg_val<uint32_t>(5);

    // x (in0) multicast runtime args (indices 13..22).
    const uint32_t is_in0_sender_u32 = get_arg_val<uint32_t>(13);
    const bool is_in0_sender = is_in0_sender_u32 != 0;
    const uint32_t in0_ready_sem_id = get_arg_val<uint32_t>(14);
    const uint32_t in0_valid_sem_id = get_arg_val<uint32_t>(15);
    const uint32_t in0_num_receivers = get_arg_val<uint32_t>(16);
    const uint32_t in0_mcast_nx_start = get_arg_val<uint32_t>(17);
    const uint32_t in0_mcast_ny_start = get_arg_val<uint32_t>(18);
    const uint32_t in0_mcast_nx_end = get_arg_val<uint32_t>(19);
    const uint32_t in0_mcast_ny_end = get_arg_val<uint32_t>(20);
    const uint32_t in0_sender_nx = get_arg_val<uint32_t>(21);
    const uint32_t in0_sender_ny = get_arg_val<uint32_t>(22);

    // Activated L1 mcast sems. Sender (gx == kb at phase-4 K-block kb) waits
    // on its act_ready_sem for GRID_X - 1 incs from the receivers; then
    // mcasts cb_activated -> all M-row cores' cb_in0_down_full L1; then
    // mcasts act_valid_sem to release receivers.
    const uint32_t act_ready_sem_id = get_arg_val<uint32_t>(23);
    const uint32_t act_valid_sem_id = get_arg_val<uint32_t>(24);

    // UP_SPLIT local handshake (reader <-> writer): up_go = slot reserved,
    // up_done = up block landed in L1. Monotonic; gy=0 in1-sender cores only.
    const uint32_t up_go_sem_id = get_arg_val<uint32_t>(25);
    const uint32_t up_done_sem_id = get_arg_val<uint32_t>(26);
    Semaphore<> up_go_sem(up_go_sem_id);
    Semaphore<> up_done_sem(up_done_sem_id);

    // M-row NoC coord table: GRID_X (x, y) pairs starting at M_ROW_NOC_RT_OFFSET.
    // Used to resolve the sender's NoC addr per phase-4 K-block kb (= gx).
    // COUNTS_BCAST occupies 7 args before the M-row NoC table.
    constexpr uint32_t COUNTS_BCAST_RT = 27;
    // DOWN_SPLIT go/done sems sit between COUNTS_BCAST and the M-row NoC table.
    constexpr uint32_t DOWN_SEM_RT = COUNTS_BCAST_RT + 7;
    // IN1_WRITER_MCAST go/done sems follow the DOWN_SPLIT pair.
    constexpr uint32_t MCAST_SEM_RT = DOWN_SEM_RT + 2;
    constexpr uint32_t M_ROW_NOC_RT_OFFSET = MCAST_SEM_RT + 2;

    // -------------------------- compile-time args -------------------------
    constexpr uint32_t cb_in0_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in1_gate = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in1_up = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in1_down = get_compile_time_arg_val(3);
    constexpr uint32_t cb_in0_down_full = get_compile_time_arg_val(4);
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(5);
    constexpr uint32_t cb_idx_scratch = get_compile_time_arg_val(6);

    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(7);
    // per_core_M_max: the CB-sized maximum per-core M (= chunk_M_max / GRID_Y).
    // The RUNTIME per_core_M is picked from each expert's device token count below
    // and is <= this. CBs are allocated to the max; a smaller runtime pick uses fewer.
    constexpr uint32_t per_core_M_max = get_compile_time_arg_val(8);
    constexpr uint32_t per_core_N_gu = get_compile_time_arg_val(9);
    constexpr uint32_t per_core_N_d = get_compile_time_arg_val(10);
    constexpr uint32_t K_gate_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t K_down_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t in0_block_w_gu = get_compile_time_arg_val(13);
    constexpr uint32_t in0_block_w_d = get_compile_time_arg_val(14);
    constexpr uint32_t N_gate_tiles_full = get_compile_time_arg_val(15);
    constexpr uint32_t N_down_tiles_full = get_compile_time_arg_val(16);
    constexpr uint32_t M_tiles_full = get_compile_time_arg_val(17);
    // num_chunks_max: compile-time upper bound on the runtime chunk count (clamp).
    // chunk_M_max: CB-sized maximum chunk (per_core_M_max * GRID_Y); the runtime
    // picker never exceeds it.
    constexpr uint32_t num_chunks_max = get_compile_time_arg_val(18);
    constexpr uint32_t chunk_M_max = get_compile_time_arg_val(19);
    // The whole-grid chunk; the runtime chunk is per_core_M_max * group_rows (chunk_M below).
    static_assert(
        chunk_M_max == per_core_M_max * adaptive_chunk::kGridY, "chunk_M_max must be per_core_M_max * GRID_Y");
    constexpr uint32_t cb_activated = get_compile_time_arg_val(20);
    constexpr uint32_t GRID_X_NOC = get_compile_time_arg_val(21);  // M-row mcast group size
    // Per-layout weight-multicast topologies follow the M-row NoC table: one set of
    // GROUP_TOPO_ARGS per row-group layout (adaptive_chunk layout table order), each for this
    // core's group: {is_sender, num_receivers, mcast nx_start, ny_start, nx_end, ny_end, sender nx, sender ny}.
    constexpr uint32_t GROUP_TOPO_RT = M_ROW_NOC_RT_OFFSET + 2 * GRID_X_NOC;
    constexpr uint32_t GROUP_TOPO_ARGS = 8;
    constexpr uint32_t K_down_tiles_padded = get_compile_time_arg_val(22);
    // `up` read mode: reader_reads_up = reader does the DRAM read (LEGACY);
    // reader_mcasts_up = reader NoC-0 mcasts up (UP_SPLIT: writer reads it on
    // NoC 1 into cb_in1_up); both 0 = UP_WRITER_MCAST, reader skips up.
    constexpr uint32_t reader_reads_up = get_compile_time_arg_val(23);
    constexpr uint32_t reader_mcasts_up = get_compile_time_arg_val(24);
    // cb_start_scratch holds the fetched `start` page.
    constexpr uint32_t cb_start_scratch = get_compile_time_arg_val(25);
    // x_is_row_major: x is ROW_MAJOR bf16 — stream sticks into cb_x_rm for the
    // compute kernel to tilize. 0 => x is TILE bf8_b, read directly.
    constexpr uint32_t x_is_row_major = get_compile_time_arg_val(26);
    constexpr uint32_t cb_x_rm = get_compile_time_arg_val(27);
    // Tile height: rows (token-row sticks) per tile-row. Used to size row-major
    // reads and to convert token counts to tile-rows.
    constexpr uint32_t TILE_HEIGHT = get_compile_time_arg_val(28);
    // Byte size of one row-major x element: x is bf16 in the row-major path.
    constexpr uint32_t X_RM_ELEM_BYTES = get_compile_time_arg_val(29);
    // UP_SPLIT iff the reader multicasts up but does not read it from DRAM.
    constexpr bool up_split = (reader_mcasts_up != 0) && (reader_reads_up == 0);

    constexpr uint32_t g_in1_block_num_tiles = per_core_N_gu * in0_block_w_gu;
    // CB block sizes are the compile-time MAX and never vary with the runtime
    // per_core_M. Only the DRAM reads and the multicast payload below shrink to
    // the live rows (#50885); the CB pointers always advance by a full max block.
    // A block size that changed between experts would overshoot fifo_limit, which
    // only wraps on exact equality — see the note in adaptive_chunk.hpp.
    constexpr uint32_t g_in0_block_tiles_max = per_core_M_max * in0_block_w_gu;
    constexpr uint32_t act_tiles_max = per_core_M_max * in0_block_w_d;
    // cb_in0_down_full stays sized to the compile-time MAX per-core M: the down
    // matmul keeps a full ring and MAC-skips rows >= the runtime per_core_M, so
    // the reader pushes the full block (filling only per_core_M runtime rows via
    // the activated mcast; the rest are MAC-skipped downstream).
    constexpr uint32_t d_in0_block_num_tiles = per_core_M_max * in0_block_w_d;
    constexpr uint32_t d_in1_block_num_tiles = per_core_N_d * in0_block_w_d;
    constexpr uint32_t num_blocks_gu = K_gate_tiles / in0_block_w_gu;
    constexpr uint32_t num_blocks_d = K_down_tiles_padded / in0_block_w_d;

    // DOWN_SPLIT: leading tile columns of EVERY down K-row this RISC reads on NoC 0; the writer
    // reads the rest of each row on NoC 1. ~60/40 rather than half/half because NoC 1's
    // responses share the DRAM columns' links (see the program factory's NoC balance note).
    // Equals per_core_N_d when the split is off.
    constexpr uint32_t d_head_tiles = get_compile_time_arg_val(30);
    constexpr bool split_down = d_head_tiles < per_core_N_d;
    // UP_SPLIT: leading `up` K-rows of each gate/up block this RISC reads on NoC 0, beside the
    // whole gate block; the writer reads the rest on NoC 1. Same NoC balance reason. The host
    // sets it only when `up` has gate's exact accessor layout and tile size, so those rows are
    // read through a gate-layout accessor at up's base address (one read loop, one code copy).
#ifdef RE_UP_HEAD_ROWS
    constexpr uint32_t up_head_rows = RE_UP_HEAD_ROWS;
#else
    constexpr uint32_t up_head_rows = 0;
#endif
    // IN1_WRITER_MCAST: 1 => the WRITER runs the gate/up multicast on its own NoC 1
    // and this RISC only reads gate from DRAM, so the read overlaps the multicast.
    constexpr bool writer_mcasts_in1 = get_compile_time_arg_val(31) != 0;
    constexpr uint32_t min_active_tokens = get_compile_time_arg_val(32);
    constexpr uint32_t max_active_tokens = get_compile_time_arg_val(33);
    // Tile columns per DRAM ND shard of the weight tensors, 0 when they are DRAM-interleaved.
    // A core's per_core_N-wide slice of one K-row is exactly one shard, so it reads as a single
    // NoC transaction instead of per_core_N of them; see weight_runs.hpp. gate and up share one
    // value (the program factory pins them equal), down carries its own.
    constexpr uint32_t GU_SHARD_W = get_compile_time_arg_val(34);
    constexpr uint32_t D_SHARD_W = get_compile_time_arg_val(35);
    using GuRuns = unified_routed_expert_ffn::WeightRuns<GU_SHARD_W>;
    using DRuns = unified_routed_expert_ffn::WeightRuns<D_SHARD_W>;
    // Row-shared tilize (row-major x only): the owner core's compute tilizes a K-block of x into
    // this bf8 CB, and this RISC multicasts it into every row core's cb_in0_x. See the gate/up loop.
    constexpr uint32_t cb_x_tl = get_compile_time_arg_val(36);

    constexpr uint32_t x_accessor_offset = 37;
    constexpr auto x_args = TensorAccessorArgs<x_accessor_offset>();
    const auto x_acc = TensorAccessor(x_args, x_addr, get_tile_size(cb_in0_x));
    // Row-major x accessor (x_is_row_major): x is a ROW_MAJOR bf16 buffer whose
    // page is one token-row stick = emb elements = K_gate_tiles*TILE_HEIGHT bf16
    // (X_RM_ELEM_BYTES each). Partial-stick reads index page_id=token-row,
    // offset_bytes=column window. Only used in the row-major path; the tile-page
    // x_acc above serves the TILE path.
    constexpr uint32_t x_rm_stick_bytes = K_gate_tiles * TILE_HEIGHT * X_RM_ELEM_BYTES;
    const auto x_acc_rm = TensorAccessor(x_args, x_addr, x_rm_stick_bytes);

    // gate/up/down accessor LAYOUT descriptors (compile-time). All experts share
    // one layout; the per-expert accessor is built inside the expert loop from
    // the descriptor + that expert's base address (runtime args).
    constexpr uint32_t gate_accessor_offset = x_args.next_compile_time_args_offset();
    constexpr auto gate_args = TensorAccessorArgs<gate_accessor_offset>();

    constexpr uint32_t up_accessor_offset = gate_args.next_compile_time_args_offset();
    constexpr auto up_args = TensorAccessorArgs<up_accessor_offset>();

    constexpr uint32_t down_accessor_offset = up_args.next_compile_time_args_offset();
    constexpr auto down_args = TensorAccessorArgs<down_accessor_offset>();

    constexpr uint32_t counts_accessor_offset = down_args.next_compile_time_args_offset();
    constexpr auto counts_args = TensorAccessorArgs<counts_accessor_offset>();
    const auto counts_acc = TensorAccessor(counts_args, counts_addr);

    constexpr uint32_t idx_accessor_offset = counts_args.next_compile_time_args_offset();
    constexpr auto idx_args = TensorAccessorArgs<idx_accessor_offset>();
    const auto idx_acc = TensorAccessor(idx_args, idx_table_addr);

    // `start` (= expert_region_offsets) accessor. Appended last in the reader's
    // accessor stream. Its buffer binding is shared across all worker cores.
    const uint32_t start_addr = get_common_arg_val<uint32_t>(3);
    constexpr uint32_t start_accessor_offset = idx_args.next_compile_time_args_offset();
    constexpr auto start_args = TensorAccessorArgs<start_accessor_offset>();
    const auto start_acc = TensorAccessor(start_args, start_addr);

    // Per-expert weight base addresses follow start_addr in three contiguous
    // common-runtime-arg blocks of experts_per_chip each: gate[0..N), up[0..N),
    // down[0..N). gate_addr(e) = arg[COMMON_WEIGHTS + e], etc.
    constexpr uint32_t COMMON_WEIGHTS = 4;

#ifdef FUSE_BIAS
    // gpt-oss expert biases. CT bias CB ids + accessor descriptors follow the
    // start accessor; per-expert bias base addresses follow the weight blocks in
    // three further runtime-arg blocks (gate_bias[0..N), up_bias, down_bias)
    // starting at COMMON_BIAS. Read per expert in the loop, added by the compute
    // kernel (gate/up before the activation, down after the down matmul).
    constexpr uint32_t COMMON_BIAS = COMMON_WEIGHTS + 3 * experts_per_chip;
    constexpr uint32_t bias_cb_offset = start_args.next_compile_time_args_offset();
    constexpr uint32_t cb_gate_bias = get_compile_time_arg_val(bias_cb_offset + 0);
    constexpr uint32_t cb_up_bias = get_compile_time_arg_val(bias_cb_offset + 1);
    constexpr uint32_t cb_down_bias = get_compile_time_arg_val(bias_cb_offset + 2);
    constexpr uint32_t gate_bias_accessor_offset = bias_cb_offset + 3;
    constexpr auto gate_bias_args = TensorAccessorArgs<gate_bias_accessor_offset>();
    constexpr uint32_t up_bias_accessor_offset = gate_bias_args.next_compile_time_args_offset();
    constexpr auto up_bias_args = TensorAccessorArgs<up_bias_accessor_offset>();
    constexpr uint32_t down_bias_accessor_offset = up_bias_args.next_compile_time_args_offset();
    constexpr auto down_bias_args = TensorAccessorArgs<down_bias_accessor_offset>();
    CircularBuffer cb_gate_bias_obj(cb_gate_bias);
    CircularBuffer cb_up_bias_obj(cb_up_bias);
    CircularBuffer cb_down_bias_obj(cb_down_bias);
#endif

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
    CircularBuffer cb_x_tl_obj(cb_x_tl);
    CircularBuffer cb_activated_obj(cb_activated);

    // x CB the matmul consumes, filled by this RISC on both layouts. TILE: the row's fixed x
    // sender reads bf8 tiles from DRAM and multicasts them. Row-major: the block's owner core
    // tilizes it once (cb_x_rm -> cb_x_tl on its compute) and multicasts the bf8 tiles, so the
    // other ten cores of the row never tilize it.
    CircularBuffer x_stage_obj(cb_in0_x);

    // D2.0 Semaphore wrappers.
    Semaphore<> in0_ready_sem(in0_ready_sem_id);
    Semaphore<> in0_valid_sem(in0_valid_sem_id);
    Semaphore<> act_ready_sem(act_ready_sem_id);
    Semaphore<> act_valid_sem(act_valid_sem_id);

    // Look up active token count for this expert from device-side buffers.
    // Reserve+read+push so the compute kernel (TRISC) and writer kernel
    // (BRISC) can cb_wait_front on these CBs and read the same L1 data.
    //
    // Each scratch CB is a single page sized (host-side) to hold up to
    // MAX_GLOBAL_EXPERTS UINT32 entries, so `1` here is the whole buffer and a
    // single async_read lands the entire counts / idx vector. The
    // later counts_ptr[global_expert_id] / idx_ptr[local_expert_id] indexing
    // therefore stays in-bounds for any model up to MAX_GLOBAL_EXPERTS experts
    // (DeepSeek V3 256, Kimi 384, ... up to 1024).
#ifdef RE_PACE_CB
    // Cross-group pacing (see below plan_launch): bytes [0, 32) of this CB are the progress board on
    // the counts reader (logical (0,0)), bytes [32, 64) this core's snapshot of it. Zeroed before the
    // counts handshake: no other core touches the board before it has the counts multicast.
    const uint32_t pace_l1 = CircularBuffer(RE_PACE_CB).get_write_ptr();
    for (uint32_t i = 0; i < 16; ++i) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pace_l1)[i] = 0;
    }
#endif
    cb_counts_scratch_obj.reserve_back(1);
    cb_idx_scratch_obj.reserve_back(1);
    const uint32_t counts_l1 = cb_counts_scratch_obj.get_write_ptr();
    const uint32_t idx_l1 = cb_idx_scratch_obj.get_write_ptr();
    const uint32_t counts_page_size = counts_acc.get_aligned_page_size();
    const uint32_t idx_page_size = idx_acc.get_aligned_page_size();
    {
        // COUNTS_BCAST: one core reads the two pages and multicasts them to the grid.
        // Every core needs counts/idx to derive its chunking, but all of them hitting the
        // same two DRAM pages at once serialises on a single bank. One read plus one L1
        // multicast pays the DRAM latency once.
        const bool is_counts_reader = get_arg_val<uint32_t>(COUNTS_BCAST_RT) != 0;
        const uint32_t cb_nx_start = get_arg_val<uint32_t>(COUNTS_BCAST_RT + 1);
        const uint32_t cb_ny_start = get_arg_val<uint32_t>(COUNTS_BCAST_RT + 2);
        const uint32_t cb_nx_end = get_arg_val<uint32_t>(COUNTS_BCAST_RT + 3);
        const uint32_t cb_ny_end = get_arg_val<uint32_t>(COUNTS_BCAST_RT + 4);
        Semaphore<> counts_valid_sem(get_arg_val<uint32_t>(COUNTS_BCAST_RT + 5));
        const uint32_t counts_num_receivers = get_arg_val<uint32_t>(COUNTS_BCAST_RT + 6);
        if (is_counts_reader) {
            noc_read.async_read(counts_acc, CoreLocalMem<uint32_t>(counts_l1), counts_page_size, {.page_id = 0}, {});
            noc_read.async_read(idx_acc, CoreLocalMem<uint32_t>(idx_l1), idx_page_size, {.page_id = 0}, {});
            noc_read.async_read_barrier();
            if (counts_num_receivers > 0) {
                // linked=true so the valid-sem multicast is ordered behind both data
                // multicasts on the same reserved path (as for the weight mcast).
                noc.async_write_multicast(
                    CoreLocalMem<uint32_t>(counts_l1),
                    MulticastEndpoint{},
                    counts_page_size,
                    counts_num_receivers,
                    {.offset_bytes = 0},
                    {.noc_x_start = cb_nx_start,
                     .noc_y_start = cb_ny_start,
                     .noc_x_end = cb_nx_end,
                     .noc_y_end = cb_ny_end,
                     .addr = counts_l1},
                    /*linked=*/true);
                noc.async_write_multicast(
                    CoreLocalMem<uint32_t>(idx_l1),
                    MulticastEndpoint{},
                    idx_page_size,
                    counts_num_receivers,
                    {.offset_bytes = 0},
                    {.noc_x_start = cb_nx_start,
                     .noc_y_start = cb_ny_start,
                     .noc_x_end = cb_nx_end,
                     .noc_y_end = cb_ny_end,
                     .addr = idx_l1},
                    /*linked=*/true);
                noc.async_writes_flushed();
                counts_valid_sem.set(1);
                counts_valid_sem.set_multicast<NocOptions::DEFAULT>(
                    noc, cb_nx_start, cb_ny_start, cb_nx_end, cb_ny_end, counts_num_receivers);
            }
        } else {
            counts_valid_sem.wait(1);
            // Reset our own copy rather than trusting the runtime to re-init semaphores on
            // every launch: each core waits on its OWN L1 copy, so a cached program must
            // not see a stale 1 and skip the wait — that would read the previous
            // invocation's counts, silently correct only while consecutive calls agree.
            counts_valid_sem.set(0);
        }
    }
    cb_counts_scratch_obj.push_back(1);
    cb_idx_scratch_obj.push_back(1);

    const volatile tt_l1_ptr uint32_t* counts_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_l1);
    const volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);

    // Read the `start` (= expert_region_offsets) page ONCE into resident L1. x
    // is the shared dispatched buffer; each expert's rows begin at
    // start[global_id]. Indexed per expert in the loop below (must agree with
    // the writer's row_offset_tiles so x is read and y is written to the same
    // region).
    cb_start_scratch_obj.reserve_back(1);
    const uint32_t start_l1 = cb_start_scratch_obj.get_write_ptr();
    noc_read.async_read(
        start_acc, CoreLocalMem<uint32_t>(start_l1), start_acc.get_aligned_page_size(), {.page_id = 0}, {});
    noc_read.async_read_barrier();
    cb_start_scratch_obj.push_back(1);
    const volatile tt_l1_ptr uint32_t* start_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(start_l1);

    // -------- expert-parallel row groups (adaptive_chunk.hpp) --------
    // Local expert e's tile-row count, band-filtered and clamped exactly as the compute and
    // writer kernels do. The clamp's loop cap is sized for the SMALLEST group (the host sizes
    // num_chunks_max for it), so the count never depends on the group size picked below.
    constexpr uint32_t clamp_chunk = per_core_M_max * adaptive_chunk::kMinGroupRows;
    auto tiles_of = [&](uint32_t e) -> uint32_t {
        const uint32_t count =
            adaptive_chunk::count_in_band(counts_ptr[idx_ptr[e]], min_active_tokens, max_active_tokens);
        return adaptive_chunk::clamp_count_tiles(
            (count + TILE_HEIGHT - 1) / TILE_HEIGHT, clamp_chunk, num_chunks_max, M_tiles_full);
    };
    // Launch plan (adaptive_chunk::plan_launch): phase A runs full waves of experts on one uniform
    // layout, phase B the rest on a larger one; or phase B alone is the single-layout schedule
    // (uniform or mixed-height). The layout, this core's group and row within it, and the
    // weight-multicast topology are per phase.
    uint8_t expert_group[2][experts_per_chip];
    const adaptive_chunk::LaunchPlan plan = adaptive_chunk::plan_launch(tiles_of, per_core_M_max, expert_group);

    // -------- cross-group pacing --------
    // The row groups share DRAM but not fairly: the topmost groups that stream get far less of it than
    // the bottom ones for identical work (dsv3 phase A: one expert each, top group done at ~373 us,
    // bottom at ~203), and the chip ends with the top group. Static placements only move the loss to
    // the next group up, so the groups that are ahead yield instead. Every block a group's column-0
    // weight sender starts adds that block's weight tiles to the group's counter on core (0,0)'s board
    // (one NoC atomic; kPaceDone once the group has finished the phase), and every weight sender starts
    // its next block (its own reads and the writer's up_go / down_go) only while its group is at most
    // one gate/up block of weight tiles ahead of every unfinished group. Each sender refreshes its
    // snapshot of the board with one 32-byte read per block. Only the first phase is paced (phase B
    // starts as phase A ends, which pacing lines up), and only a layout of 2+ groups. Progress is
    // absolute, so groups with more work than others finish last, as without pacing. No data, slot or
    // order changes.
    constexpr uint32_t kPaceUgu = 2 * g_in1_block_num_tiles;
    constexpr uint32_t kPaceUd = d_in1_block_num_tiles;
#ifdef RE_PACE_CB
    const uint32_t pace_phase = (plan.layout[0] != adaptive_chunk::kNoLayout) ? 0u : 1u;
    Pacer pacer;
    pacer.board =
        get_noc_addr(get_arg_val<uint32_t>(COUNTS_BCAST_RT + 1), get_arg_val<uint32_t>(COUNTS_BCAST_RT + 2), pace_l1);
    pacer.snap_l1 = pace_l1 + 32;
#endif
    // Phase A's own weight-multicast semaphore pair follows the per-layout topologies.
    constexpr uint32_t PHASE_A_SEM_RT =
        GROUP_TOPO_RT + adaptive_chunk::num_layouts(adaptive_chunk::kGridY) * GROUP_TOPO_ARGS;

    const uint32_t x_tile_bytes = get_tile_size(cb_in0_x);
    const uint32_t gate_tile_bytes = get_tile_size(cb_in1_gate);
    const uint32_t up_tile_bytes = get_tile_size(cb_in1_up);
    const uint32_t down_tile_bytes = get_tile_size(cb_in1_down);

    // Weight-multicast handshake constants (see the K-block loops below).
    constexpr uint32_t IN1_VALID = 1;
    constexpr uint32_t IN0_VALID = 1;

    // UP_SPLIT handshake counter, kept in lockstep with the writer's ACROSS all
    // experts: both kernels loop experts in the same order with the same
    // per-expert effective_chunks, so the per-K-block increments stay matched.
    uint32_t up_seq = 0;
    // Separate counter and sems so the gate/up slot indexing stays a pure gate/up count
    // (DOWN_SEM_RT is defined with the other runtime-arg offsets above).
    uint32_t down_seq = 0;
    Semaphore<> down_go_sem(get_arg_val<uint32_t>(DOWN_SEM_RT));
    Semaphore<> down_done_sem(get_arg_val<uint32_t>(DOWN_SEM_RT + 1));
    // IN1_WRITER_MCAST handshake: gate/up blocks only, so it needs its own counter
    // for the same cross-chunk reason down_seq does.
    Semaphore<> mcast_go_sem(get_arg_val<uint32_t>(MCAST_SEM_RT));
    Semaphore<> mcast_done_sem(get_arg_val<uint32_t>(MCAST_SEM_RT + 1));
    uint32_t mc_seq = 0;

    // Row-shared tilize (row-major x). Gate/up K-blocks are numbered across every chunk and
    // expert this row works on (x_blk = number of earlier blocks); block x_blk + kb belongs to the
    // row column adaptive_chunk::x_owner_col(x_blk + kb, ...). The compute kernel keeps the same
    // counter, and all cores of a row run the same experts and chunks in the same group row, so
    // every core of the row agrees on each block's owner.
    uint32_t x_blk = 0;
    // The whole M-row: the row-shared x multicast and the phase-4 activated multicast both span it.
    const uint32_t mrow_first_nx = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 0);
    const uint32_t mrow_first_ny = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 1);
    const uint32_t mrow_last_nx = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * (GRID_X_NOC - 1) + 0);
    const uint32_t mrow_last_ny = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * (GRID_X_NOC - 1) + 1);
    // Row multicast (x in gate/up, activated in down): `bytes` from src into dst on every core of this
    // M-row, this one included, then its valid sem (value 1) on the same linked path, so the sem cannot
    // overtake the data at any receiver. A zero-byte block sends the sem alone. Out of line and captured
    // by value: both row streams share one copy and pin no kernel_main locals.
    auto mcast_row = [mrow_first_nx, mrow_first_ny, mrow_last_nx, mrow_last_ny](
                         uint32_t src, uint32_t dst, uint32_t bytes, Semaphore<> valid_sem) __attribute__((noinline)) {
        Noc noc_mc;
        if (bytes > 0) {
            noc_mc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                CoreLocalMem<uint32_t>(src),
                MulticastEndpoint{},
                bytes,
                GRID_X_NOC,
                {.offset_bytes = 0},
                {.noc_x_start = mrow_first_nx,
                 .noc_y_start = mrow_first_ny,
                 .noc_x_end = mrow_last_nx,
                 .noc_y_end = mrow_last_ny,
                 .addr = dst},
                /*linked=*/true);
        }
        noc_mc.async_writes_flushed();
        valid_sem.set(1);
        valid_sem.set_multicast<NocOptions::MCAST_INCL_SRC>(
            noc_mc, mrow_first_nx, mrow_first_ny, mrow_last_nx, mrow_last_ny, GRID_X_NOC);
    };

    // ======================= per-phase, per-local-expert loop =======================
    // Per expert we resolve its global id (idx_table[e]), token count
    // (counts[global_id]), region offset (start[global_id]) and per-expert
    // weight base addresses, then drive the same chunked matmul pipeline.
    for (uint32_t phase = 0; phase < 2; ++phase) {
        const uint32_t layout = plan.layout[phase];
        if (layout == adaptive_chunk::kNoLayout) {
            continue;
        }
        uint32_t group_base = 0;
        uint32_t group_rows = 0;
        const uint32_t my_group = adaptive_chunk::layout_group(
            adaptive_chunk::layout_at(adaptive_chunk::kGridY, layout), my_gy, group_base, group_rows);
        const uint32_t my_mt = my_gy - group_base;  // M-row within the group
        const uint32_t chunk_M = per_core_M_max * group_rows;

        // This group's weight (in1) multicast topology: sender row, rectangle over the group's rows.
        const uint32_t topo = GROUP_TOPO_RT + GROUP_TOPO_ARGS * layout;
        const bool is_in1_sender = get_arg_val<uint32_t>(topo + 0) != 0;
        const uint32_t in1_num_receivers = get_arg_val<uint32_t>(topo + 1);
        const uint32_t in1_mcast_nx_start = get_arg_val<uint32_t>(topo + 2);
        const uint32_t in1_mcast_ny_start = get_arg_val<uint32_t>(topo + 3);
        const uint32_t in1_mcast_nx_end = get_arg_val<uint32_t>(topo + 4);
        const uint32_t in1_mcast_ny_end = get_arg_val<uint32_t>(topo + 5);
        const uint32_t in1_sender_nx = get_arg_val<uint32_t>(topo + 6);
        const uint32_t in1_sender_ny = get_arg_val<uint32_t>(topo + 7);
        // Phase A has its own pair: a core already in phase B acks ITS phase-B sender, which may still
        // be sending (or receiving) for phase A, and must not be counted there.
        Semaphore<> in1_ready_sem((phase == 0) ? get_arg_val<uint32_t>(PHASE_A_SEM_RT + 0) : in1_ready_sem_id);
        Semaphore<> in1_valid_sem((phase == 0) ? get_arg_val<uint32_t>(PHASE_A_SEM_RT + 1) : in1_valid_sem_id);
        // Senders sit on their group's top or bottom row (adaptive_chunk::sender_row_in_group). A top sender's
        // down multicast leaves this RISC on NoC 0 (+y) from the rectangle's start corner. A bottom
        // sender's would wrap around the whole column of the torus on NoC 0, so its WRITER multicasts it
        // on NoC 1 (-y) instead, once this RISC's head of the block has landed (mcast_go; same go/done
        // pair and sequence as the gate/up multicast, which the writer walks in the same block order).
        // The writer must already read the down tail (DOWN_SPLIT) to own the slot's multicast.
        const bool wmc_down = writer_mcasts_in1 && split_down && is_in1_sender && my_mt + 1 == group_rows;
        // Gate/up, the mirror image: the writer's NoC-1 multicast starts at the rectangle's bottom corner, so
        // only a bottom sender keeps it. A top sender's would wrap around the column, so THIS RISC multicasts
        // its gate/up blocks on NoC 0 instead, and its writer only reads. The writer computes the same
        // predicate; mc_seq / mcast_go / mcast_done count writer multicasts only.
        const bool wmc_gu = writer_mcasts_in1 && my_mt + 1 == group_rows;
        // One linked data multicast into the same L1 address on every column receiver of this group.
        // Captured by value: a by-reference capture would pin these kernel_main locals to the stack. The
        // Noc is this RISC's default one, built here so its id stays a compile-time constant.
        auto mcast_in1 =
            [in1_num_receivers, in1_mcast_nx_start, in1_mcast_ny_start, in1_mcast_nx_end, in1_mcast_ny_end](
                uint32_t l1, uint32_t bytes) __attribute__((noinline)) {
                Noc noc_mc;
                noc_mc.async_write_multicast(
                    CoreLocalMem<uint32_t>(l1),
                    MulticastEndpoint{},
                    bytes,
                    in1_num_receivers,
                    {.offset_bytes = 0},
                    {.noc_x_start = in1_mcast_nx_start,
                     .noc_y_start = in1_mcast_ny_start,
                     .noc_x_end = in1_mcast_nx_end,
                     .noc_y_end = in1_mcast_ny_end,
                     .addr = l1},
                    /*linked=*/true);
            };
        // Close a column multicast: once the data left this core, the valid sem follows it on the same
        // reserved path (linked above), so it cannot overtake the data at any receiver.
        auto mcast_in1_valid = [in1_valid_sem,
                                in1_num_receivers,
                                in1_mcast_nx_start,
                                in1_mcast_ny_start,
                                in1_mcast_nx_end,
                                in1_mcast_ny_end]() mutable __attribute__((noinline)) {
            Noc noc_mc;
            noc_mc.async_writes_flushed();
            in1_valid_sem.set(IN1_VALID);
            in1_valid_sem.set_multicast<NocOptions::DEFAULT>(
                noc_mc, in1_mcast_nx_start, in1_mcast_ny_start, in1_mcast_nx_end, in1_mcast_ny_end, in1_num_receivers);
        };
        // Pacing: this phase is paced only if it is the first one and has several groups; the gate at the
        // start of each weight block is pace_start (Pacer::start).
#ifdef RE_PACE_CB
        const uint32_t layout_groups = adaptive_chunk::layout_at(adaptive_chunk::kGridY, layout).groups;
        pacer.groups = (phase == pace_phase && layout_groups > 1) ? layout_groups : 0u;
        pacer.me = my_group;
        pacer.pub = pacer.groups != 0 && is_in1_sender && my_nt_gu == 0;
        auto pace_start = [&pacer](uint32_t units, bool blocking) { return pacer.start<kPaceUgu>(units, blocking); };
#else
        auto pace_start = [](uint32_t, bool) { return true; };
#endif
        // The row-shared x owners skip this row's sender columns (adaptive_chunk::x_owner_key).
        [[maybe_unused]] const uint32_t x_key = adaptive_chunk::x_owner_key(group_base, group_rows, my_mt);

        for (uint32_t local_expert_id = 0; local_expert_id < experts_per_chip; ++local_expert_id) {
            // Per-expert weight accessors, built from the shared layout descriptors
            // and this expert's base addresses (runtime-arg arrays after start).
            const uint32_t gate_addr_e =
                get_common_arg_val<uint32_t>(COMMON_WEIGHTS + 0 * experts_per_chip + local_expert_id);
            const uint32_t up_addr_e =
                get_common_arg_val<uint32_t>(COMMON_WEIGHTS + 1 * experts_per_chip + local_expert_id);
            const uint32_t down_addr_e =
                get_common_arg_val<uint32_t>(COMMON_WEIGHTS + 2 * experts_per_chip + local_expert_id);
            const auto gate_acc = TensorAccessor(gate_args, gate_addr_e, gate_tile_bytes);
            const auto up_acc = TensorAccessor(up_args, up_addr_e, up_tile_bytes);
            // `up` through gate's layout descriptor (identical by the host's RE_UP_HEAD_ROWS check).
            const auto up_acc_gl = TensorAccessor(gate_args, up_addr_e, gate_tile_bytes);
            const auto down_acc = TensorAccessor(down_args, down_addr_e, down_tile_bytes);

            const uint32_t global_expert_id = idx_ptr[local_expert_id];
            // Hybrid dispatch: experts outside this op's band belong to the other routed-expert
            // op and are dropped here exactly like a zero count. So is an expert another row group
            // owns (every kernel runs the same plan over the same counts, so all agree).
            const bool mine = expert_group[phase][local_expert_id] == my_group;
            const uint32_t count_value =
                mine ? adaptive_chunk::count_in_band(counts_ptr[global_expert_id], min_active_tokens, max_active_tokens)
                     : 0u;
            // counts[] is device-produced and unvalidated: bound it by the capacity
            // this program was built for (num_chunks_max * chunk_M_max tile-rows)
            // BEFORE deriving anything from it. The clamp is arithmetic so it also
            // holds in Release, where ASSERT is a no-op; the assert below still
            // hard-fails a mismatch in watcher builds. Compute and writer clamp
            // identically, so all three keep the same row mapping (see
            // adaptive_chunk::clamp_count_tiles).
            const uint32_t count_tiles_raw = (count_value + TILE_HEIGHT - 1) / TILE_HEIGHT;
            const uint32_t count_tiles =
                adaptive_chunk::clamp_count_tiles(count_tiles_raw, clamp_chunk, num_chunks_max, M_tiles_full);
            ASSERT(count_tiles == count_tiles_raw);
            // Runtime chunk layout from THIS expert's actual token count (identical
            // math in the compute and writer kernels, so all three agree on the row
            // mapping). Full chunks span chunk_M_max tile-rows; the tail chunk shrinks
            // per_core_M to the remainder, so per_core_M is per-chunk (see the chunk
            // loop) and adapts to each expert's own load with minimal phantom work.
            const uint32_t effective_chunks = adaptive_chunk::num_chunks(count_tiles, chunk_M);

            // x-read row offset: this expert's rows begin at start[global_id].
            // Convert the token row to a tile-page offset (row_tile * K_gate_tiles)
            // for the TILE path; the row-major path offsets the token-row stick.
            const uint32_t start_value = start_ptr[global_expert_id];
            const uint32_t x_start_tile_idx = (start_value / TILE_HEIGHT) * K_gate_tiles;
            const uint32_t x_start_stick = start_value;

            // Pre-zero bound: rows past min(count_tiles, M_tiles_full) are invalid
            // and must feed zeros to the matmul (see the per-chunk pre-zero below).
            const uint32_t M_bound = (count_tiles < M_tiles_full) ? count_tiles : M_tiles_full;

#ifdef FUSE_BIAS
            // Only experts this group runs: the compute kernel pops biases for those alone.
            if (effective_chunks > 0) {
                // This expert's N-column slice of its (1, N) biases. The compute
                // kernel wait_fronts then pops at the end of this expert so the next
                // expert can reuse the single-buffered bias CBs. Bias is a single
                // tile-row tensor, so the DRAM page index == tile column. Phantom
                // columns (col >= actual N) are zero-filled: unlike the per-K-block
                // weight reads this runs once per expert, so the fill is cheap, and it
                // keeps the bias add from turning a phantom column into Inf/NaN. Those
                // columns are discarded either way — the gate/up ones by the down
                // phase's K bound, the down ones by the writer.
                {
                    const uint32_t gbias_addr =
                        get_common_arg_val<uint32_t>(COMMON_BIAS + 0 * experts_per_chip + local_expert_id);
                    const uint32_t ubias_addr =
                        get_common_arg_val<uint32_t>(COMMON_BIAS + 1 * experts_per_chip + local_expert_id);
                    const uint32_t dbias_addr =
                        get_common_arg_val<uint32_t>(COMMON_BIAS + 2 * experts_per_chip + local_expert_id);
                    const auto gate_bias_acc = TensorAccessor(gate_bias_args, gbias_addr, get_tile_size(cb_gate_bias));
                    const auto up_bias_acc = TensorAccessor(up_bias_args, ubias_addr, get_tile_size(cb_up_bias));
                    const auto down_bias_acc = TensorAccessor(down_bias_args, dbias_addr, get_tile_size(cb_down_bias));
                    const uint32_t gbias_tb = get_tile_size(cb_gate_bias);
                    cb_gate_bias_obj.reserve_back(per_core_N_gu);
                    cb_up_bias_obj.reserve_back(per_core_N_gu);
                    uint32_t lg = cb_gate_bias_obj.get_write_ptr();
                    uint32_t lu = cb_up_bias_obj.get_write_ptr();
                    for (uint32_t n = 0; n < per_core_N_gu; ++n) {
                        const uint32_t col = my_nt_gu * per_core_N_gu + n;
                        if (col < N_gate_tiles_full) {
                            noc_read.async_read(
                                gate_bias_acc, CoreLocalMem<uint32_t>(lg), gbias_tb, {.page_id = col}, {});
                            noc_read.async_read(
                                up_bias_acc, CoreLocalMem<uint32_t>(lu), gbias_tb, {.page_id = col}, {});
                        } else {
                            volatile tt_l1_ptr uint64_t* pg = reinterpret_cast<volatile tt_l1_ptr uint64_t*>(lg);
                            volatile tt_l1_ptr uint64_t* pu = reinterpret_cast<volatile tt_l1_ptr uint64_t*>(lu);
                            for (uint32_t i = 0; i < gbias_tb / 8; ++i) {
                                pg[i] = 0;
                                pu[i] = 0;
                            }
                        }
                        lg += gbias_tb;
                        lu += gbias_tb;
                    }
                    const uint32_t dbias_tb = get_tile_size(cb_down_bias);
                    cb_down_bias_obj.reserve_back(per_core_N_d);
                    uint32_t ld = cb_down_bias_obj.get_write_ptr();
                    for (uint32_t n = 0; n < per_core_N_d; ++n) {
                        const uint32_t col = my_nt_d * per_core_N_d + n;
                        if (col < N_down_tiles_full) {
                            noc_read.async_read(
                                down_bias_acc, CoreLocalMem<uint32_t>(ld), dbias_tb, {.page_id = col}, {});
                        } else {
                            volatile tt_l1_ptr uint64_t* pd = reinterpret_cast<volatile tt_l1_ptr uint64_t*>(ld);
                            for (uint32_t i = 0; i < dbias_tb / 8; ++i) {
                                pd[i] = 0;
                            }
                        }
                        ld += dbias_tb;
                    }
                    noc_read.async_read_barrier();
                    cb_gate_bias_obj.push_back(per_core_N_gu);
                    cb_up_bias_obj.push_back(per_core_N_gu);
                    cb_down_bias_obj.push_back(per_core_N_d);
                }
            }
#endif

    // Bound the chunk loop by effective_chunks (= ceil_div(count, chunk_M_tiles))
    // so this expert only does work proportional to its actual token count,
    // not the max-tokens-padded shape of the input. chunk_M_tiles / per_core_M
    // were picked from the count above; the row mapping is contiguous (core gy
    // owns rows [chunk*chunk_M + gy*per_core_M, + per_core_M)).
    for (uint32_t chunk = 0; chunk < effective_chunks; ++chunk) {
        // Per-chunk per_core_M: per_core_M_max for full chunks, a smaller divisor
        // for the tail. Chunk starts are UNIFORM at chunk*chunk_M_max (full chunks
        // are max_chunk; the tail is last, so its start is num_full*max_chunk too).
        const uint32_t per_core_M = adaptive_chunk::per_core_M_for_chunk(chunk, count_tiles, chunk_M, group_rows);
        const uint32_t this_core_first_row = chunk * chunk_M + my_mt * per_core_M;

        // Valid tile-rows of this core's M-slice: a contiguous prefix of per_core_M. Rows past the
        // token count hold stale L1 and are dropped downstream (no FFN stage reduces across M).
        const uint32_t valid_tile_rows =
            (this_core_first_row < count_tiles)
                ? ((count_tiles - this_core_first_row < per_core_M) ? (count_tiles - this_core_first_row) : per_core_M)
                : 0;

        // Row-major x: read K-block kb_rm's column window (in0_block_w_gu tiles wide) of this
        // core's tile-rows into cb_x_rm, each tile-row laid out as one TILE_HEIGHT x
        // (in0_block_w_gu*32) strip for the tilize. Issued only; the caller barriers and pushes.
        // Only sticks < count_value are real tokens; the padding sticks of the last tile-row and
        // whole rows >= count_tiles are skipped and stay stale (free-dim rows, dropped downstream).
        [[maybe_unused]] auto issue_x_rm_read = [&](uint32_t kb_rm) {
            constexpr uint32_t rm_kblock_bytes = in0_block_w_gu * TILE_HEIGHT * X_RM_ELEM_BYTES;
            cb_x_rm_obj.reserve_back(g_in0_block_tiles_max);
            uint32_t l1_x = cb_x_rm_obj.get_write_ptr();
            const uint32_t col_off_bytes = kb_rm * rm_kblock_bytes;
            for (uint32_t m = 0; m < per_core_M; ++m) {
                const uint32_t tile_row = this_core_first_row + m;
                if (tile_row < count_tiles) {
                    const uint32_t row_base = tile_row * TILE_HEIGHT;
                    const uint32_t real_r =
                        (row_base + TILE_HEIGHT <= count_value) ? TILE_HEIGHT : (count_value - row_base);
                    for (uint32_t r = 0; r < real_r; ++r) {
                        noc_read.async_read(
                            x_acc_rm,
                            CoreLocalMem<uint32_t>(l1_x),
                            rm_kblock_bytes,
                            {.page_id = x_start_stick + row_base + r, .offset_bytes = col_off_bytes},
                            {});
                        l1_x += rm_kblock_bytes;
                    }
                    l1_x += (TILE_HEIGHT - real_r) * rm_kblock_bytes;
                } else {
                    l1_x += TILE_HEIGHT * rm_kblock_bytes;
                }
            }
        };
        // Owner column of this chunk's x block kb_x: never this row's weight sender for that column
        // (adaptive_chunk::x_owner_col), so the weight stream's RISC carries no x duty.
        [[maybe_unused]] auto x_owner = [&](uint32_t kb_x) {
            return adaptive_chunk::x_owner_col(x_blk + kb_x, GRID_X_NOC, x_key & 0xFFFF, x_key >> 16);
        };
        [[maybe_unused]] auto owns_x_block = [&](uint32_t kb_x) { return x_owner(kb_x) == my_nt_gu; };

        // Row-shared tilize, row-major x. Every core of an M-row needs the same x K-block. Instead
        // of each of the GRID_X cores tilizing it, block x_blk + kb is read, tilized and multicast
        // by ONE owner core of the row, x_owner(kb) (the row's non-weight-sender columns in turn),
        // so each core tilizes a few of the blocks and skips the per-block tilize/matmul reconfig
        // on the rest:
        //   * this RISC on the owner reads the raw block into cb_x_rm TWO blocks ahead (below), so
        //     it is in L1 when the owner's compute tilizes it one block ahead (cb_x_rm -> cb_x_tl,
        //     before that compute's matmul of the previous block);
        //   * it then multicasts the bf8 tiles from cb_x_tl into the cb_in0_x slot of every row
        //     core, itself included. Every core pushes cb_in0_x one constant block per K-block
        //     from kernel start, so the slot sits at the same L1 address on all of them.
        // Blocks 0 and 1 of the chunk are read up front; the loop reads block kb + 2.
        if constexpr (x_is_row_major != 0) {
            for (uint32_t kb_x = 0; kb_x < 2 && kb_x < num_blocks_gu; ++kb_x) {
                if (owns_x_block(kb_x)) {
                    issue_x_rm_read(kb_x);
                    noc_read.async_read_barrier();
                    cb_x_rm_obj.push_back(g_in0_block_tiles_max);
                }
            }
        }

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
        //
        // In1 sender, gate DRAM read of block kb_rd into this core's slot at l1_slot. Issued only;
        // the sender section below barriers. N-OOB hidden padding columns (col >=
        // N_gate_tiles_full == down's K_down_tiles) are left UNWRITTEN: their gate output lands on
        // a down K position the down matmul never reduces (see the compute's real_k_tiles bound),
        // so the stale L1 is dropped. Mirrors the down-weight K-OOB skip in phase 4.
        const uint32_t gate_col0 = my_nt_gu * per_core_N_gu;
        const uint32_t gate_col_end =
            (gate_col0 + per_core_N_gu < N_gate_tiles_full) ? gate_col0 + per_core_N_gu : N_gate_tiles_full;
        // UP_SPLIT also issues the first up_head_rows K-rows of the `up` block into its slot at
        // l1_up_slot here, on NoC 0 (the writer reads the remaining rows on NoC 1); the same
        // barrier covers them before mcast_go. Rows [0, w) are gate, [w, w + up_head_rows) up.
        constexpr uint32_t gu_rows_read = in0_block_w_gu + (up_split ? up_head_rows : 0);
        auto issue_gate_read = [&](uint32_t kb_rd, uint32_t l1_slot, uint32_t l1_up_slot) {
            uint32_t l1_w = l1_slot;
            for (uint32_t k = 0; k < gu_rows_read; ++k) {
                const bool is_up = k >= in0_block_w_gu;
                if (k == in0_block_w_gu) {
                    l1_w = l1_up_slot;
                }
                GuRuns::read(
                    noc_read,
                    is_up ? up_acc_gl : gate_acc,
                    kb_rd * in0_block_w_gu + (is_up ? k - in0_block_w_gu : k),
                    N_gate_tiles_full,
                    gate_col0,
                    gate_col_end,
                    l1_w,
                    gate_tile_bytes);
                l1_w += per_core_N_gu * gate_tile_bytes;
            }
        };
        // Weight look-ahead (IN1_WRITER_MCAST + UP_SPLIT): once block kb is pushed, the sender
        // starts block kb + 1 -- releases the writer's NoC-1 `up` read and issues its own NoC-0
        // gate read -- as soon as THIS core's ring slot is free (its compute has popped the slot's
        // previous block) and the slot's earlier multicast has drained (the writer's, when it
        // multicasts; this RISC's own is flushed before its valid sem), instead of at the top of
        // iteration kb + 1, i.e. after the row's x block kb has also reached this core.
        // So the column's weight stream no longer waits for the sender row's x stream. Same slots,
        // same per-RISC order of reads and multicasts, same bytes: the data is unchanged.
        constexpr bool gu_lookahead = writer_mcasts_in1 && up_split;
        bool gu_read_issued = false;
        auto try_gu_lookahead = [&](uint32_t kb_next) __attribute__((noinline)) {
            if constexpr (gu_lookahead) {
                if (!gu_read_issued && kb_next < num_blocks_gu &&
                    (!wmc_gu || mc_seq + 1 < 3 || mcast_done_sem.value() >= mc_seq - 1) &&
                    cb_in1_gate_obj.pages_reservable_at_back(g_in1_block_num_tiles) &&
                    cb_in1_up_obj.pages_reservable_at_back(g_in1_block_num_tiles) && pace_start(kPaceUgu, false)) {
                    ++up_seq;
                    up_go_sem.set(up_seq);
                    if (wmc_gu) {
                        ++mc_seq;
                    }
                    issue_gate_read(kb_next, cb_in1_gate_obj.get_write_ptr(), cb_in1_up_obj.get_write_ptr());
                    gu_read_issued = true;
                }
            }
        };
        for (uint32_t kb = 0; kb < num_blocks_gu; ++kb) {
            // This block's x source: the rotating owner (row-major) or the row's fixed x sender (TILE).
            const uint32_t x_owner_kb = (x_is_row_major != 0) ? x_owner(kb) : 0;
            const bool is_x_src = (x_is_row_major != 0) ? (x_owner_kb == my_nt_gu) : is_in0_sender;
            [[maybe_unused]] bool x_rm_pending = false;
            if constexpr (x_is_row_major != 0) {
                if (kb + 2 < num_blocks_gu && owns_x_block(kb + 2)) {
                    issue_x_rm_read(kb + 2);
                    x_rm_pending = true;
                }
            }

            x_stage_obj.reserve_back(g_in0_block_tiles_max);
            cb_in1_gate_obj.reserve_back(g_in1_block_num_tiles);
            if constexpr (reader_mcasts_up) {
                cb_in1_up_obj.reserve_back(g_in1_block_num_tiles);
            }

            // Pacing: hold this block (its reads below and the writer's up_go) while the group is ahead.
            if (is_in1_sender && !gu_read_issued) {
                pace_start(kPaceUgu, true);
            }
            // UP_SPLIT: slot reserved -> release writer to read `up` on NoC 1,
            // concurrent with the reader's NoC-0 `gate` read below (unless the look-ahead in
            // iteration kb - 1 already did both).
            if constexpr (up_split) {
                if (is_in1_sender && !gu_read_issued) {
                    ++up_seq;
                    up_go_sem.set(up_seq);
                }
            }

            // Step 1: receivers ack BOTH senders upfront so both senders can
            // proceed in parallel. The senders are usually disjoint sets of
            // cores; the only core that's both senders is (0,0) which doesn't
            // self-inc (it's its own sender for both).
            if (!is_x_src) {
                in0_valid_sem.set(0);
                if constexpr (x_is_row_major != 0) {
                    const uint32_t owner = x_owner_kb;
                    in0_ready_sem.up(
                        noc,
                        get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * owner + 0),
                        get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * owner + 1),
                        1);
                } else {
                    in0_ready_sem.up(noc, in0_sender_nx, in0_sender_ny, 1);
                }
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
            // (Row-major: the owner's x multicast runs AFTER its weight work, below, because
            // it waits on its own compute's tilize.)
            if (x_is_row_major == 0 && is_x_src) {
                in0_ready_sem.wait(in0_num_receivers);
                in0_ready_sem.set(0);

                uint32_t l1_x = x_stage_obj.get_write_ptr();
                const uint32_t block_start = l1_x;
                // Rows past count_tiles are NOT valid — they hold uninitialized
                // DRAM. Skip the read and leave the stale L1 as-is: no FFN stage
                // reduces across M (tokens), so garbage stays confined to its own
                // row, and the writer drops every row >= count_tiles (its
                // `row < count_tiles` guard), so it never reaches the output.
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
                        // Invalid row (>= count_tiles): skip the read, advance the
                        // write ptr. Stale L1 here is dropped by the writer (see above).
                        l1_x += in0_block_w_gu * x_tile_bytes;
                    }
                }
                noc_read.async_read_barrier();

                // Multicast only the real rows of the block (whole valid tile-rows; a 32-row
                // tile can't be split). A fully-padding M-row sends no data — only the valid
                // sem, so receivers still advance.
                const uint32_t mcast_bytes = valid_tile_rows * in0_block_w_gu * x_tile_bytes;
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
                if (mcast_bytes > 0) {
                    noc.async_write_multicast(
                        CoreLocalMem<uint32_t>(block_start),
                        MulticastEndpoint{},
                        mcast_bytes,
                        in0_num_receivers,
                        {.offset_bytes = 0},
                        {.noc_x_start = in0_mcast_nx_start,
                         .noc_y_start = in0_mcast_ny_start,
                         .noc_x_end = in0_mcast_nx_end,
                         .noc_y_end = in0_mcast_ny_end,
                         .addr = block_start},
                        /*linked=*/true);
                }
                x_stage_obj.push_back(g_in0_block_tiles_max);

                noc.async_writes_flushed();
                in0_valid_sem.set(IN0_VALID);
                in0_valid_sem.set_multicast<NocOptions::DEFAULT>(
                    noc, in0_mcast_nx_start, in0_mcast_ny_start, in0_mcast_nx_end, in0_mcast_ny_end, in0_num_receivers);
            }

            if (is_in1_sender) {
                if (wmc_gu) {
                    // The writer multicasts, so it owns the receivers' ready handshake. This
                    // RISC only needs its own slot back: block mc_seq reuses the slot block
                    // mc_seq-slots used (slots = the host ring depth, 2 or 3), and THAT block's
                    // multicast must have drained. Waiting on mc_seq-2 covers it for any depth
                    // >= 2 (the writer multicasts in order), and waiting on it rather than on
                    // mc_seq is what lets this DRAM read overlap the writer's multicast of the
                    // previous block, which is the whole point.
                    // (Already done, with the gate read, by the look-ahead in iteration kb - 1.)
                    if (!gu_read_issued) {
                        ++mc_seq;
                        if (mc_seq >= 3) {
                            mcast_done_sem.wait_min(mc_seq - 2);
                        }
                    }
                }
                // This RISC multicasts (top sender): its own slot is free, so the read goes out now and
                // only the multicast below waits for the receivers' acks.

                // `up` slot. LEGACY: reader reads it on NoC 0. UP_SPLIT: the writer reads all
                // but its first up_head_rows K-rows on NoC 1 (this RISC issues those with the
                // gate read); the reader waits on up_done (below) before mcasting.
                uint32_t up_block_start = 0;
                if constexpr (reader_mcasts_up) {
                    up_block_start = cb_in1_up_obj.get_write_ptr();
                }
                // DRAM read gate region first (see issue_gate_read).
                const uint32_t gate_block_start = cb_in1_gate_obj.get_write_ptr();
                if (!gu_read_issued) {
                    issue_gate_read(kb, gate_block_start, up_block_start);
                }
                gu_read_issued = false;
                if constexpr (reader_reads_up) {
                    uint32_t l1_w_up = up_block_start;
                    // N-OOB hidden padding columns left unwritten: same rationale as the gate
                    // read above, and `up` shares gate's N split so it shares the bounds.
                    for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
                        GuRuns::read(
                            noc_read,
                            up_acc,
                            kb * in0_block_w_gu + k,
                            N_gate_tiles_full,
                            gate_col0,
                            gate_col_end,
                            l1_w_up,
                            up_tile_bytes);
                        l1_w_up += per_core_N_gu * up_tile_bytes;
                    }
                }
                noc_read.async_read_barrier();

                // IN1_WRITER_MCAST: gate is in L1 -- hand this slot to the writer, which
                // multicasts it (and `up`, which it read itself) on NoC 1. Signalled BEFORE
                // the up_done wait below so the writer starts as early as possible.
                if (wmc_gu) {
                    mcast_go_sem.set(mc_seq);
                }

                // UP_SPLIT: wait for the writer's NoC-1 `up` read before mcast.
                if constexpr (up_split) {
                    up_done_sem.wait_min(up_seq);
                }
                // GRID_Y == 1: no column receivers — skip mcast/valid-sem; the
                // locally-read weights go straight to compute via cb_push_back.
                // A bottom sender's writer runs this multicast on NoC 1 (wmc_gu).
                const bool rmc_gu = !wmc_gu && in1_num_receivers > 0;
                if (rmc_gu) {
                    in1_ready_sem.wait(in1_num_receivers);
                    in1_ready_sem.set(0);
                    // The LAST in1 data multicast before the in1_valid sem must
                    // be linked=true so the (posted) valid-sem multicast travels
                    // the SAME reserved path and lands AFTER the data at every
                    // receiver. Otherwise, under NoC contention (heavy fabric
                    // load), the valid sem can overtake the weight data -> the
                    // receiver pushes cb_in1_{gate,up} and compute reads STALE
                    // weights -> wrong matmul output (rare, timing-dependent;
                    // a flush/barrier does not fix posted multicast writes on
                    // Blackhole). Mirrors the phase-4 activated mcast. Both data
                    // multicasts are linked, so `up` holds the path for the sem.
                    mcast_in1(gate_block_start, g_in1_block_num_tiles * gate_tile_bytes);
                    if constexpr (reader_mcasts_up) {
                        mcast_in1(up_block_start, g_in1_block_num_tiles * up_tile_bytes);
                    }
                }
                cb_in1_gate_obj.push_back(g_in1_block_num_tiles);
                if constexpr (reader_mcasts_up) {
                    cb_in1_up_obj.push_back(g_in1_block_num_tiles);
                }
                if (rmc_gu) {
                    // Close the multicast before any read: a unicast (DRAM read included) issued while
                    // a linked multicast is open on this NoC deadlocks it (watcher's
                    // DebugSanitizeNocLinkedTransactionViolation), so block kb + 1's look-ahead below
                    // waits for this block's data to leave the core.
                    mcast_in1_valid();
                }
                // Block kb is pushed: start kb + 1 now if its slot is already free.
                try_gu_lookahead(kb + 1);
            }

            // Row-major x, owner of this block: multicast the tiles its compute tilized into
            // cb_x_tl to every row core's cb_in0_x slot (this core's included), once the other
            // GRID_X - 1 cores have reserved theirs. Data and valid sem both go INCL_SRC on one
            // linked path, as the phase-4 activated multicast does, and the write barrier is
            // what proves the loopback copy landed before this core pushes its own slot.
            if constexpr (x_is_row_major != 0) {
                if (is_x_src) {
                    cb_x_tl_obj.wait_front(g_in0_block_tiles_max);
                    in0_ready_sem.wait(GRID_X_NOC - 1);
                    in0_ready_sem.set(0);
                    // Whole valid tile-rows only; rows past the token count stay stale downstream.
                    const uint32_t x_mcast_bytes = valid_tile_rows * in0_block_w_gu * x_tile_bytes;
                    static_assert(IN0_VALID == 1, "mcast_row sets the valid sem to 1");
                    mcast_row(cb_x_tl_obj.get_read_ptr(), x_stage_obj.get_write_ptr(), x_mcast_bytes, in0_valid_sem);
                    noc.async_write_barrier();
                    cb_x_tl_obj.pop_front(g_in0_block_tiles_max);
                    x_stage_obj.push_back(g_in0_block_tiles_max);
                }
                // The look-ahead raw block issued at the top of this iteration.
                if (x_rm_pending) {
                    noc_read.async_read_barrier();
                    cb_x_rm_obj.push_back(g_in0_block_tiles_max);
                }
            }

            // Step 3: receivers wait for both valid semaphores and push.
            if (!is_x_src) {
                if (gu_lookahead && is_in1_sender) {
                    // The sender's compute frees the look-ahead slot while this core waits for
                    // its x block, so keep re-checking until x lands.
                    while (in0_valid_sem.value() != IN0_VALID) {
                        try_gu_lookahead(kb + 1);
                    }
                } else {
                    in0_valid_sem.wait(IN0_VALID);
                }
                x_stage_obj.push_back(g_in0_block_tiles_max);
            }
            if (!is_in1_sender) {
                in1_valid_sem.wait(IN1_VALID);
                cb_in1_gate_obj.push_back(g_in1_block_num_tiles);
                if constexpr (reader_mcasts_up) {
                    cb_in1_up_obj.push_back(g_in1_block_num_tiles);
                }
            }
        }
        x_blk += num_blocks_gu;

        // IN1_WRITER_MCAST: drain the decoupled weight-multicast pipeline before phase 4.
        // in1_ready_sem is SHARED by phase 3 (gate/up mcast) and phase 4 (down mcast):
        // receivers up() it in both. Once the phase-3 multicast runs on the writer it can
        // lag into the window where receivers are already acking for phase 4, and the writer
        // then consumes acks meant for the down multicast -- after which the reader's
        // phase-4 mcast waits forever for acks that were eaten.
        if constexpr (writer_mcasts_in1) {
            if (is_in1_sender) {
                mcast_done_sem.wait_min(mc_seq);
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
        // (mrow_first_* / mrow_last_* are read once, before the expert loop.)
        constexpr uint32_t ACT_VALID = 1;
        const uint32_t intermed_tile_bytes = cb_in0_down_full_obj.get_tile_size();

        // In1_down sender: start down block kb's DRAM reads into this core's slot at l1_slot. Its
        // share (the leading d_head_tiles of every K-row) here on NoC 0 (issued only; step 3
        // barriers), and the DOWN_SPLIT writer's share (the rest of each row) on NoC 1 (released
        // via down_go; the writer reads slot (down_blk + kb) % IN1_D_SLOTS, which is this slot
        // because every block is pushed exactly once).
        // Both OOB directions are left UNWRITTEN.
        //   * K-OOB (row >= K_down_tiles, reduction dim): the compute bounds its K-loop
        //     by real_k_tiles, so these are never reduced. Were they reduced, stale L1
        //     decoding to Inf would NaN every valid output column — the bound is what
        //     makes the skip safe.
        //   * N-OOB (col >= N_down_tiles_full, free dim): the output column is dropped
        //     by the writer's col guard.
        const uint32_t down_col0 = my_nt_d * per_core_N_d;
        const uint32_t down_col_end =
            (down_col0 + per_core_N_d < N_down_tiles_full) ? down_col0 + per_core_N_d : N_down_tiles_full;
        auto release_writer_down = [&]() {
            if constexpr (split_down) {
                ++down_seq;
                down_go_sem.set(down_seq);
            }
        };
        // Leading d_head_tiles of each K-row (the whole row when DOWN_SPLIT is off).
        const uint32_t down_head_end =
            (down_col0 + d_head_tiles < down_col_end) ? down_col0 + d_head_tiles : down_col_end;
        auto issue_down_read = [&](uint32_t kb_rd, uint32_t l1_slot) {
            uint32_t l1_w = l1_slot;
            for (uint32_t k = 0; k < in0_block_w_d; ++k) {
                const uint32_t row = kb_rd * in0_block_w_d + k;
                if (row < K_down_tiles) {
                    DRuns::read(
                        noc_read, down_acc, row, N_down_tiles_full, down_col0, down_head_end, l1_w, down_tile_bytes);
                }
                l1_w += per_core_N_d * down_tile_bytes;
            }
        };
        // Block kb's reads were already issued during block kb - 1 (see the look-ahead after the
        // in1_down multicast below).
        bool down_read_issued = false;
        // Look-ahead for block kb + 1 into the ring slot after block kb's (at in1_block_start, not
        // yet pushed), taken only once this core's compute has popped enough for both to fit.
        auto try_down_lookahead = [&](uint32_t kb_cur, uint32_t cur_slot) __attribute__((noinline)) {
            if (!down_read_issued && kb_cur + 1 < num_blocks_d &&
                cb_in1_down_obj.pages_reservable_at_back(2 * d_in1_block_num_tiles) &&
                (!wmc_down || mcast_done_sem.value() >= mc_seq - 1) && pace_start(kPaceUd, false)) {
                const auto& down_cbi = get_local_cb_interface(cb_in1_down_obj.get_cb_id());
                uint32_t next_slot = cur_slot + d_in1_block_num_tiles * down_cbi.fifo_page_size;
                if (next_slot == down_cbi.fifo_limit) {
                    next_slot -= down_cbi.fifo_size;
                }
                release_writer_down();
                issue_down_read(kb_cur + 1, next_slot);
                down_read_issued = true;
            }
        };

        for (uint32_t kb = 0; kb < num_blocks_d; ++kb) {
            const bool is_act_sender = (my_nt_d == kb);

            cb_in1_down_obj.reserve_back(d_in1_block_num_tiles);
            cb_in0_down_full_obj.reserve_back(d_in0_block_num_tiles);

            // Slot reserved -> start the block unless block kb - 1 already did. The read targets only
            // this core's own slot; the receivers' acks gate the multicast alone. Block 0 keeps the
            // original order (writer's NoC-1 half now, this RISC's half after the acks, below), so
            // no extra down weight is fetched during the activation gap, where it would only take
            // DRAM share from groups still in gate/up.
            if (is_in1_sender && !down_read_issued) {
                pace_start(kPaceUd, true);
                release_writer_down();
                if (kb > 0) {
                    // Writer-multicast slots: block kb - 2 left this slot (in-order multicasts).
                    if (wmc_down) {
                        mcast_done_sem.wait_min(mc_seq - 1);
                    }
                    issue_down_read(kb, cb_in1_down_obj.get_write_ptr());
                    down_read_issued = true;
                }
            }

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

            // Step 2: in1_down sender waits for the receivers' acks and (block 0) kicks off DRAM
            // reads (NoC 0) without barriering — reads run concurrently with the activated mcast
            // below on NoC 1.
            uint32_t in1_block_start = 0;
            if (is_in1_sender) {
                if (!wmc_down) {
                    in1_ready_sem.wait(in1_num_receivers);
                    in1_ready_sem.set(0);
                } else if (!down_read_issued) {
                    // Block 0 keeps its reads-after-acks order; the writer consumes the acks.
                    in1_ready_sem.wait_min(in1_num_receivers);
                }
                in1_block_start = cb_in1_down_obj.get_write_ptr();
                if (!down_read_issued) {
                    issue_down_read(kb, in1_block_start);
                }
                down_read_issued = false;
            }

            // Step 3: in1_down sender finishes — barrier on the DRAM reads it
            // kicked off in step 2, then mcasts the down-weight block. Reordered
            // BEFORE the activated mcast: the down weight has no compute
            // dependency (DRAM-ready early), so sending it first clears NoC 1 for
            // the compute-gated activated mcast below. Tradeoff: on the one core
            // that is BOTH senders (gy==0 && gx==kb) the DRAM-read barrier is no
            // longer hidden under the activated wait — measure before keeping.
            if (is_in1_sender) {
                noc_read.async_read_barrier();
                // Bottom sender: this RISC's head is in L1, the writer multicasts the slot.
                if (wmc_down) {
                    ++mc_seq;
                    mcast_go_sem.set(mc_seq);
                }
                // DOWN_SPLIT: the writer's block must have landed before it is multicast
                // (or consumed locally at GRID_Y == 1).
                if constexpr (split_down) {
                    down_done_sem.wait_min(down_seq);
                }
                // GRID_Y == 1: no column receivers — skip mcast/valid-sem; this
                // core consumes the locally-read down weight directly.
                if (!wmc_down && in1_num_receivers > 0) {
                    // linked=true so the in1_valid-sem multicast is ordered behind
                    // the weight data on the same reserved path (see the activated
                    // mcast below for the full rationale).
                    mcast_in1(in1_block_start, d_in1_block_num_tiles * down_tile_bytes);
                    mcast_in1_valid();
                }

                // Look-ahead: start block kb + 1 now, while the activated multicast of kb and the
                // receivers' acks for kb + 1 are still in flight, if its slot is already free
                // (non-blocking: this core's compute has popped enough that both the unpushed block
                // kb and block kb + 1 fit). Its slot is the next ring slot, where the push at the
                // end of this iteration moves the write pointer. One block ahead within the
                // existing ring, so delivery order, slots and bytes are unchanged. Re-checked in
                // step 5 while this core waits for the activated block (see there).
                try_down_lookahead(kb, in1_block_start);
            }

            // Step 4: activated L1 mcast (sender for this K-block = gx==kb). Runs
            // after the in1_down mcast; starts as soon as compute pushes
            // cb_activated AND the ready acks are in (already done in step 1).
            if (is_act_sender) {
                // cb_activated carries a FULL max block (constant CB block size —
                // see adaptive_chunk.hpp), of which only the first per_core_M
                // (runtime) x in0_block_w_d tiles hold this chunk's real rows. Drain
                // the whole block but MCAST only the real tiles (#50885). Likewise
                // cb_in0_down_full is pushed FULL below, and the down matmul
                // MAC-skips the rows past per_core_M.
                const uint32_t re_act_tiles = per_core_M * in0_block_w_d;
                cb_activated_obj.wait_front(act_tiles_max);
                act_ready_sem.wait(GRID_X_NOC - 1);
                act_ready_sem.set(0);

                const uint32_t src_l1 = cb_activated_obj.get_read_ptr();
                const uint32_t dst_l1 = cb_in0_down_full_obj.get_write_ptr();
                const uint32_t mcast_bytes = re_act_tiles * intermed_tile_bytes;
                // linked=true keeps the multicast path RESERVED so the
                // valid-semaphore multicast below travels the SAME path and is
                // delivered AFTER the data at every receiver. With linked=false
                // the path is released and the valid-sem multicast can
                // overtake the bulk data multicast at a receiver -> the receiver
                // observes act_valid, pushes cb_in0_down_full, and compute reads
                // stale L1 -> that core's whole down-matmul output block is wrong
                // (run-to-run nondeterministic). Path-linking orders the sem behind
                // the data for free; the alternative, an ack-wait before sending the
                // sem, would stall this core on every receiver's ack. (The mcast is
                // non-posted and ack-counted, so an ack-wait also orders it — step 5
                // needs exactly that for the sender's own loopback copy.) Mirrors the
                // canonical matmul in0 sender
                // (reader_bmm_tile_layout_in0_sender_padding.cpp).
                static_assert(ACT_VALID == 1, "mcast_row sets the valid sem to 1");
                mcast_row(src_l1, dst_l1, mcast_bytes, act_valid_sem);

                cb_activated_obj.pop_front(act_tiles_max);
            }

            // Step 5: receivers wait for both valid sems and push. The in1_down sender keeps
            // re-checking its look-ahead while it waits: its compute frees the next slot meanwhile.
            if (!is_act_sender) {
                if (is_in1_sender) {
                    while (act_valid_sem.value() != ACT_VALID) {
                        try_down_lookahead(kb, in1_block_start);
                    }
                } else {
                    act_valid_sem.wait(ACT_VALID);
                }
            } else {
                // The sender's own copy arrives via the INCL_SRC loopback, so it needs a wait
                // too. async_writes_flushed() is not one: it polls NIU_MST_NONPOSTED_WR_REQ_SENT
                // (request left this NIU), not the ack. The mcast is non-posted, so only the
                // ack-wait proves the data LANDED before this core's compute reads the slot.
                // Placed after the valid-sem mcast so receivers are not held up; note the
                // wait is whole-queue, so it also covers that sem mcast's acks, not just
                // the loopback copy.
                noc.async_write_barrier();
            }
            cb_in0_down_full_obj.push_back(d_in0_block_num_tiles);

            if (!is_in1_sender) {
                in1_valid_sem.wait(IN1_VALID);
            }
            cb_in1_down_obj.push_back(d_in1_block_num_tiles);
        }
    }  // end chunk loop
        }  // end per-local-expert loop
#ifdef RE_PACE_CB
        // This group has started all its blocks of the paced phase: it holds nobody from here on.
        pacer.publish(kPaceDone);
#endif
        // The writer may still multicast this phase's last down blocks out of the ring. Drain it before
        // the next phase, where this core may be a receiver whose slots other senders write into.
        if (wmc_down) {
            mcast_done_sem.wait_min(mc_seq);
        }
    }  // end phase loop

    // The last in-flight Semaphore<>::set_multicast (act_valid / in1_valid)
    // is a posted atomic; without an explicit barrier it can still be in
    // flight at kernel exit, leading to timing-dependent corruption.
    noc.async_atomic_barrier();
}
