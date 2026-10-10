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

// Per-expert NCRISC wait counters for the gate/up K-block handshake, compiled only into
// device-profiler builds (like the compute kernel's REA-* counters) and reported once per
// expert as timestamped data: RD-XSEND (x sender: tilized-block + receiver-ack waits),
// RD-WBAR (own weight read barrier + writer up_done), RD-PEER (PAIR_SPLIT partner ack),
// RD-XVAL (x valid), RD-WVAL (partner weight-half valid).
#if defined(PROFILE_KERNEL)
#include "tools/profiler/kernel_profiler.hpp"
#define RD_DIAG 1
namespace {
uint32_t rd_xsend = 0;
uint32_t rd_wbar = 0;
uint32_t rd_peer = 0;
uint32_t rd_xval = 0;
uint32_t rd_wval = 0;
FORCE_INLINE uint32_t rd_now() {
    return *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
}
}  // namespace
#define RD_TIMED(acc, ...)                 \
    {                                      \
        const uint32_t rd_t0_ = rd_now();  \
        __VA_ARGS__;                       \
        acc += rd_now() - rd_t0_;          \
    }
#else
#define RD_TIMED(acc, ...) \
    { __VA_ARGS__; }
#endif

void kernel_main() {
    // -------------------------- runtime args ------------------------------
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t counts_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t idx_table_addr = get_common_arg_val<uint32_t>(2);

    const uint32_t my_mt = get_arg_val<uint32_t>(0);
    const uint32_t my_nt_gu = get_arg_val<uint32_t>(1);
    const uint32_t my_nt_d = get_arg_val<uint32_t>(2);

    // Weight-multicast runtime args (indices 3..12).
    const uint32_t is_in1_sender_u32 = get_arg_val<uint32_t>(3);
    const bool is_in1_sender = is_in1_sender_u32 != 0;
    const uint32_t in1_ready_sem_id = get_arg_val<uint32_t>(4);
    const uint32_t in1_valid_sem_id = get_arg_val<uint32_t>(5);
    const uint32_t in1_num_receivers = get_arg_val<uint32_t>(6);
    const uint32_t in1_mcast_nx_start = get_arg_val<uint32_t>(7);
    const uint32_t in1_mcast_ny_start = get_arg_val<uint32_t>(8);
    const uint32_t in1_mcast_nx_end = get_arg_val<uint32_t>(9);
    const uint32_t in1_mcast_ny_end = get_arg_val<uint32_t>(10);
    const uint32_t in1_sender_nx = get_arg_val<uint32_t>(11);
    const uint32_t in1_sender_ny = get_arg_val<uint32_t>(12);

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
    constexpr uint32_t cb_activated = get_compile_time_arg_val(20);
    constexpr uint32_t GRID_X_NOC = get_compile_time_arg_val(21);  // M-row mcast group size
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

    // DOWN_SPLIT: K-rows of each down block this RISC reads on NoC 0; the writer reads
    // [down_split_k, in0_block_w_d) on NoC 1. Equals in0_block_w_d when the split is off.
    constexpr uint32_t down_split_k = get_compile_time_arg_val(30);
    constexpr bool split_down = down_split_k < in0_block_w_d;
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

    // PAIR_SPLIT (2-row groups): both rows are weight senders. This core owns K-rows
    // [lo, hi) of every gate/up and down block, reads them into its own slot and multicasts
    // that byte range to the same slot of its partner (in1 rectangle = the partner, in1
    // "sender" coords = the partner, as ack target). It pushes a block only once its own
    // valid sem says the partner's half has landed.
#ifdef PAIR_SPLIT
    constexpr bool pair_split = true;
#else
    constexpr bool pair_split = false;
#endif
    const uint32_t pk_gu_lo = pair_split ? unified_routed_expert_ffn::pair_rows_lo(my_mt, in0_block_w_gu) : 0;
    const uint32_t pk_gu_hi = pair_split ? unified_routed_expert_ffn::pair_rows_hi(my_mt, in0_block_w_gu) : in0_block_w_gu;
    const uint32_t pk_d_lo = pair_split ? unified_routed_expert_ffn::pair_rows_lo(my_mt, in0_block_w_d) : 0;
    const uint32_t pk_d_hi = pair_split ? unified_routed_expert_ffn::pair_rows_hi(my_mt, in0_block_w_d) : in0_block_w_d;
    // Rows the reader itself reads; the writer reads [pk_d_mid, pk_d_hi) when the down split is on.
    const uint32_t pk_d_mid = pair_split ? (split_down ? unified_routed_expert_ffn::pair_rows_mid(pk_d_lo, pk_d_hi) : pk_d_hi)
                                      : down_split_k;

    // Row-major x is tilized ONCE per row: block kb's x sender tilizes it (compute, into
    // cb_x_tz) and this RISC multicasts the bf8_b tiles into every row core's cb_in0_x.
    constexpr uint32_t cb_x_tz = get_compile_time_arg_val(36);
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
    // Stateful per-bank stick reads (x sender, row-major, DRAM-interleaved x): sticks s and
    // s + NUM_DRAM_BANKS sit in the same bank one aligned page apart, so each bank's sticks of a
    // K-block leave as one set_state + N cheap with_state requests (3 register writes each)
    // instead of a full accessor address computation + request per stick.
    constexpr bool x_rm_stateful = (x_is_row_major != 0) && !x_args.is_sharded &&
                                   x_args.is_dram &&
                                   (in0_block_w_gu * TILE_HEIGHT * X_RM_ELEM_BYTES <= NOC_MAX_BURST_SIZE);
    const uint32_t x_rm_bank_stride =
        (uint32_t)x_acc_rm.get_noc_addr(NUM_DRAM_BANKS) - (uint32_t)x_acc_rm.get_noc_addr(0);

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
    // PAIR_SPLIT down read-ahead addresses the NEXT ring slot directly. cb_in1_down is exactly two
    // blocks (check_ring) and nothing has been reserved yet, so this is slot 0.
    const uint32_t down_ring_base = cb_in1_down_obj.get_write_ptr();
    // Same for the 2-block gate ring (next-pass prefetch, see gu_pf_blocks below).
    const uint32_t gate_ring_base = cb_in1_gate_obj.get_write_ptr();
    CircularBuffer cb_in0_down_full_obj(cb_in0_down_full);
    CircularBuffer cb_counts_scratch_obj(cb_counts_scratch);
    CircularBuffer cb_idx_scratch_obj(cb_idx_scratch);
    CircularBuffer cb_start_scratch_obj(cb_start_scratch);
    CircularBuffer cb_x_rm_obj(cb_x_rm);
    CircularBuffer cb_activated_obj(cb_activated);

    // x exchange CB for the in0 path: every core of a row receives each K-block of x into
    // cb_in0_x. TILE: the block's sender reads the tiles into its own slot and multicasts
    // them. Row-major: the block's sender reads the bf16 sticks into cb_x_rm one block
    // ahead, its compute tilizes them into cb_x_tz, and the sender multicasts those bf8_b
    // tiles (loopback included) into cb_in0_x, so the row tilizes each block once, not 11x.
    constexpr uint32_t x_stage_cb = cb_in0_x;
    CircularBuffer x_stage_obj(x_stage_cb);
    CircularBuffer cb_x_tz_obj(cb_x_tz);
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
    // (BRISC) can cb_wait_front on these CBs and read the same L1 data.
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

    // Rotating x sender. A fixed sender column carries the x stick reads, the row multicast and
    // its own gate-weight stream on one NCRISC every K-block; it becomes the row's slowest core
    // and the other 10 wait for x behind it. Instead, gate/up K-block j (counted across every
    // chunk and expert; all cores of a physical row run the same block sequence) is sent by
    // column (base + j) % GRID_X, base = the factory's sender column (keeps rows staggered).
    // The multicast rectangle is already the whole row (non-loopback, GRID_X - 1 dests), so it
    // serves any sender column; receivers ack the block's sender via the M-row NoC table.
    uint32_t x_send_col = 0;
    for (uint32_t i = 0; i < GRID_X_NOC; ++i) {
        if (get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * i + 0) == in0_sender_nx &&
            get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * i + 1) == in0_sender_ny) {
            x_send_col = i;
        }
    }
    (void)is_in0_sender;
    // The sender's in0_valid set_multicast reads its value from this core's own sem word, which is
    // also this core's receive flag in its next (receiver) block: flush the multicast before the reset.
    bool x_sent_prev = false;

    // ======================= per-local-expert loop =======================
    // Per expert we resolve its global id (idx_table[e]), token count
    // (counts[global_id]), region offset (start[global_id]) and per-expert
    // weight base addresses, then drive the same chunked matmul pipeline.
    // Expert-parallel row groups: this core's group runs local experts
    // expert_first, expert_first + expert_stride, ...; grid_m rows share each expert's M.
    const uint32_t expert_first = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * GRID_X_NOC + 0);
    const uint32_t expert_stride = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * GRID_X_NOC + 1);
    const uint32_t grid_m = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * GRID_X_NOC + 2);
    // Pair tail (see adaptive_chunk::sched_first and the phase-4 exchange below).
    constexpr uint32_t PAIR_RT = M_ROW_NOC_RT_OFFSET + 2 * GRID_X_NOC + 3;
    const uint32_t full_end = get_arg_val<uint32_t>(PAIR_RT + 0);
    const uint32_t tail_expert = get_arg_val<uint32_t>(PAIR_RT + 1);
    const uint32_t tail_part = get_arg_val<uint32_t>(PAIR_RT + 2);
    Semaphore<> pair_ready_sem(get_arg_val<uint32_t>(PAIR_RT + 3));
    Semaphore<> pair_valid1_sem(get_arg_val<uint32_t>(PAIR_RT + 4));
    const uint32_t pair_ny_lo = get_arg_val<uint32_t>(PAIR_RT + 5);
    const uint32_t pair_ny_hi = get_arg_val<uint32_t>(PAIR_RT + 6);
    const uint32_t pair_row_ny[2] = {get_arg_val<uint32_t>(PAIR_RT + 7), get_arg_val<uint32_t>(PAIR_RT + 8)};
    // cb_in0_down_full blocks pushed so far. The pair exchange multicasts into the SENDER's
    // write address, so both rows of a pair must enter the tail expert at the same ring
    // position; the full rounds before it may have pushed different block counts.
    uint32_t down_full_blocks = 0;
    // Next-pass gate/up prefetch (G=1 rows only: every core reads its own weights). The
    // activation pass of all rows lines up chip-wide and the DRAM idles through it, while the
    // gate/up rings sit empty from the end of gate/up until the next pass. So at down K-block 0
    // the reader issues the next pass's gate K-blocks 0..1 into both gate slots (no barrier)
    // and releases the writer's matching up reads via up_go; that pass's gate/up loop then
    // skips the issue for its first gu_pf_blocks blocks. up_seq still counts gate/up blocks.
    const bool gu_prefetch = (grid_m == 1) && is_in1_sender && up_split && !pair_split && !writer_mcasts_in1;
    uint32_t gu_pf_blocks = 0;
    // Count-sorted placement (adaptive_chunk::slot_order): the schedule walks slots, and this
    // table picks the expert of each slot. Compute and writer build the same table.
    uint8_t expert_of_slot[experts_per_chip];
    adaptive_chunk::slot_order(
        expert_of_slot, counts_ptr, idx_ptr, min_active_tokens, max_active_tokens, expert_stride, full_end);
    for (uint32_t sched_slot = adaptive_chunk::sched_first(expert_first, full_end, tail_expert);
         sched_slot < experts_per_chip;
         sched_slot = adaptive_chunk::sched_next(sched_slot, expert_stride, full_end, tail_expert, experts_per_chip)) {
        const uint32_t local_expert_id = expert_of_slot[sched_slot];
        // Pair-tail expert: this core owns column part tail_part of 2 of its gate/up and
        // down slices (reads only those weight columns) and exchanges activated halves with
        // the partner row.
        const bool pair_item = sched_slot >= full_end;
        const uint32_t pf = pair_item ? 2u : 1u;
        const uint32_t pp = pair_item ? tail_part : 0u;
        const uint32_t gu_lo = adaptive_chunk::part_lo(per_core_N_gu, pf, pp);
        const uint32_t gu_hi = adaptive_chunk::part_hi(per_core_N_gu, pf, pp);
        const uint32_t d_lo = adaptive_chunk::part_lo(per_core_N_d, pf, pp);
        const uint32_t d_hi = adaptive_chunk::part_hi(per_core_N_d, pf, pp);
        if (pair_item && (down_full_blocks & 1u)) {
            // Pointer-only pad to an even (== 0 of the 2-block ring) position; the compute
            // kernel drops the same block.
            cb_in0_down_full_obj.reserve_back(d_in0_block_num_tiles);
            cb_in0_down_full_obj.push_back(d_in0_block_num_tiles);
            ++down_full_blocks;
        }
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
        const auto down_acc = TensorAccessor(down_args, down_addr_e, down_tile_bytes);

        const uint32_t global_expert_id = idx_ptr[local_expert_id];
        // Hybrid dispatch: experts outside this op's band belong to the other routed-expert
        // op and are dropped here exactly like a zero count.
        const uint32_t count_value =
            adaptive_chunk::count_in_band(counts_ptr[global_expert_id], min_active_tokens, max_active_tokens);
        // counts[] is device-produced and unvalidated: bound it by the capacity
        // this program was built for (num_chunks_max * chunk_M_max tile-rows)
        // BEFORE deriving anything from it. The clamp is arithmetic so it also
        // holds in Release, where ASSERT is a no-op; the assert below still
        // hard-fails a mismatch in watcher builds. Compute and writer clamp
        // identically, so all three keep the same row mapping (see
        // adaptive_chunk::clamp_count_tiles).
        const uint32_t count_tiles_raw = (count_value + TILE_HEIGHT - 1) / TILE_HEIGHT;
        const uint32_t count_tiles =
            adaptive_chunk::clamp_count_tiles(count_tiles_raw, chunk_M_max, num_chunks_max, M_tiles_full);
        ASSERT(count_tiles == count_tiles_raw);
        // Runtime chunk layout from THIS expert's actual token count (identical
        // math in the compute and writer kernels, so all three agree on the row
        // mapping). Full chunks span chunk_M_max tile-rows; the tail chunk shrinks
        // per_core_M to the remainder, so per_core_M is per-chunk (see the chunk
        // loop) and adapts to each expert's own load with minimal phantom work.
        const uint32_t effective_chunks = adaptive_chunk::num_chunks(count_tiles, chunk_M_max);

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
                    noc_read.async_read(gate_bias_acc, CoreLocalMem<uint32_t>(lg), gbias_tb, {.page_id = col}, {});
                    noc_read.async_read(up_bias_acc, CoreLocalMem<uint32_t>(lu), gbias_tb, {.page_id = col}, {});
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
                    noc_read.async_read(down_bias_acc, CoreLocalMem<uint32_t>(ld), dbias_tb, {.page_id = col}, {});
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
        const uint32_t per_core_M = adaptive_chunk::per_core_M_for_chunk(chunk, count_tiles, chunk_M_max, grid_m);
        const uint32_t this_core_first_row = chunk * chunk_M_max + my_mt * per_core_M;

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
        // Row-major: read K-block xb's column window (in0_block_w_gu tiles wide) from each
        // real token-row stick of this row's tile-rows into cb_x_rm, laid contiguously so each
        // tile-row forms one TILE_HEIGHT x (in0_block_w_gu*32) strip for the tilize. Rows past
        // count_tiles and the padding sticks of the last tile-row are NOT read: they stay stale,
        // and no FFN stage reduces across M, so the garbage stays confined to rows the writer
        // drops. Called only by the block's x sender (once per GRID_X blocks).
        const auto read_x_sticks = [&](uint32_t xb) {
            constexpr uint32_t rm_kblock_bytes = in0_block_w_gu * TILE_HEIGHT * X_RM_ELEM_BYTES;
            const uint32_t col_off_bytes = xb * rm_kblock_bytes;
            cb_x_rm_obj.reserve_back(g_in0_block_tiles_max);
            const uint32_t block_start = cb_x_rm_obj.get_write_ptr();
            if constexpr (x_rm_stateful) {
                // The real sticks are a contiguous prefix [s0, s0 + n) of this core's rows
                // (only the last valid tile-row is partial), landing at block_start + i * B.
                uint32_t n = 0;
                if (this_core_first_row < count_tiles) {
                    const uint32_t row_end = this_core_first_row + per_core_M;
                    const uint32_t stick_end = (row_end < count_tiles) ? row_end * TILE_HEIGHT : count_value;
                    n = stick_end - this_core_first_row * TILE_HEIGHT;
                }
                const uint32_t s0 = x_start_stick + this_core_first_row * TILE_HEIGHT;
                const uint32_t nb = (n < NUM_DRAM_BANKS) ? n : NUM_DRAM_BANKS;
                const uint8_t xnoc = noc_read.get_noc_id();
                for (uint32_t j = 0; j < nb; ++j) {
                    const uint64_t a = x_acc_rm.get_noc_addr(s0 + j, col_off_bytes);
                    noc_async_read_one_packet_set_state(a, rm_kblock_bytes, 0, xnoc);
                    uint32_t src = (uint32_t)a;
                    uint32_t dst = block_start + j * rm_kblock_bytes;
                    for (uint32_t i = j; i < n; i += NUM_DRAM_BANKS) {
                        noc_async_read_one_packet_with_state(src, dst, 0, xnoc);
                        src += x_rm_bank_stride;
                        dst += NUM_DRAM_BANKS * rm_kblock_bytes;
                    }
                }
            } else {
                uint32_t l1_x = block_start;
                for (uint32_t m = 0; m < per_core_M; ++m) {
                    const uint32_t tile_row = this_core_first_row + m;
                    if (tile_row < count_tiles) {
                        const uint32_t row_base = tile_row * TILE_HEIGHT;
                        const uint32_t real_r =
                            (row_base + TILE_HEIGHT <= count_value) ? TILE_HEIGHT : (count_value - row_base);
                        for (uint32_t r = 0; r < real_r; ++r) {
                            const uint32_t stick = x_start_stick + row_base + r;
                            noc_read.async_read(
                                x_acc_rm,
                                CoreLocalMem<uint32_t>(l1_x),
                                rm_kblock_bytes,
                                {.page_id = stick, .offset_bytes = col_off_bytes},
                                {});
                            l1_x += rm_kblock_bytes;
                        }
                        l1_x += (TILE_HEIGHT - real_r) * rm_kblock_bytes;
                    } else {
                        l1_x += TILE_HEIGHT * rm_kblock_bytes;
                    }
                }
            }
            noc_read.async_read_barrier();
            cb_x_rm_obj.push_back(g_in0_block_tiles_max);
        };

        for (uint32_t kb = 0; kb < num_blocks_gu; ++kb) {
            const bool is_x_sender = (my_nt_gu == x_send_col);
            if constexpr (x_is_row_major != 0) {
                // Sticks for the block this core sends: block 0 of the chunk in iteration 0,
                // otherwise one block ahead, so compute tilizes block kb+1 before it MACs kb.
                const uint32_t next_send_col = (x_send_col + 1 == GRID_X_NOC) ? 0 : x_send_col + 1;
                if (kb == 0 && is_x_sender) {
                    read_x_sticks(0);
                }
                if (kb + 1 < num_blocks_gu && my_nt_gu == next_send_col) {
                    read_x_sticks(kb + 1);
                }
            }
            x_stage_obj.reserve_back(g_in0_block_tiles_max);
            cb_in1_gate_obj.reserve_back(g_in1_block_num_tiles);
            if constexpr (reader_mcasts_up) {
                cb_in1_up_obj.reserve_back(g_in1_block_num_tiles);
            }

            // UP_SPLIT: slot reserved -> release writer to read `up` on NoC 1,
            // concurrent with the reader's NoC-0 `gate` read below.
            // Prefetched block: its gate read and up_go were issued during the previous down phase.
            const bool w_prefetched = kb < gu_pf_blocks;
            if constexpr (up_split) {
                if (is_in1_sender && !w_prefetched) {
                    ++up_seq;
                    up_go_sem.set(up_seq);
                }
            }

            // Step 1: receivers ack BOTH senders upfront so both senders can
            // proceed in parallel. The senders are usually disjoint sets of
            // cores; the only core that's both senders is (0,0) which doesn't
            // self-inc (it's its own sender for both).
            if (!is_x_sender) {
                if (x_sent_prev) {
                    noc.async_writes_flushed();
                }
                in0_valid_sem.set(0);
                in0_ready_sem.up(
                    noc,
                    get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * x_send_col + 0),
                    get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * x_send_col + 1),
                    1);
            }
            if (!is_in1_sender || pair_split) {
                in1_valid_sem.set(0);
                in1_ready_sem.up(noc, in1_sender_nx, in1_sender_ny, 1);
            }

            // Weight sender: issue this block's gate (and LEGACY up) DRAM reads into this core's own,
            // already reserved slot, without a barrier. Row-major x: issued BEFORE the x sender path
            // below, so the K-block's x sender no longer starts its weight half only after the
            // row-wide x ack wait + multicast (with PAIR_SPLIT the partner's push waited on that
            // chain too). The pre-read waits are the same as before; the barrier stays in the
            // weight path after the x path. TILE x keeps the old order (its x reads share the
            // read barrier of the x path).
            uint32_t gate_block_start = 0;
            uint32_t up_block_start = 0;
            const auto issue_weight_reads = [&]() {
                if constexpr (writer_mcasts_in1) {
                    // The writer multicasts, so it owns the receivers' ready handshake. This
                    // RISC only needs its own slot back: cb_in1_* is double-buffered, so the
                    // block numbered mc_seq reuses the slot mc_seq-2 used, and THAT block's
                    // multicast must have drained. Waiting on mc_seq-2 rather than mc_seq is
                    // what lets this DRAM read overlap the writer's multicast of the previous
                    // block, which is the whole point.
                    ++mc_seq;
                    if (mc_seq >= 3) {
                        mcast_done_sem.wait_min(mc_seq - 2);
                    }
                } else if constexpr (!pair_split) {
                    in1_ready_sem.wait(in1_num_receivers);
                    in1_ready_sem.set(0);
                }

                // DRAM read gate region first.
                uint32_t l1_w_gate = cb_in1_gate_obj.get_write_ptr();
                gate_block_start = l1_w_gate;
                if (w_prefetched) {
                    if constexpr (reader_mcasts_up) {
                        up_block_start = cb_in1_up_obj.get_write_ptr();
                    }
                    return;
                }
                // N-OOB hidden padding columns (col >= N_gate_tiles_full == down's K_down_tiles)
                // are left UNWRITTEN: their gate output lands on a down K position the down
                // matmul never reduces (see the compute's real_k_tiles bound), so the stale L1
                // is dropped. Mirrors the down-weight K-OOB skip below.
                // Pair tail: only this core's column part [gu_lo, gu_hi) of the slice.
                // PAIR_SPLIT: only this core's K-rows [pk_gu_lo, pk_gu_hi) of the block.
                const uint32_t gate_col0 = my_nt_gu * per_core_N_gu + gu_lo;
                const uint32_t gate_col_hi = my_nt_gu * per_core_N_gu + gu_hi;
                const uint32_t gate_col_end = (gate_col_hi < N_gate_tiles_full) ? gate_col_hi : N_gate_tiles_full;
                l1_w_gate += pk_gu_lo * per_core_N_gu * gate_tile_bytes;
                for (uint32_t k = pk_gu_lo; k < pk_gu_hi; ++k) {
                    GuRuns::read(
                        noc_read,
                        gate_acc,
                        kb * in0_block_w_gu + k,
                        N_gate_tiles_full,
                        gate_col0,
                        gate_col_end,
                        l1_w_gate + gu_lo * gate_tile_bytes,
                        gate_tile_bytes);
                    l1_w_gate += per_core_N_gu * gate_tile_bytes;
                }
                // `up` slot. LEGACY: reader reads it on NoC 0. UP_SPLIT: writer
                // already read it on NoC 1; reader just takes the L1 start and
                // waits on up_done (below) before mcasting.
                if constexpr (reader_mcasts_up) {
                    up_block_start = cb_in1_up_obj.get_write_ptr();
                }
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
                            l1_w_up + gu_lo * up_tile_bytes,
                            up_tile_bytes);
                        l1_w_up += per_core_N_gu * up_tile_bytes;
                    }
                }
            };
            constexpr bool weight_reads_early = (x_is_row_major != 0);
            if constexpr (weight_reads_early) {
                if (is_in1_sender) {
                    issue_weight_reads();
                }
            }

            // Step 2: senders run their work. in0 sender path and in1 sender
            // path can each start as soon as their ready sem is satisfied —
            // for the common case where a core is one type of sender, the
            // work begins immediately. For core (0,0) (both senders), in0
            // runs first then in1, ~60µs sequentially — same as before.
            if (is_x_sender) {
                // The DRAM reads only fill this core's own (already reserved) slot; the receivers'
                // ready acks gate just the multicast, so they are awaited after the reads issue.
                uint32_t l1_x = x_stage_obj.get_write_ptr();
                const uint32_t block_start = l1_x;
                // Rows past count_tiles are NOT valid — they hold uninitialized
                // DRAM. Skip the read and leave the stale L1 as-is: no FFN stage
                // reduces across M (tokens), so garbage stays confined to its own
                // row, and the writer drops every row >= count_tiles (its
                // `row < count_tiles` guard), so it never reaches the output.
                if constexpr (x_is_row_major != 0) {
                    // Row-major: this core's compute tilized the block into cb_x_tz (from the
                    // sticks read one iteration ago); nothing to read here.
                    RD_TIMED(rd_xsend, cb_x_tz_obj.wait_front(g_in0_block_tiles_max));
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
                            // Invalid row (>= count_tiles): skip the read, advance the
                            // write ptr. Stale L1 here is dropped by the writer (see above).
                            l1_x += in0_block_w_gu * x_tile_bytes;
                        }
                    }
                }
                RD_TIMED(rd_xsend, in0_ready_sem.wait(in0_num_receivers));
                in0_ready_sem.set(0);
                noc_read.async_read_barrier();

                // Multicast only the real rows of the block. Valid tile-rows form a
                // contiguous prefix from block_start; padding rows past the token
                // count stay stale at receivers and are dropped downstream (same
                // free-dim-M safety as the skipped reads above), in whole valid tile-rows
                // (a 32-row tile can't be split). A fully-padding M-row (this_core_first_row >= count_tiles)
                // sends no data — only the valid sem, so receivers still advance.
                const uint32_t valid_tile_rows =
                    (this_core_first_row < count_tiles)
                        ? ((count_tiles - this_core_first_row < per_core_M) ? (count_tiles - this_core_first_row)
                                                                            : per_core_M)
                        : 0;
                // Both paths send whole valid tile-rows of bf8_b tiles (row-major: the tilized block).
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
                if constexpr (x_is_row_major != 0) {
                    // Row-major: the tilized block goes to the whole row INCLUDING this core
                    // (its own cb_in0_x slot is block_start too), then the valid sem on the same
                    // linked path. The own copy is awaited (ack barrier) in step 3, before the
                    // push, and cb_x_tz is popped only then (it is the multicast source).
                    if (mcast_bytes > 0) {
                        noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                            CoreLocalMem<uint32_t>(cb_x_tz_obj.get_read_ptr()),
                            MulticastEndpoint{},
                            mcast_bytes,
                            in0_num_receivers + 1,
                            {.offset_bytes = 0},
                            {.noc_x_start = in0_mcast_nx_start,
                             .noc_y_start = in0_mcast_ny_start,
                             .noc_x_end = in0_mcast_nx_end,
                             .noc_y_end = in0_mcast_ny_end,
                             .addr = block_start},
                            /*linked=*/true);
                    }
                    noc.async_writes_flushed();
                    in0_valid_sem.set(IN0_VALID);
                    in0_valid_sem.set_multicast<NocOptions::MCAST_INCL_SRC>(
                        noc,
                        in0_mcast_nx_start,
                        in0_mcast_ny_start,
                        in0_mcast_nx_end,
                        in0_mcast_ny_end,
                        in0_num_receivers + 1);
                } else {
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
                        noc,
                        in0_mcast_nx_start,
                        in0_mcast_ny_start,
                        in0_mcast_nx_end,
                        in0_mcast_ny_end,
                        in0_num_receivers);
                }
            }

            if (is_in1_sender) {
                if constexpr (!weight_reads_early) {
                    issue_weight_reads();
                }
                RD_TIMED(rd_wbar, noc_read.async_read_barrier());

                // IN1_WRITER_MCAST: gate is in L1 -- hand this slot to the writer, which
                // multicasts it (and `up`, which it read itself) on NoC 1. Signalled BEFORE
                // the up_done wait below so the writer starts as early as possible.
                if constexpr (writer_mcasts_in1) {
                    mcast_go_sem.set(mc_seq);
                }

                // UP_SPLIT: wait for the writer's NoC-1 `up` read before mcast.
                if constexpr (up_split) {
                    RD_TIMED(rd_wbar, up_done_sem.wait_min(up_seq));
                }
                // GRID_Y == 1: no column receivers — skip mcast/valid-sem; the
                // locally-read weights go straight to compute via cb_push_back.
                // IN1_WRITER_MCAST moves this whole multicast to the writer's NoC 1.
                // PAIR_SPLIT: send only this core's K-rows, once the partner's slot is free.
                if constexpr (pair_split) {
                    RD_TIMED(rd_peer, in1_ready_sem.wait(in1_num_receivers));
                    in1_ready_sem.set(0);
                }
                const uint32_t gu_part_tiles = (pk_gu_hi - pk_gu_lo) * per_core_N_gu;
                const uint32_t gu_part_off_tiles = pk_gu_lo * per_core_N_gu;
                if (!writer_mcasts_in1 && in1_num_receivers > 0) {
                    const uint32_t gate_block_bytes = gu_part_tiles * gate_tile_bytes;
                    const uint32_t gate_send_start = gate_block_start + gu_part_off_tiles * gate_tile_bytes;
                    // The LAST in1 data multicast before the in1_valid sem must
                    // be linked=true so the (posted) valid-sem multicast travels
                    // the SAME reserved path and lands AFTER the data at every
                    // receiver. Otherwise, under NoC contention (heavy fabric
                    // load), the valid sem can overtake the weight data -> the
                    // receiver pushes cb_in1_{gate,up} and compute reads STALE
                    // weights -> wrong matmul output (rare, timing-dependent;
                    // a flush/barrier does not fix posted multicast writes on
                    // Blackhole). Mirrors the phase-4 activated mcast. When `up`
                    // is mcast (LEGACY/UP_SPLIT) it is the last write, so gate
                    // links into it and up holds the path for the sem; in the
                    // retired UP_WRITER_MCAST mode (no up mcast) gate is last and
                    // holds the path itself.
                    noc.async_write_multicast(
                        CoreLocalMem<uint32_t>(gate_send_start),
                        MulticastEndpoint{},
                        gate_block_bytes,
                        in1_num_receivers,
                        {.offset_bytes = 0},
                        {.noc_x_start = in1_mcast_nx_start,
                         .noc_y_start = in1_mcast_ny_start,
                         .noc_x_end = in1_mcast_nx_end,
                         .noc_y_end = in1_mcast_ny_end,
                         .addr = gate_send_start},
                        /*linked=*/true);

                    if constexpr (reader_mcasts_up) {
                        const uint32_t up_block_bytes = gu_part_tiles * up_tile_bytes;
                        const uint32_t up_send_start = up_block_start + gu_part_off_tiles * up_tile_bytes;
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(up_send_start),
                            MulticastEndpoint{},
                            up_block_bytes,
                            in1_num_receivers,
                            {.offset_bytes = 0},
                            {.noc_x_start = in1_mcast_nx_start,
                             .noc_y_start = in1_mcast_ny_start,
                             .noc_x_end = in1_mcast_nx_end,
                             .noc_y_end = in1_mcast_ny_end,
                             .addr = up_send_start},
                            /*linked=*/!pair_split);
                    }
                }
                // PAIR_SPLIT pushes in step 3, after the partner's half has landed.
                if constexpr (!pair_split) {
                    cb_in1_gate_obj.push_back(g_in1_block_num_tiles);
                    if constexpr (reader_mcasts_up) {
                        cb_in1_up_obj.push_back(g_in1_block_num_tiles);
                    }
                }
                if constexpr (pair_split) {
                    // This core's in1_valid_sem is its OWN receive flag here, so it cannot be the
                    // source of a set_multicast. Wait for the data multicast's acks (it is
                    // non-posted), then atomically bump the partner's flag.
                    noc.async_write_barrier();
                    in1_valid_sem.up(noc, in1_sender_nx, in1_sender_ny, 1);
                } else if (!writer_mcasts_in1 && in1_num_receivers > 0) {
                    noc.async_writes_flushed();
                    in1_valid_sem.set(IN1_VALID);
                    in1_valid_sem.set_multicast<NocOptions::DEFAULT>(
                        noc,
                        in1_mcast_nx_start,
                        in1_mcast_ny_start,
                        in1_mcast_nx_end,
                        in1_mcast_ny_end,
                        in1_num_receivers);
                }
            }

            // Step 3: receivers wait for both valid semaphores and push.
            if (!is_x_sender) {
                RD_TIMED(rd_xval, in0_valid_sem.wait(IN0_VALID));
                x_stage_obj.push_back(g_in0_block_tiles_max);
            } else if constexpr (x_is_row_major != 0) {
                // Own loopback copy: only the ack-wait proves it landed (the multicast is
                // non-posted; a flush only means the request left). It also frees cb_x_tz.
                noc.async_write_barrier();
                cb_x_tz_obj.pop_front(g_in0_block_tiles_max);
                x_stage_obj.push_back(g_in0_block_tiles_max);
            }
            x_sent_prev = is_x_sender;
            x_send_col = (x_send_col + 1 == GRID_X_NOC) ? 0 : x_send_col + 1;
            if (!is_in1_sender || pair_split) {
                RD_TIMED(rd_wval, in1_valid_sem.wait(IN1_VALID));
                cb_in1_gate_obj.push_back(g_in1_block_num_tiles);
                if constexpr (reader_mcasts_up) {
                    cb_in1_up_obj.push_back(g_in1_block_num_tiles);
                }
            }
        }
        gu_pf_blocks = 0;

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
        const uint32_t mrow_first_nx = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 0);
        const uint32_t mrow_first_ny = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 1);
        const uint32_t mrow_last_nx = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * (GRID_X_NOC - 1) + 0);
        const uint32_t mrow_last_ny = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * (GRID_X_NOC - 1) + 1);
        constexpr uint32_t ACT_VALID = 1;
        const uint32_t intermed_tile_bytes = cb_in0_down_full_obj.get_tile_size();

        // This RISC's DRAM reads of down block kb_i into the ring slot at slot_l1, no barrier.
        // DOWN_SPLIT: only the rows this RISC keeps; the writer reads the rest on
        // NoC 1 into the upper part of the same slot.
        // Both OOB directions are left UNWRITTEN.
        //   * K-OOB (row >= K_down_tiles, reduction dim): the compute bounds its K-loop
        //     by real_k_tiles, so these are never reduced. Were they reduced, stale L1
        //     decoding to Inf would NaN every valid output column — the bound is what
        //     makes the skip safe.
        //   * N-OOB (col >= N_down_tiles_full, free dim): the output column is dropped
        //     by the writer's col guard.
        // Pair tail: only this core's column part [d_lo, d_hi) of the slice.
        // PAIR_SPLIT: only this RISC's K-rows [pk_d_lo, pk_d_mid) of the block.
        const uint32_t down_col0 = my_nt_d * per_core_N_d + d_lo;
        const uint32_t down_col_hi = my_nt_d * per_core_N_d + d_hi;
        const uint32_t down_col_end = (down_col_hi < N_down_tiles_full) ? down_col_hi : N_down_tiles_full;
        auto issue_down_rows = [&](uint32_t kb_i, uint32_t slot_l1) {
            uint32_t l1_w = slot_l1 + pk_d_lo * per_core_N_d * down_tile_bytes;
            for (uint32_t k = pk_d_lo; k < pk_d_mid; ++k) {
                const uint32_t row = kb_i * in0_block_w_d + k;
                if (row < K_down_tiles) {
                    DRuns::read(
                        noc_read,
                        down_acc,
                        row,
                        N_down_tiles_full,
                        down_col0,
                        down_col_end,
                        l1_w + d_lo * down_tile_bytes,
                        down_tile_bytes);
                }
                l1_w += per_core_N_d * down_tile_bytes;
            }
        };

        // PAIR_SPLIT down read-ahead. The per-block chain (own reads -> barrier -> partner half
        // -> activated row exchange -> partner valid -> push) used to issue block kb+1's DRAM
        // reads only at the top of iteration kb+1, so each core had one block in flight and the
        // DRAM round trip sat on every block. Here block kb+1's reads (this RISC's rows, plus the
        // writer's via down_go) are issued as soon as block kb's own half has been sent to the
        // partner, provided the next ring slot is already free (compute popped kb-1; checked
        // without blocking so the activated exchange is never held up). Otherwise it is retried
        // after the activated exchange, and failing that issued at the top of kb+1 as before.
        // The partner writes its half of slot kb+1 only after this core's ack at the top of
        // kb+1, into the other K-rows, so it cannot collide with the early reads.
        const uint32_t down_slot_bytes = d_in1_block_num_tiles * down_tile_bytes;
        bool down_pre_issued = false;  // block kb's reads were issued during iteration kb-1
        // G=1 rows read every down block themselves (no multicast), so the same read-ahead applies.
        const bool down_read_ahead = pair_split || (grid_m == 1 && is_in1_sender);

        // Next gate/up pass of this row: the next chunk of this expert, else chunk 0 of the next
        // scheduled expert that runs (the writer walks the same schedule for its up half).
        uint32_t pf_slot = experts_per_chip;
        if (gu_prefetch) {
            pf_slot = (chunk + 1 < effective_chunks) ? sched_slot
                                                       : adaptive_chunk::next_active_slot(
                                                             sched_slot,
                                                             expert_stride,
                                                             full_end,
                                                             tail_expert,
                                                             experts_per_chip,
                                                             expert_of_slot,
                                                             counts_ptr,
                                                             idx_ptr,
                                                             min_active_tokens,
                                                             max_active_tokens);
        }
        // Issue the next pass's gate K-blocks 0..pf_n-1 into the gate ring and release the writer's
        // up reads of the same blocks. Blocks until compute has drained the rings (it has finished
        // this pass's gate/up matmul, which needs nothing more from this RISC), so it runs while
        // the activation is still on, before the activated exchange of down block 0.
        auto prefetch_next_gu = [&](uint32_t ps) {
            constexpr uint32_t pf_n = (num_blocks_gu < 2) ? num_blocks_gu : 2;
            cb_in1_gate_obj.reserve_back(pf_n * g_in1_block_num_tiles);
            if constexpr (reader_mcasts_up) {
                cb_in1_up_obj.reserve_back(pf_n * g_in1_block_num_tiles);
            }
            const bool pe_pair = ps >= full_end;
            const uint32_t pe = expert_of_slot[ps];
            const uint32_t pe_lo = adaptive_chunk::part_lo(per_core_N_gu, pe_pair ? 2u : 1u, pe_pair ? tail_part : 0u);
            const uint32_t pe_hi = adaptive_chunk::part_hi(per_core_N_gu, pe_pair ? 2u : 1u, pe_pair ? tail_part : 0u);
            const uint32_t col0 = my_nt_gu * per_core_N_gu + pe_lo;
            const uint32_t col_hi = my_nt_gu * per_core_N_gu + pe_hi;
            const uint32_t col_end = (col_hi < N_gate_tiles_full) ? col_hi : N_gate_tiles_full;
            const auto pe_gate_acc = TensorAccessor(
                gate_args, get_common_arg_val<uint32_t>(COMMON_WEIGHTS + 0 * experts_per_chip + pe), gate_tile_bytes);
            const uint32_t gate_slot_bytes = g_in1_block_num_tiles * gate_tile_bytes;
            uint32_t slot = cb_in1_gate_obj.get_write_ptr();
            for (uint32_t j = 0; j < pf_n; ++j) {
                ++up_seq;
                up_go_sem.set(up_seq);
                uint32_t l1_w = slot;
                for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
                    GuRuns::read(
                        noc_read,
                        pe_gate_acc,
                        j * in0_block_w_gu + k,
                        N_gate_tiles_full,
                        col0,
                        col_end,
                        l1_w + pe_lo * gate_tile_bytes,
                        gate_tile_bytes);
                    l1_w += per_core_N_gu * gate_tile_bytes;
                }
                slot = (slot == gate_ring_base) ? gate_ring_base + gate_slot_bytes : gate_ring_base;
            }
            gu_pf_blocks = pf_n;
        };

        for (uint32_t kb = 0; kb < num_blocks_d; ++kb) {
            const bool is_act_sender = (my_nt_d == kb);

            cb_in1_down_obj.reserve_back(d_in1_block_num_tiles);
            cb_in0_down_full_obj.reserve_back(d_in0_block_num_tiles);

            // DOWN_SPLIT: slot reserved -> release the writer to read the block on NoC 1.
            if constexpr (split_down) {
                if (is_in1_sender && !down_pre_issued) {
                    ++down_seq;
                    down_go_sem.set(down_seq);
                }
            }

            // Step 1: receivers ack BOTH senders (in1_down and act) at the
            // top of the K-block iter. The in1_down ack lets the in1_down
            // sender immediately start DRAM reads; the act ack lets the act
            // sender start mcasting as soon as compute pushes cb_activated.
            // Without the early act ack the sender would only see receivers
            // after the in1_down section finishes, serializing the two paths.
            if (!is_in1_sender || pair_split) {
                in1_valid_sem.set(0);
                in1_ready_sem.up(noc, in1_sender_nx, in1_sender_ny, 1);
            }
            if (pair_item) {
                // Pair exchange: K-block kb's activated columns live half on (kb, row 2j) and
                // half on (kb, row 2j + 1). Ack both senders (except ourselves) on the
                // dedicated pair_ready sem. Both valid sems are reset BEFORE the acks, so no
                // set for this block can land ahead of the reset.
                act_valid_sem.set(0);
                pair_valid1_sem.set(0);
                const uint32_t sender_nx = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * kb + 0);
                for (uint32_t q = 0; q < 2; ++q) {
                    if (!(is_act_sender && q == tail_part)) {
                        pair_ready_sem.up(noc, sender_nx, pair_row_ny[q], 1);
                    }
                }
            } else if (!is_act_sender) {
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
                if constexpr (!pair_split) {
                    in1_ready_sem.wait(in1_num_receivers);
                    in1_ready_sem.set(0);
                }
                in1_block_start = cb_in1_down_obj.get_write_ptr();
                if (!down_pre_issued) {
                    issue_down_rows(kb, in1_block_start);
                }
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
                // DOWN_SPLIT: the writer's block must have landed before it is multicast
                // (or consumed locally at GRID_Y == 1).
                if constexpr (split_down) {
                    down_done_sem.wait_min(down_seq);
                }
                // GRID_Y == 1: no column receivers — skip mcast/valid-sem; this
                // core consumes the locally-read down weight directly.
                if constexpr (pair_split) {
                    in1_ready_sem.wait(in1_num_receivers);
                    in1_ready_sem.set(0);
                }
                if (in1_num_receivers > 0) {
                    const uint32_t block_bytes = (pk_d_hi - pk_d_lo) * per_core_N_d * down_tile_bytes;
                    const uint32_t send_start = in1_block_start + pk_d_lo * per_core_N_d * down_tile_bytes;
                    // linked=true so the in1_valid-sem multicast is ordered behind
                    // the weight data on the same reserved path (see the activated
                    // mcast below for the full rationale).
                    noc.async_write_multicast(
                        CoreLocalMem<uint32_t>(send_start),
                        MulticastEndpoint{},
                        block_bytes,
                        in1_num_receivers,
                        {.offset_bytes = 0},
                        {.noc_x_start = in1_mcast_nx_start,
                         .noc_y_start = in1_mcast_ny_start,
                         .noc_x_end = in1_mcast_nx_end,
                         .noc_y_end = in1_mcast_ny_end,
                         .addr = send_start},
                        /*linked=*/!pair_split);
                    if constexpr (pair_split) {
                        // Own valid sem is this core's receive flag: ack-wait, then bump the partner's.
                        noc.async_write_barrier();
                        in1_valid_sem.up(noc, in1_sender_nx, in1_sender_ny, 1);
                    } else {
                        noc.async_writes_flushed();

                        in1_valid_sem.set(IN1_VALID);
                        in1_valid_sem.set_multicast<NocOptions::DEFAULT>(
                            noc,
                            in1_mcast_nx_start,
                            in1_mcast_ny_start,
                            in1_mcast_nx_end,
                            in1_mcast_ny_end,
                            in1_num_receivers);
                    }
                }
            }

            // PAIR_SPLIT read-ahead of block kb+1 (see down_pre_issued above): first chance, right
            // after this block's own half went to the partner.
            bool next_pre_issued = false;
            [[maybe_unused]] auto try_issue_next_down = [&]() {
                if (!next_pre_issued && kb + 1 < num_blocks_d &&
                    cb_in1_down_obj.pages_reservable_at_back(2 * d_in1_block_num_tiles)) {
                    const uint32_t next_slot =
                        (in1_block_start == down_ring_base) ? down_ring_base + down_slot_bytes : down_ring_base;
                    if constexpr (split_down) {
                        ++down_seq;
                        down_go_sem.set(down_seq);
                    }
                    issue_down_rows(kb + 1, next_slot);
                    next_pre_issued = true;
                }
            };
            if (down_read_ahead) {
                try_issue_next_down();
            }
            if (kb == 0 && pf_slot < experts_per_chip) {
                prefetch_next_gu(pf_slot);
            }

            if (pair_item) {
                // Pair exchange, steps 4+5. Each of the two senders multicasts ITS column part
                // of every live tile-row (one segment per row; the block is row-major
                // per_core_M x in0_block_w_d) to all 2 * GRID_X cores of the pair's rows,
                // itself included, into the same offsets of cb_in0_down_full. Then it
                // set-multicasts its own valid sem (part 0: act_valid, part 1: pair_valid1)
                // on the same linked path, so the sem lands behind the data.
                constexpr uint32_t kPairDests = 2 * GRID_X_NOC;
                if (is_act_sender) {
                    cb_activated_obj.wait_front(act_tiles_max);
                    pair_ready_sem.wait(kPairDests - 1);
                    pair_ready_sem.set(0);
                    const uint32_t src_l1 = cb_activated_obj.get_read_ptr();
                    const uint32_t dst_l1 = cb_in0_down_full_obj.get_write_ptr();
                    const uint32_t seg_bytes = (gu_hi - gu_lo) * intermed_tile_bytes;
                    if (seg_bytes > 0) {
                        for (uint32_t m = 0; m < per_core_M; ++m) {
                            const uint32_t off = (m * in0_block_w_d + gu_lo) * intermed_tile_bytes;
                            noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                                CoreLocalMem<uint32_t>(src_l1 + off),
                                MulticastEndpoint{},
                                seg_bytes,
                                kPairDests,
                                {.offset_bytes = 0},
                                {.noc_x_start = mrow_first_nx,
                                 .noc_y_start = pair_ny_lo,
                                 .noc_x_end = mrow_last_nx,
                                 .noc_y_end = pair_ny_hi,
                                 .addr = dst_l1 + off},
                                /*linked=*/true);
                        }
                    }
                    noc.async_writes_flushed();
                    Semaphore<>& my_valid = (tail_part == 0) ? act_valid_sem : pair_valid1_sem;
                    my_valid.set(ACT_VALID);
                    my_valid.set_multicast<NocOptions::MCAST_INCL_SRC>(
                        noc, mrow_first_nx, pair_ny_lo, mrow_last_nx, pair_ny_hi, kPairDests);
                    cb_activated_obj.pop_front(act_tiles_max);
                }
                act_valid_sem.wait(ACT_VALID);
                pair_valid1_sem.wait(ACT_VALID);
                if (is_act_sender) {
                    // Own loopback copy: only the ack-wait proves it landed (see below).
                    noc.async_write_barrier();
                }
            } else {
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
                if (mcast_bytes > 0) {
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
                }
                noc.async_writes_flushed();

                act_valid_sem.set(ACT_VALID);
                act_valid_sem.set_multicast<NocOptions::MCAST_INCL_SRC>(
                    noc, mrow_first_nx, mrow_first_ny, mrow_last_nx, mrow_last_ny, GRID_X_NOC);

                cb_activated_obj.pop_front(act_tiles_max);
            }

            // Step 5: receivers wait for both valid sems and push.
            if (!is_act_sender) {
                act_valid_sem.wait(ACT_VALID);
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
            }  // end non-pair activated exchange
            cb_in0_down_full_obj.push_back(d_in0_block_num_tiles);
            ++down_full_blocks;

            // Second chance for the read-ahead, before the partner-valid wait.
            if (down_read_ahead) {
                try_issue_next_down();
            }
            down_pre_issued = next_pre_issued;

            if (!is_in1_sender || pair_split) {
                in1_valid_sem.wait(IN1_VALID);
            }
            cb_in1_down_obj.push_back(d_in1_block_num_tiles);
        }
    }  // end chunk loop
#ifdef RD_DIAG
        DeviceTimestampedData("RD-XSEND", rd_xsend);
        DeviceTimestampedData("RD-WBAR", rd_wbar);
        DeviceTimestampedData("RD-PEER", rd_peer);
        DeviceTimestampedData("RD-XVAL", rd_xval);
        DeviceTimestampedData("RD-WVAL", rd_wval);
        rd_xsend = rd_wbar = rd_peer = rd_xval = rd_wval = 0;
#endif
    }  // end per-local-expert loop

    // The last in-flight Semaphore<>::set_multicast (act_valid / in1_valid)
    // is a posted atomic; without an explicit barrier it can still be in
    // flight at kernel exit, leading to timing-dependent corruption.
    noc.async_atomic_barrier();
}
