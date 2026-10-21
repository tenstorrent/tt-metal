// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular buffers, semaphores and argument slots of the matmul_reduce_scatter kernels. The kernels and the
// program factory index every slot through these names.
//
// Per chip, one program with two core roles:
//   matmul cores: READER (NCRISC), COMPUTE (TRISCs), WRITER (BRISC)
//   port cores:   PORT_READER (NCRISC), PORT_SENDER (BRISC), one port per (direction, link)
//
// Runtime args are split by how often they change:
//   * common runtime args carry what is the same on every core of a kernel: the buffer addresses that change
//     from call to call (A, W, the output, the staging tensor) and the per-wave words;
//   * per-core runtime args carry what the core's position in the grid / schedule fixes.
// A list with a variable-length part puts its fixed slots first (the enums below) and the tail after them,
// in the order documented next to each enum.
namespace matmul_reduce_scatter {

// Dense from 0, the values the kernels used when the operation was a Python generic_op.
namespace cb {
constexpr uint32_t ACT_K_WINDOW = 0;     // A K-block windows (resident: the whole core_m x Kt slice)
constexpr uint32_t WEIGHT_K_WINDOW = 1;  // W K-block windows (resident: the whole Kt x core_n slice)
constexpr uint32_t PARTIAL_HOLD = 2;     // fp32 matmul partial slots
constexpr uint32_t FINISHED = 3;         // finished sub-blocks (bf16), staging offset 0; read remotely by the ports
constexpr uint32_t ARRIVALS_FWD = 4;     // forward arrival zone slots (staging, written over the fabric)
constexpr uint32_t ARRIVALS_BWD = 5;     // backward arrival zone slots (staging)
constexpr uint32_t PORT_CHUNKS = 6;      // port cores: gathered chunks waiting for the fabric sender
constexpr uint32_t ZERO = 7;             // one bf16 zero tile (the writer fills it once)
}  // namespace cb

// Program-local semaphore ids (on every core of the program, pushed first, in this order).
namespace sem {
constexpr uint32_t CONSUMED_FWD = 0;  // matmul core: acks of the fwd port for its handoffs
constexpr uint32_t CONSUMED_BWD = 1;  // matmul core: acks of the bwd port
constexpr uint32_t FREED = 2;         // port core: chunks' worth of arrivals its matmul cores consumed
}  // namespace sem

// Compute finish roles (low byte of a compute per-core entry word; bit 8 = lag).
namespace role {
constexpr uint32_t COPY = 0;
constexpr uint32_t ADD_FWD = 1;
constexpr uint32_t ADD_BWD = 2;
constexpr uint32_t ADD_BOTH = 3;
}  // namespace role

// Writer handoff kinds (one per entry in the writer's per-core tail).
namespace kind {
constexpr uint32_t FWD = 0;
constexpr uint32_t BWD = 1;
constexpr uint32_t FIN = 2;
}  // namespace kind

// ---------------------------------------------------------------------------------------------------------
// Matmul reader (NCRISC, NoC0)
// ---------------------------------------------------------------------------------------------------------
// Compile-time args; TensorAccessorArgs of A then W follow at COUNT; the multicast families "a", "a2", "b",
// "b2" are attached after them (named args <prefix>_ct_offset / <prefix>_rt_offset).
namespace reader_ct {
enum : uint32_t {
    CB_ACT_K_WINDOW,
    CB_WEIGHT_K_WINDOW,
    CB_ARRIVALS_FWD,
    CB_ARRIVALS_BWD,
    CORE_M,  // streamed A window rows (one wave)
    CORE_N,  // streamed W window cols (one wave)
    KB,
    NUM_K_BLOCKS,
    KT,
    NT,
    A_ROW_PER_BLOCK,  // scatter -2: block_m, else 0
    W_COL_PER_BLOCK,  // scatter -1: block_n, else 0
    RESIDENT_A,
    RESIDENT_W,
    P,
    NUM_ENTRIES,
    A_TILE_BYTES,
    W_TILE_BYTES,
    A_CAP_WINDOWS,
    W_CAP_WINDOWS,
    A_SPLIT,
    W_SPLIT,
    PREFETCH_STEPS,
    LOOKAHEAD_AFTER_BARRIER,
    A_READ_NOC,
    W_READ_NOC,
    NUM_WAVES,
    NUM_CHUNKS,
    N_INNER,
    M_OUTER,
    CHUNK_M,
    CHUNK_N,
    BLOCK_M,
    BLOCK_N,
    COUNT
};
}  // namespace reader_ct

// Common runtime args; then NUM_WAVES wave words (row_off | col_off << 8 | rows << 16 | cols << 24).
// The arrival counters are persistent GlobalSemaphores (stable for the mesh's life).
namespace reader_common {
enum : uint32_t { A_ADDR, W_ADDR, ARR_FWD_ADDR, ARR_BWD_ADDR, REPORT_CHUNKS, COUNT };
}  // namespace reader_common

// Per-core runtime args; then NUM_ENTRIES x ENTRY_WORDS entry words
// [block | arrival_mask << 16 | lag << 20, thr_fwd | thr_bwd << 16]; then the multicast tail.
namespace reader_rt {
enum : uint32_t {
    ROW0,       // gm * core_m (chunk-local)
    COL0,       // gn * core_n (chunk-local)
    TOTAL_FWD,  // arrival increments my counters receive per chunk
    TOTAL_BWD,
    SLOTS_FWD,  // arrival slots I consume per chunk
    SLOTS_BWD,
    REP_FWD_XY,  // port counting my consumed fwd / bwd chunks (x | y << 16), 0 when none
    REP_BWD_XY,
    COUNT
};
constexpr uint32_t ENTRY_WORDS = 2;
}  // namespace reader_rt

// ---------------------------------------------------------------------------------------------------------
// Matmul compute (TRISCs)
// ---------------------------------------------------------------------------------------------------------
namespace compute_ct {
enum : uint32_t {
    CB_ACT_K_WINDOW,
    CB_WEIGHT_K_WINDOW,
    CB_PARTIAL_HOLD,
    CB_FINISHED,
    CB_ARRIVALS_FWD,
    CB_ARRIVALS_BWD,
    WIN_N,
    KB,
    NUM_K_BLOCKS,
    P,
    NUM_ENTRIES,
    NUM_WAVES,
    SUPER_BLOCK,
    CB_ZERO,
    NUM_CHUNKS,
    N_INNER,
    RESIDENT,
    COUNT
};
}  // namespace compute_ct

// Common runtime args: NUM_WAVES words rows | cols << 8 | sb_h << 16 | sb_w << 24, starting at WAVES.
namespace compute_common {
enum : uint32_t { WAVES, COUNT = WAVES };
}  // namespace compute_common

// Per-core runtime args: NUM_ENTRIES words role | lag << 8, starting at ROLES.
namespace compute_rt {
enum : uint32_t { ROLES, COUNT = ROLES };
}  // namespace compute_rt

// ---------------------------------------------------------------------------------------------------------
// Matmul writer (BRISC, NoC1)
// ---------------------------------------------------------------------------------------------------------
// Compile-time args; TensorAccessorArgs of the output follow at COUNT.
namespace writer_ct {
enum : uint32_t {
    CB_FINISHED,
    P,
    NUM_WAVES,
    OUT_N_TILES,
    DEPTH_HANDOFF,
    TILE_BYTES,
    NUM_ENTRIES,
    CB_ZERO,
    NUM_CHUNKS,
    N_INNER,
    M_OUTER,
    CHUNK_M,
    CHUNK_N,
    BLOCK_M,
    BLOCK_N,
    COUNT
};
}  // namespace writer_ct

// Common runtime args; then NUM_WAVES wave words (as the reader's).
namespace writer_common {
enum : uint32_t { OUT_ADDR, COUNT };
}  // namespace writer_common

// Per-core runtime args; then NUM_ENTRIES handoff kinds (kind::*).
// MAILBOX_SLOT_ADDR: the persistent mailbox tensor's address + k * MAILBOX_STRIDE.
namespace writer_rt {
enum : uint32_t { ROW0, COL0, FWD_PORT_XY, BWD_PORT_XY, MAILBOX_SLOT_ADDR, COUNT };
}  // namespace writer_rt

// ---------------------------------------------------------------------------------------------------------
// Port reader (NCRISC, NoC0) and port sender (BRISC, NoC1): one compile-time list for both
// ---------------------------------------------------------------------------------------------------------
namespace port_ct {
enum : uint32_t {
    CB_PORT_CHUNKS,
    P,
    CHUNK_TILES,
    GROUP,
    TILE_BYTES,
    DEPTH_HANDOFF,
    MAILBOX_STRIDE,
    MAX_CORES,
    NUM_WAVES,
    INC_EVERY_WAVE,
    NUM_CHUNKS,
    ENTRIES_PER_CHUNK,
    ZONE_SETS,
    COUNT
};
}  // namespace port_ct

// Port reader common runtime args; then NUM_WAVES wave tile counts.
// FINISHED_BASE is the staging tensor (cb_finished slot 0 on every core); MAILBOX_BASE the persistent mailbox.
namespace port_reader_common {
enum : uint32_t { FINISHED_BASE, MAILBOX_BASE, COUNT };
}  // namespace port_reader_common

// Port reader per-core runtime args; then NUM_CORES core words (x | y << 8 | half << 16 | k << 20), then
// NUM_UNITS unit words (entry | half_mask << 8 | h1_first << 10 | slot0 << 16 | slot1 << 24).
namespace port_reader_rt {
enum : uint32_t { CONSUMED_ID, NUM_CORES, NUM_UNITS, COUNT };
}  // namespace port_reader_rt

// Port sender common runtime args; then NUM_WAVES wave tile counts.
// STAGING_ADDR is the staging tensor; the downstream zone is STAGING_ADDR + ZONE_OFFSET (per core).
// READY_ADDR / CREDIT_ADDR are persistent GlobalSemaphores.
namespace port_sender_common {
enum : uint32_t { STAGING_ADDR, READY_ADDR, CREDIT_ADDR, COUNT };
}  // namespace port_sender_common

// Port sender per-core runtime args; then NUM_CORES core words, NUM_UNITS unit words (as the port reader's),
// then the fabric connection args (tt::tt_fabric::append_fabric_connection_rt_args).
namespace port_sender_rt {
enum : uint32_t {
    ZONE_OFFSET,   // byte offset of this direction's downstream zone in the staging tensor
    ARRIVAL_ADDR,  // global arr_d on the downstream matmul cores
    SEND_READY,
    READY_X,
    READY_Y,
    DST_MESH_ID,
    DST_CHIP_ID,
    NUM_CORES,
    NUM_UNITS,
    WAIT_CREDITS,
    EXPECT,
    FWD_TOTAL,
    RING_SLOTS,
    DOWN_PER_CHUNK0,
    DOWN_PER_CHUNK1,
    COUNT
};
}  // namespace port_sender_rt

}  // namespace matmul_reduce_scatter
