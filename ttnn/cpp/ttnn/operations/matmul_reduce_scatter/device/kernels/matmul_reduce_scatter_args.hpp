// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular buffers, semaphores and argument slots of the matmul_reduce_scatter kernels. The kernels and the
// program factory index every slot through these names, so the two sides agree by construction.
//
// Runtime args are split by who shares them:
//   * common runtime args carry what every core of a kernel gets (buffer / semaphore addresses, the per-wave
//     words, which are the same on every core);
//   * per-core runtime args carry what differs by core (origins, ports, schedule words), fixed slots first, the
//     variable-length tails after them at the positions named here.
namespace matmul_reduce_scatter {

// Dense from 0, the values the generic op used; never renumbered.
namespace cb {
constexpr uint32_t ACT_K_WINDOW = 0;     // A K-block windows (resident: the whole core_m x Kt slice)
constexpr uint32_t WEIGHT_K_WINDOW = 1;  // W K-block windows (resident: the whole Kt x core_n slice)
constexpr uint32_t PARTIAL_HOLD = 2;     // fp32 matmul partial slots
constexpr uint32_t FINISHED = 3;         // finished sub-blocks, staging offset 0 (backed by the staging tensor)
constexpr uint32_t ARRIVALS_FWD = 4;     // forward arrival zone slots (backed by the staging tensor)
constexpr uint32_t ARRIVALS_BWD = 5;     // backward arrival zone slots (backed by the staging tensor)
constexpr uint32_t PORT_CHUNKS = 6;      // port cores: gathered chunks waiting for the fabric sender
constexpr uint32_t ZERO = 7;             // one bf16 zero tile (the writer fills it once)
}  // namespace cb

// Program-local semaphores, pushed first (before the multicasts and fabric connections add theirs) on every
// matmul and port core; the factory checks find_available_semaphore_id returns these values.
namespace sem {
constexpr uint32_t CONSUMED_FWD = 0;  // port (fwd) acks a matmul core's finished slot
constexpr uint32_t CONSUMED_BWD = 1;  // port (bwd) acks a matmul core's finished slot
constexpr uint32_t FREED = 2;         // port cores: chunks' worth of arrivals this port's matmul cores consumed
}  // namespace sem

// Kernel order in every chip's ProgramDescriptor (and so in the cached Program).
namespace kernel_index {
enum : uint32_t { READER, COMPUTE, WRITER, PORT_READER, PORT_SENDER, COUNT };
}  // namespace kernel_index

// Compute finish roles (low byte of a compute per-entry word; bit 8 = lag).
namespace role {
enum : uint32_t { COPY, ADD_FWD, ADD_BWD, ADD_BOTH };
}  // namespace role

// Writer handoff kinds (one per entry).
namespace kind {
enum : uint32_t { FWD, BWD, FIN };
}  // namespace kind

// ---- matmul-core reader (NCRISC) ------------------------------------------------------------------------------
// Compile-time args; TensorAccessorArgs of A then W follow from COUNT. Named compile-time args "a", "a2", "b",
// "b2" _ct_offset / _rt_offset are added by the Mcast1D attachments.
namespace reader_ct {
enum : uint32_t {
    CB_ACT_K_WINDOW,
    CB_WEIGHT_K_WINDOW,
    CB_ARRIVALS_FWD,
    CB_ARRIVALS_BWD,
    CORE_M,  // streamed window rows of one wave (win_m)
    CORE_N,  // streamed window cols of one wave (win_n)
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

// Common runtime args; WAVES starts num_waves words: row_off | col_off << 8 | rows << 16 | cols << 24.
namespace reader_common {
enum : uint32_t { A_ADDR, W_ADDR, ARR_FWD_ADDR, ARR_BWD_ADDR, REPORT_CHUNKS, WAVES };
}  // namespace reader_common

// Per-core runtime args; ENTRIES starts num_entries x ENTRY_WORDS words
// ([block | arrival_mask << 16 | lag << 20, thr_fwd | thr_bwd << 16]); the multicast tail follows.
namespace reader_rt {
enum : uint32_t { ROW0, COL0, TOTAL_FWD, TOTAL_BWD, SLOTS_FWD, SLOTS_BWD, REP_FWD_XY, REP_BWD_XY, ENTRIES };
constexpr uint32_t ENTRY_WORDS = 2;
}  // namespace reader_rt

// ---- matmul-core compute ---------------------------------------------------------------------------------------
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

// Common runtime args: num_waves words rows | cols << 8 | sb_h << 16 | sb_w << 24.
namespace compute_common {
enum : uint32_t { WAVES };
}  // namespace compute_common

// Per-core runtime args: num_entries words role | lag << 8.
namespace compute_rt {
enum : uint32_t { ROLES };
}  // namespace compute_rt

// ---- matmul-core writer (BRISC) --------------------------------------------------------------------------------
// Compile-time args; the output's TensorAccessorArgs follow from COUNT.
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

// Common runtime args; WAVES starts num_waves words (as reader_common::WAVES).
namespace writer_common {
enum : uint32_t { OUT_ADDR, WAVES };
}  // namespace writer_common

// Per-core runtime args; KINDS starts num_entries kind words.
namespace writer_rt {
enum : uint32_t { ROW0, COL0, FWD_PORT_XY, BWD_PORT_XY, MAILBOX_SLOT_ADDR, KINDS };
}  // namespace writer_rt

// ---- port cores (reader NCRISC, sender BRISC): one compile-time list for both ---------------------------------
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

// Port reader common runtime args; WAVE_TILES starts num_waves words (tiles one core sends per wave).
namespace port_reader_common {
enum : uint32_t { FINISHED_BASE, MAILBOX_BASE, WAVE_TILES };
}  // namespace port_reader_common

// Port reader per-core runtime args; CORES starts num_cores words (x | y << 8 | half << 16 | k << 20), then
// num_units unit words (entry | half_mask << 8 | h1_first << 10 | slot0 << 16 | slot1 << 24).
namespace port_reader_rt {
enum : uint32_t { CONSUMED_ID, NUM_CORES, NUM_UNITS, CORES };
}  // namespace port_reader_rt

// Port sender common runtime args; WAVE_TILES starts num_waves words.
namespace port_sender_common {
enum : uint32_t { READY_ADDR, CREDIT_ADDR, WAVE_TILES };
}  // namespace port_sender_common

// Port sender per-core runtime args; CORES starts num_cores core words, then num_units unit words, then the
// fabric connection args (tt::tt_fabric::append_fabric_connection_rt_args).
namespace port_sender_rt {
enum : uint32_t {
    ZONE_BASE,  // staging address + this direction's zone offset
    ARRIVAL_ADDR,
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
    CORES
};
}  // namespace port_sender_rt

}  // namespace matmul_reduce_scatter
