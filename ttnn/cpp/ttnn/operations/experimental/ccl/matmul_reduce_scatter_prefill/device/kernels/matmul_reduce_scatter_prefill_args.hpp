// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular buffers, semaphores and argument slots of the matmul_reduce_scatter_prefill kernels. The kernels and the
// program factory index every slot through these names.
//
// Per-core runtime args carry what differs by core (block origins, port coordinates, schedule words); common runtime
// args carry what is the same on every core of a kernel (tensor / staging / mailbox addresses, global semaphore
// addresses, chunk-credit counts). A per-core list has its fixed slots (COUNT of them) first, then a variable tail
// whose start each kernel takes from that COUNT.
namespace matmul_reduce_scatter_prefill {

// Dense from 0, the generic op's ids. The kernels receive them at the compile-time positions below (unchanged).
namespace cb {
constexpr uint32_t ACT_K_WINDOW = 0;     // A K-block windows (resident: the whole core_m x Kt slice)
constexpr uint32_t WEIGHT_K_WINDOW = 1;  // W K-block windows (resident: the whole Kt x core_n slice)
constexpr uint32_t PARTIAL_HOLD = 2;     // fp32 matmul partial slots
constexpr uint32_t FINISHED = 3;         // finished sub-blocks, staging offset 0 (tensor-backed)
constexpr uint32_t ARRIVALS_FWD = 4;     // forward arrival zone slots (tensor-backed, staging + zone_fwd_off)
constexpr uint32_t ARRIVALS_BWD = 5;     // backward arrival zone slots (tensor-backed, staging + zone_bwd_off)
constexpr uint32_t PORT_CHUNKS = 6;      // port cores: gathered chunks waiting for the fabric sender
constexpr uint32_t ZERO = 7;             // one bf16 zero tile (the writer fills it once)
}  // namespace cb

// Program-local semaphores, pushed first (before any multicast attaches its own), in this order. The factory checks
// find_available_semaphore_id returns exactly these ids.
namespace sem {
constexpr uint32_t CONSUMED_FWD = 0;  // matmul cores: acks of the fwd port
constexpr uint32_t CONSUMED_BWD = 1;  // matmul cores: acks of the bwd port
constexpr uint32_t FREED = 2;         // port cores: chunk-consumption reports of the matmul cores
}  // namespace sem

// Kernels in ProgramDescriptor::kernels order (= kernel handle in the cached Program).
namespace kernel_index {
enum : uint32_t { READER, COMPUTE, WRITER, PORT_READER, PORT_SENDER, COUNT };
}  // namespace kernel_index

// Compute finish roles (low byte of a compute roles word; bit LAG_SHIFT = lag).
namespace role {
constexpr uint32_t COPY = 0;
constexpr uint32_t ADD_FWD = 1;
constexpr uint32_t ADD_BWD = 2;
constexpr uint32_t ADD_BOTH = 3;
constexpr uint32_t LAG_SHIFT = 8;
}  // namespace role

// Writer handoff kinds (one word per entry).
namespace kind {
constexpr uint32_t FWD = 0;
constexpr uint32_t BWD = 1;
constexpr uint32_t FIN = 2;
}  // namespace kind

// ---------------------------------------------------------------------------------------------------------------
// Matmul reader (NCRISC, NoC0)
// ---------------------------------------------------------------------------------------------------------------
// Compile-time args; TensorAccessorArgs of A then W follow from COUNT, then the multicast families' args (named
// a_/a2_/b_/b2_ ct/rt offsets).
namespace reader_ct {
enum : uint32_t {
    CB_ACT_K_WINDOW,
    CB_WEIGHT_K_WINDOW,
    CB_ARRIVALS_FWD,
    CB_ARRIVALS_BWD,
    CORE_M,  // win_m
    CORE_N,  // win_n
    KB,
    NUM_K_BLOCKS,
    KT,
    NT,
    A_ROW_PER_BLOCK,
    W_COL_PER_BLOCK,
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
namespace reader_common {
enum : uint32_t { A_ADDR, W_ADDR, ARR_FWD_ADDR, ARR_BWD_ADDR, REPORT_CHUNKS, COUNT };
}  // namespace reader_common
// Per-core; tail from COUNT: num_waves wave words (row_off | col_off << 8 | rows << 16 | cols << 24), then per entry
// ENTRY_WORDS words [block | mask << 16 | lag << 20, thr_fwd | thr_bwd << 16], then the multicast args.
namespace reader_rt {
enum : uint32_t { ROW0, COL0, TOTAL_FWD, TOTAL_BWD, SLOTS_FWD, SLOTS_BWD, REP_FWD_XY, REP_BWD_XY, COUNT };
constexpr uint32_t ENTRY_WORDS = 2;
}  // namespace reader_rt

// ---------------------------------------------------------------------------------------------------------------
// Compute (TRISCs)
// ---------------------------------------------------------------------------------------------------------------
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
// Per-core: no fixed slots. num_waves words (rows | cols << 8 | sb_h << 16 | sb_w << 24), then per entry
// role | lag << role::LAG_SHIFT.

// ---------------------------------------------------------------------------------------------------------------
// Matmul writer (BRISC, NoC1)
// ---------------------------------------------------------------------------------------------------------------
// Compile-time args; TensorAccessorArgs of the output follow from COUNT.
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
namespace writer_common {
enum : uint32_t { OUT_ADDR, MAILBOX_ADDR, COUNT };
}  // namespace writer_common
// Per-core; MAILBOX_SLOT_OFFSET = k * MAILBOX_STRIDE (the kernel adds MAILBOX_ADDR). Tail from COUNT: num_waves wave
// words, then one kind word per entry.
namespace writer_rt {
enum : uint32_t { ROW0, COL0, FWD_PORT_XY, BWD_PORT_XY, MAILBOX_SLOT_OFFSET, COUNT };
}  // namespace writer_rt

// ---------------------------------------------------------------------------------------------------------------
// Port reader (NCRISC, NoC0) and port sender (BRISC, NoC1): one shared compile-time list
// ---------------------------------------------------------------------------------------------------------------
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
namespace port_reader_common {
enum : uint32_t { FINISHED_BASE, MAILBOX_BASE, COUNT };
}  // namespace port_reader_common
// Per-core; tail from COUNT: num_waves wave-tile counts, NUM_CORES core words (x | y << 8 | half << 16 | k << 20),
// NUM_UNITS unit words (entry | half_mask << 8 | h1_first << 10 | slot0 << 16 | slot1 << 24).
namespace port_reader_rt {
enum : uint32_t { CONSUMED_ID, NUM_CORES, NUM_UNITS, COUNT };
}  // namespace port_reader_rt
namespace port_sender_common {
enum : uint32_t { STAGING_ADDR, READY_ADDR, CREDIT_ADDR, COUNT };
}  // namespace port_sender_common
// Per-core; ZONE_OFFSET is added to STAGING_ADDR (the downstream zone of this direction), ARRIVAL_ADDR is the global
// arrival semaphore of this direction. Tail from COUNT: wave tiles, core words, unit words, then the fabric
// connection args.
namespace port_sender_rt {
enum : uint32_t {
    ZONE_OFFSET,
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
    COUNT
};
}  // namespace port_sender_rt

}  // namespace matmul_reduce_scatter_prefill
