// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular buffers, semaphores and argument slots of the distributed_rms_norm kernels. The kernels and the
// program factory index every slot through these names.
//
// Per-core runtime args carry what differs by core (the work split, the exchange wiring); common runtime args
// carry what every core of a kernel shares (buffer addresses, the landing layout, the scalars).
namespace distributed_rms_norm {

// The ids the generic op used, unchanged.
namespace cb {
constexpr uint32_t X_TILES = 1;
constexpr uint32_t REDUCE_SCALER = 2;
constexpr uint32_t SUMSQ_PARTIAL = 3;
constexpr uint32_t INV_RMS = 4;
constexpr uint32_t GAMMA_TILES = 6;
constexpr uint32_t X_GAMMA = 7;
constexpr uint32_t PARTIAL_COL = 9;
constexpr uint32_t GATHER = 10;
constexpr uint32_t RMS_ROOT = 11;
constexpr uint32_t SUMSQ_LAST = 12;
constexpr uint32_t SUMSQ_ACC = 13;
constexpr uint32_t OUTPUT = 16;
constexpr uint32_t DEV_PARTIAL = 17;
constexpr uint32_t STATS_GATHER = 18;
constexpr uint32_t STATS_RECORD = 19;
constexpr uint32_t MAX_ID = STATS_RECORD;
}  // namespace cb

// Program-local semaphore ids (declared on every compute and port core before the fabric connections take theirs).
// GATHER_BASE + k: member partial arrivals of wave k of a block. The output-split semaphores follow at
// GATHER_BASE + nwpb + {0, 1, 2} (compute cores only) and reach the kernels as compile-time args.
namespace sem {
constexpr uint32_t INV = 0;
constexpr uint32_t OWN_STATS = 1;
constexpr uint32_t LANDED_FWD = 2;
constexpr uint32_t LANDED_BWD = 3;
constexpr uint32_t OWN_COUNTED = 4;
constexpr uint32_t GATHER_BASE = 5;
constexpr uint32_t PORT_FREED = 0;  // on a port core: INV's id, unused there
constexpr uint32_t MAX_WAVES_PER_BLOCK = 8;
constexpr uint32_t NUM_OUT_SPLIT = 3;  // ready, done, addr
}  // namespace sem

// Exchange directions of a port.
namespace dir {
constexpr uint32_t FWD = 0;
constexpr uint32_t BWD = 1;
}  // namespace dir

constexpr uint32_t RECORD_WORDS = 32;  // one fp32 word per row of a tile-row
constexpr uint32_t RECORD_BYTES = RECORD_WORDS * 4;
constexpr uint32_t MAX_HEADERS = 8;  // packet headers per port (one per destination chip)

// ---- reader (compute cores, RISCV_1) ----
// Compile-time: these slots, then TensorAccessorArgs(input), TensorAccessorArgs(output), then one word
// FIN_COMPACT (at output_args.next_compile_time_args_offset()).
namespace reader_ct {
enum : uint32_t {
    BLOCK_ROWS,
    W_CHUNK_TILES,
    NUM_W_CHUNKS,
    WT,
    WS,
    X_DEPTH,
    SHARDED,
    COMBINE,
    G_W,
    WAVE_ROWS,
    WAVES_PER_BLOCK,
    GROUP_DEVICES,
    T_IN,
    T_I,
    W_VALID_LAST,
    ES_ZERO,
    MASKED,
    NUM_WAVES,
    INTERM_FP32,
    READ_WAVE_GATE,
    LANDING_WAVES,
    OUT_SPLIT_X,
    NOC0_BANKS_WEST,
    NOC0_BANKS_EAST,
    SEM_OUT_READY,
    SEM_OUT_DONE,
    SEM_OUT_ADDR,
    T_OUT,
    COUNT
};
}  // namespace reader_ct

// Per-core fixed slots, in the generic op's order with its common slots taken out. After COUNT:
//   [esw_rt::COUNT words when DRMS_ESW_OFFSETS is defined], then the member runs: n_runs,
//   then MEMBER_RUN_WORDS per run (lo, hi, count), zero-padded to G_W - 1 runs on every core.
namespace reader_rt {
enum : uint32_t {
    START_TILE_ROW,
    ROWS,
    W0,
    V,
    IS_ROOT,
    MY_INDEX,
    ROOT_X,
    ROOT_Y,
    REC_OFF,
    LINK,  // this core's link (0 on a member): its row of the common per-link block
    COUNT
};
}  // namespace reader_rt
namespace esw_rt {
enum : uint32_t { WAVE_OFF, READ_START, COUNT };
}  // namespace esw_rt
constexpr uint32_t MEMBER_RUN_WORDS = 3;

// Common: these slots (the chip's exchange wiring is the same on every core: its entry counts, its own slot, its
// sending ports), then after COUNT the N_IN_FWD + N_IN_BWD entry slot ids (entry i of a direction, landing order),
// then one row per link of reader_link_row::COUNT + G words.
namespace reader_common {
enum : uint32_t { INPUT_ADDR, LANDING_ADDR, SEG_STRIDE, OUTPUT_ADDR, N_IN_FWD, N_IN_BWD, MY_SLOT, N_SEND, COUNT };
}  // namespace reader_common
// One common row per link: the link's N_SEND packed sending-port coords (0 when absent), the bounded landing's
// N_FREE packed ports a root reports collected waves to (0 when absent; read by roots only), then after COUNT the G
// packed slot-port coords (where slot q lands on this chip for that link's ports).
namespace reader_link_row {
enum : uint32_t { SEND_PORT_0, SEND_PORT_1, N_FREE, FREE_PORT_0, FREE_PORT_1, COUNT };
}  // namespace reader_link_row

// ---- writer (compute cores, RISCV_0) ----
// Compile-time: these slots, then TensorAccessorArgs(output), TensorAccessorArgs(gamma or none).
namespace writer_ct {
enum : uint32_t {
    HAS_GAMMA,
    GAMMA_IS_TILE,
    BLOCK_ROWS,
    W_CHUNK_TILES,
    NUM_W_CHUNKS,
    WT,
    WS,
    OUT_NATIVE,
    T_OUT,
    T_G,
    ES_G,
    WP,
    GAMMA_RESIDENT,
    OUT_SPLIT_X,
    NOC0_BANKS_WEST,
    NOC0_BANKS_EAST,
    SEM_OUT_READY,
    SEM_OUT_DONE,
    SEM_OUT_ADDR,
    COUNT
};
}  // namespace writer_ct
namespace writer_rt {
enum : uint32_t { START_TILE_ROW, ROWS, W0, V, COUNT };
}  // namespace writer_rt
namespace writer_common {
enum : uint32_t { OUTPUT_ADDR, GAMMA_ADDR, COUNT };
}  // namespace writer_common

// ---- compute ----
namespace compute_ct {
enum : uint32_t {
    HAS_GAMMA,
    BLOCK_ROWS,
    W_CHUNK_TILES,
    NUM_W_CHUNKS,
    WS,
    OUT_NATIVE,
    COMBINE,
    MASKED,
    G_W,
    SCALE_BLOCK_TILES,
    WAVE_ROWS,
    WAVES_PER_BLOCK,
    GROUP_DEVICES,
    GAMMA_RESIDENT,
    FIN_COMPACT,
    COUNT
};
}  // namespace compute_ct
namespace compute_rt {
enum : uint32_t { ROWS, SLICE_VALID, IS_ROOT, MASKED_CORE, COUNT };
}  // namespace compute_rt
namespace compute_common {
enum : uint32_t { INV_W_BITS, EPS_BITS, COUNT };
}  // namespace compute_common

// ---- port (port cores, RISCV_0 / NOC_1) ----
namespace port_ct {
enum : uint32_t {
    GROUP_DEVICES,
    NUM_WAVES,
    PAYLOAD,
    NUM_HEADERS,
    SEM_OWN_STATS,
    SEM_OWN_COUNTED,
    OWN_AHEAD,
    LANDING_WAVES,
    COUNT
};
}  // namespace port_ct

// Per-core fixed slots. After COUNT: N_IN arrival counter addresses, N_DEST * PORT_DEST_WORDS destination
// records, N_SEND * PORT_SEND_WORDS send items, N_READY_OUT * PORT_READY_WORDS ready records, NUM_WAVES segment
// byte counts, max(N_ROOTS, N_NOTIFY) packed root coords, then the fabric connection args (when HAS_CONN).
namespace port_rt {
enum : uint32_t {
    N_DEST,
    N_SEND,
    N_IN,
    N_ROOTS,
    N_READY_OUT,
    N_READY_WAIT,
    MY_X,
    MY_Y,
    N_NOTIFY,
    HAS_CONN,
    N_COUNTED,
    NOTIFY_QUANTUM,
    LANDED_SEM_ID,
    N_CREDIT_OUT,
    CREDIT_THR,
    RELAY_DONE_TO,
    N_CREDIT_IN,
    COUNT
};
}  // namespace port_rt
constexpr uint32_t PORT_DEST_WORDS = 4;   // chip id, mesh id, hops, mcast code
constexpr uint32_t PORT_SEND_WORDS = 4;   // slot q, header, destination arrival address, wait
constexpr uint32_t PORT_READY_WORDS = 3;  // header, target port x, target port y

namespace port_common {
enum : uint32_t { LANDING_ADDR, SEG_STRIDE, READY_ADDR, COUNT };
}  // namespace port_common

// Kernel order in every program's ProgramDescriptor::kernels.
namespace kernel_index {
enum : uint32_t { READER, WRITER, COMPUTE, PORT, COUNT };
}  // namespace kernel_index

}  // namespace distributed_rms_norm
