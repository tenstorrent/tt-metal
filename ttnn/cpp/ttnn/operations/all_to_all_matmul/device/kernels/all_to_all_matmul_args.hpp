// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular buffers, kernels and argument slots of the all_to_all_matmul kernels. The kernels and the program
// factory index every slot through these names.
//
// Every call-varying value is a common runtime arg: the tensor addresses and the landing scratch base. Per-core
// args hold only what the plan fixes (slot work, entry lists, routes, semaphore addresses of the per-mesh global
// semaphores, slot OFFSETS from the landing base), so a cache hit writes common args only.
namespace all_to_all_matmul {

// Dense from 0, the generic op's ids.
namespace cb {
constexpr uint32_t A_LANDED = 0;      // backed by the landing scratch (height-sharded L1 over the compute cores)
constexpr uint32_t WEIGHT = 1;        // weight blocks
constexpr uint32_t OUT = 2;           // output row groups
constexpr uint32_t PORT_CHUNKS = 3;   // port cores: one page = one fabric packet
constexpr uint32_t RETILE_STAGE = 4;  // regime R1' only (Mq % TILE_HEIGHT != 0): staging of one image's source rows
constexpr uint32_t ROPE_STAGE = 5;    // q-stem mode only: compute cores' local rope copy (DRAM -> L1 -> rope_seq)
}  // namespace cb

// Position of each kernel in ProgramDescriptor::kernels (and in the cached Program).
namespace kernel_index {
enum : uint32_t { READER, COMPUTE, WRITER, PORT_READER, PORT_SENDER, COUNT };
}  // namespace kernel_index

// Input tile addressing (both readers): tile k of tile row r of source-local head j, image b is page
//   b * IN_BATCH_STRIDE + j * IN_HEAD_STRIDE + r * IN_ROW_STRIDE + k
// (B, Hl, S, K) input: Hl*St*Kt, St*Kt, Kt. q-stem input (1, 1, S, Hl*(K+R)): -, (K+R)/32, Hl*(K+R)/32; the head's rope
// tiles follow its nope tiles (k = Kt + rt, rt < RT). RT = R/32, 0 = q-stem mode off.

// Packed per-slot word "nbw": n-blocks [nb0, nb1) and rows [m0, m1) of the head, 8 bits each.
namespace nbw_bits {
constexpr uint32_t NB0_SHIFT = 0;
constexpr uint32_t NB1_SHIFT = 8;
constexpr uint32_t M0_SHIFT = 16;
constexpr uint32_t M1_SHIFT = 24;
constexpr uint32_t FIELD_MASK = 0xFF;
}  // namespace nbw_bits

// ---- compute-core reader (RISCV_1) ----
namespace reader_ct {
enum : uint32_t {
    CB_A_LANDED,
    CB_WEIGHT,
    KT,
    NT,
    BLOCK_N,
    NUM_N_BLOCKS,
    B,
    MQ_T,
    ST,
    HL,
    IN_TILE_BYTES,
    W_TILE_BYTES,
    MQ,
    CB_RETILE_STAGE,
    WEIGHT_CHUNK_TILES,
    W_PREFETCH_BLOCKS,
    IN_BATCH_STRIDE,
    IN_HEAD_STRIDE,
    IN_ROW_STRIDE,
    RT,
    H,
    CB_ROPE_STAGE,
    ROPE_STAGE_TILES,
    COUNT  // then TensorAccessorArgs(input), TensorAccessorArgs(weight), TensorAccessorArgs(rope_seq; RT = 0: input)
};
}  // namespace reader_ct
namespace reader_common {
// ARRIVAL_BASE + s: arrival counter address of slot s (slots_per_core_max entries). ROPE_ADDR: rope_seq (0: off).
enum : uint32_t { IN_ADDR, W_ADDR, MY_Q, LANDING_ADDR, ROPE_ADDR, ARRIVAL_BASE };
}  // namespace reader_common
namespace reader_rt {
enum : uint32_t { N_SLOTS, SLOTS_BASE };
}  // namespace reader_rt
// One record per slot at SLOTS_BASE + s * COUNT.
// WORD = h | wait << WAIT_SHIFT | prefetch_blocks << PRE_SHIFT. FWD_SLOT_OFFSET is the helper slot's byte offset
// from the landing base (the kernel adds reader_common::LANDING_ADDR).
namespace reader_slot {
enum : uint32_t { WORD, NBW, FWD, FWD_X, FWD_Y, FWD_SLOT_OFFSET, FWD_ARRIVAL_ADDR, PACE, COUNT };
constexpr uint32_t HEAD_MASK = 0xFFFF;
constexpr uint32_t WAIT_SHIFT = 16;
constexpr uint32_t WAIT_MASK = 0xFF;
constexpr uint32_t PRE_SHIFT = 24;
}  // namespace reader_slot

// ---- compute (TRISC) ----
namespace compute_ct {
enum : uint32_t {
    CB_A_LANDED,
    CB_WEIGHT,
    CB_OUT,
    BLOCK_M,
    KT,
    BLOCK_N,
    NUM_N_BLOCKS,
    OUT_SUBBLOCK_H,
    OUT_SUBBLOCK_W,
    COUNT
};
}  // namespace compute_ct
namespace compute_rt {
enum : uint32_t { N_SLOTS, NBW_BASE };  // NBW_BASE + s: slot s's nbw
}  // namespace compute_rt

// ---- compute-core writer (RISCV_0) ----
namespace writer_ct {
enum : uint32_t {
    CB_OUT,
    TILE_BYTES,
    BLOCK_M,
    MQ_T,
    NT,
    BLOCK_N,
    NUM_N_BLOCKS,
    OUT_SUBBLOCK_H,
    H,
    OUT_CHUNK_TILES,
    COUNT  // then TensorAccessorArgs(output)
};
}  // namespace writer_ct
namespace writer_common {
enum : uint32_t { OUT_ADDR, COUNT };
}  // namespace writer_common
namespace writer_rt {
enum : uint32_t { N_SLOTS, SLOTS_BASE };
}  // namespace writer_rt
namespace writer_slot {
enum : uint32_t { HEAD, NBW, COUNT };
}  // namespace writer_slot

// ---- port reader (RISCV_1) ----
namespace port_reader_ct {
enum : uint32_t {
    CB_PORT_CHUNKS,
    TILE_BYTES,
    PACKET_TILES,
    BATCH_CHUNKS,
    B,
    MQ_T,
    KT,
    ST,
    HL,
    MQ,
    CB_RETILE_STAGE,
    PACK,
    WINDOW,
    IN_BATCH_STRIDE,
    IN_HEAD_STRIDE,
    IN_ROW_STRIDE,
    RT,
    ROPE_PACKET_TILES,
    COUNT  // then TensorAccessorArgs(input)
};
}  // namespace port_reader_ct
namespace port_reader_common {
enum : uint32_t { IN_ADDR, COUNT };
}  // namespace port_reader_common
// ENTRIES_BASE + e: entry word j | t << ENTRY_DEST_SHIFT. Then N_ROPE and N_ROPE rope entry words (same encoding,
// grouped by destination chip; q-stem mode only, else N_ROPE = 0).
namespace port_reader_rt {
enum : uint32_t { N_ENTRIES, ENTRIES_BASE };
constexpr uint32_t ENTRY_HEAD_MASK = 0xFFFF;
constexpr uint32_t ENTRY_DEST_SHIFT = 16;
}  // namespace port_reader_rt

// ---- port sender (RISCV_0) ----
namespace port_sender_ct {
enum : uint32_t {
    CB_PORT_CHUNKS,
    TILE_BYTES,
    PACKET_TILES,
    BATCH_CHUNKS,
    SLOT_TILES,
    MAX_HEADERS,
    PACK,
    FENCE_PER_DEST,
    RT,
    ROPE_PACKET_TILES,
    B,
    H,
    MQ_T,
    COUNT  // then TensorAccessorArgs(rope_seq; RT = 0: a placeholder)
};
}  // namespace port_sender_ct
namespace port_sender_common {
// ROPE_ADDR: rope_seq (0: off); ROPE_SEM_ADDR: the rope arrival counter (port cores; 0: off).
enum : uint32_t { LANDING_ADDR, ROPE_ADDR, ROPE_SEM_ADDR, COUNT };
}  // namespace port_sender_common
// Per-core layout (variable length):
//   HAS_CONN, N_DEST, HDRS_PER_DEST, ROPE_WAIT,
//   N_DEST x dest record, N_READY_OUT, N_READY_OUT x ready record, N_SUBS, N_SUBS x sub record,
//   N_ROPE, N_ROPE x rope word,
//   fabric connection args (appended last, only when HAS_CONN).
// ROPE_WAIT: rope groups that land on this chip (this core holds the chip's rope arrival counter; 0: none).
namespace port_sender_rt {
enum : uint32_t { HAS_CONN, N_DEST, HDRS_PER_DEST, ROPE_WAIT, DESTS_BASE };
}  // namespace port_sender_rt
// ROPE_X / ROPE_Y: the destination chip's rope arrival counter core (virtual).
namespace port_sender_dest {
enum : uint32_t { CHIP_ID, MESH_ID, READY_ADDR, ROPE_X, ROPE_Y, COUNT };  // READY_ADDR 0: no data to that chip
}  // namespace port_sender_dest
// Rope word: global head h | destination index << DEST_SHIFT, grouped by destination (one arrival increment per group).
namespace port_sender_rope {
constexpr uint32_t HEAD_MASK = 0xFFFF;
constexpr uint32_t DEST_SHIFT = 16;
}  // namespace port_sender_rope
namespace port_sender_ready {
enum : uint32_t { DEST_INDEX, X, Y, ADDR, COUNT };
}  // namespace port_sender_ready
// WORD0 = destination index | tiles << SUB_TILES_SHIFT. SLOT_OFFSET is the byte offset from the landing base (the
// kernel adds port_sender_common::LANDING_ADDR).
namespace port_sender_sub {
enum : uint32_t { WORD0, X, Y, SLOT_OFFSET, ARRIVAL_ADDR, COUNT };
constexpr uint32_t DEST_MASK = 0xFFFF;
constexpr uint32_t TILES_SHIFT = 16;
}  // namespace port_sender_sub

}  // namespace all_to_all_matmul
