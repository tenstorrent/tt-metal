// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The multi-core WH SDPA op (one chip): the contract between the compute kernel (wh_sdpa_op.cpp, the core loop of
// wh_sdpa_core.h on the three TRISCs), the K/V feed (wh_sdpa_feed.cpp, NCRISC of a group's reader cores) and the I/O
// kernel (wh_sdpa_io.cpp, BRISC of every core: Q in, O out, consumption credits to the readers). Adapted from
// tt-lab-private h3-wh src/wh_sdpa.h to tt-metal tensors (tile layout in DRAM) and an L1 scratch tensor.
//
// Work: an item is one 64-query block of one head against every key block of the sequence. The cores form feed groups
// (a rectangle); a group runs rounds, each round one head, and each core of the group takes one query block of that
// head in it (or none). The group's cores share the K/V stream: key block g of the group's stream (counting over
// every round) lands in L1 buffer g % bufs of every core of the group, read from DRAM by reader g % readers and
// multicast by it; each core returns each block to its reader once TRISC0 has unpacked it, and a reader refills a
// buffer once the whole group has consumed it.
//
// DRAM: Q, O [heads][rows][128] and K [heads][keys][128] in tile layout; Vt holds V with every 32 x 32 tile
// transposed in place ([heads][keys][128] tiles, tile (i, j) = V tile (i, j) transposed), so its faces are V^T faces.
#pragma once

#include <stdint.h>

#define WHS_ROWS 64u      // a query or key block
#define WHS_FACE 2048u    // an L1 operand chunk, [64 rows][16] BF16
#define WHS_BLOCK 32768u  // a key block in L1: K's 8 d slices [64 keys][16 d], then V^T's 8 chunks
#define WHS_TILE 2048u    // a tt-metal BF16 tile
#define WHS_TFACE 512u    // a tt-metal tile face, [16][16] BF16
#define WHS_MAX_BUFS 32u
#define WHS_MAX_READERS 4u
#define WHS_IDLE 0xFFFFu

// Scratch map (offsets from the scratch base, the same on every core). Unpacker 0's region is chunks W x 8 + Z of
// WHS_FACE bytes: W 0 Q^T, W 1 P^T's two buffers, W 2 and 3 the Q staging of even and odd items ([d slice][64
// queries][16 d]), W 4 O^T / l as the packers leave it.
#define WHS_SYNC 0u
#define WHS_UNPACK0 4096u
#define WHS_W_QT 0u
#define WHS_W_P 1u
#define WHS_W_Q 2u
#define WHS_W_OT 4u
#define WHS_W_BYTES (8u * WHS_FACE)
// O staging of even and odd items: [2 query tiles][8 d slices][32 queries][16 d].
#define WHS_O_STAGE (WHS_UNPACK0 + 5u * WHS_W_BYTES)
#define WHS_O_STAGE_BYTES (2u * 8u * 1024u)
#define WHS_KV (WHS_O_STAGE + 2u * WHS_O_STAGE_BYTES)
#define WHS_SCRATCH_BYTES(bufs) (WHS_KV + (bufs) * WHS_BLOCK)

// Sync lines, one 16-byte line each from WHS_SYNC. READY[b]: uses of buffer b landed (its reader's cast). CONSUMED[b],
// on a reader: consumptions of buffer b by the group's cores. UNPACKED: blocks of the group's stream TRISC0 is done
// with. Q_LANDED (BRISC) / Q_TAKEN (TRISC0), O_STAGED (TRISC2) / O_WRITTEN (BRISC): items. GO: this launch's token,
// written once every line is zeroed and the group has met.
#define WHS_S_READY 0u
#define WHS_S_CONSUMED (WHS_S_READY + WHS_MAX_BUFS)
#define WHS_S_UNPACKED (WHS_S_CONSUMED + WHS_MAX_BUFS)
#define WHS_S_Q_LANDED (WHS_S_UNPACKED + 1u)
#define WHS_S_Q_TAKEN (WHS_S_UNPACKED + 2u)
#define WHS_S_O_STAGED (WHS_S_UNPACKED + 3u)
#define WHS_S_O_WRITTEN (WHS_S_UNPACKED + 4u)
#define WHS_S_GO (WHS_S_UNPACKED + 5u)
#define WHS_S_STATS (WHS_S_UNPACKED + 6u)  // TRISC0 K/V wait cycles, TRISC0 Q wait, TRISC2 O wait, feed cycles
// On a reader whose casts the I/O kernel makes (WHS_CAST_BY_IO): blocks the feed has landed, in its order.
#define WHS_S_LANDED (WHS_S_STATS + 5u)
// The valid key count for this launch, written by the I/O kernel before its core counts in at the group barrier (from
// the logical length tensor when WHS_A_VALID_ADDR is set); the feed and TRISCs read it after GO.
#define WHS_S_VALID (WHS_S_STATS + 6u)
#define WHS_VALID_STAGE 3072u                // 32-byte landing spot for the logical length read
#define WHS_S_MATH_READY (WHS_S_STATS + 8u)  // TRISC1 has initialised the semaphores (this launch's token)
#define WHS_S_SAVED (WHS_S_STATS + 9u)       // TRISC0 has saved the configuration
#define WHS_S_DONE (WHS_S_STATS + 10u)       // TRISC i done at WHS_S_DONE + i
#define WHS_S_RESTORED (WHS_S_STATS + 13u)
#define WHS_S_RING_PROBE (WHS_S_STATS + 14u)  // the feed's landing word for a ring counter read
#define WHS_S_LINES (WHS_S_STATS + 15u)
// The configuration (state 0) and each TRISC's GPRs as the previous op left them, restored at the end.
#define WHS_SAVE_CFG 1536u
#define WHS_SAVE_GPR 2304u
#define WHS_LINE 16u

static inline uint32_t whs_line(uint32_t scratch, uint32_t line) { return scratch + WHS_SYNC + line * WHS_LINE; }

// Runtime args shared by all kernels (the same order everywhere), then the rounds: per round (head << 16) | block.
enum WhsArg : uint32_t {
    WHS_A_SCRATCH = 0,
    WHS_A_GO,
    WHS_A_Q,
    WHS_A_K,
    WHS_A_VT,
    WHS_A_O,
    WHS_A_HEADS,
    WHS_A_Q_TILE_ROWS,  // query tile rows a head (rows / 32)
    WHS_A_K_TILE_ROWS,  // key tile rows a head
    WHS_A_VALID_KEYS,   // keys from this on are padding
    WHS_A_BUFS,
    WHS_A_READERS,
    WHS_A_READER_INDEX,  // WHS_IDLE: not a reader
    WHS_A_GROUP_CORES,
    WHS_A_GROUP_INDEX,
    WHS_A_GROUP_X0,  // the group rectangle, NoC 0 physical coordinates
    WHS_A_GROUP_Y0,
    WHS_A_GROUP_X1,
    WHS_A_GROUP_Y1,
    WHS_A_READER0_X,  // reader r's NoC 0 physical coordinates at READER0 + 2 r
    WHS_A_READER0_Y,
    WHS_A_LEADER_X = WHS_A_READER0_X + 2u * WHS_MAX_READERS,  // the group's barrier leader (reader 0)
    WHS_A_LEADER_Y,
    WHS_A_BARRIER_SEM,  // a program semaphore id: the group barrier's count on the leader
    WHS_A_DEBUG,
    WHS_A_VALID_ADDR,     // DRAM address of a [1, 1, 1, 1] uint32 logical length tensor (0: use WHS_A_VALID_KEYS)
    WHS_A_RING_COUNTERS,  // ring gather (wh_sdpa_gather.cpp): L1 address of the landing counters, 0 without
    WHS_A_RING_X,         // the counter core, NoC 0 coordinates
    WHS_A_RING_Y,
    WHS_A_RING_TARGET,  // a head's counter once its K and V^T have landed this launch (senders x launch)
    WHS_A_ROUNDS,
    WHS_A_ROUND0,
};

#define WHS_DEBUG_FEED_ONLY 2u  // every round idle: the K/V stream alone
#define WHS_DEBUG_NO_READS 4u   // readers cast without reading K/V
#define WHS_DEBUG_NO_CASTS 8u   // readers read without casting (only the ready words)
#define WHS_FACE1K 64u          // K and V^T tiles store their faces column-major (f0 f2 f1 f3): 2 reads of 1 KiB a tile
#define WHS_DEBUG_NO_ISSUE 32u  // readers compute every read's addresses but issue nothing
#define WHS_DEBUG_NO_PATH_RESERVE \
    16u                   // casts skip multicast path reservation: only with one caster per NoC per
                          // group, casting from the corner its NoC reaches the rectangle from without
                          // wrapping (two unreserved casters into one rectangle on a NoC hung the chip)
#define WHS_VT_FULL 256u  // V^T is V transposed whole ([heads][128][keys], ttnn.transpose(v, -2, -1)), not tile by tile
#define WHS_CAST_BY_IO 128u  // this reader's casts go out from the I/O kernel (its NoC), the feed only reads
// This reader casts (data and ready words) on the buddy multicast VC: a group's second caster on a NoC, so that each
// (NoC, VC) carries one source a group (two unreserved casters on one (NoC, VC) in a group deadlocked the NoC in
// tt-lab-private h3-wh, notes/minimax-h3-wh-compute.md "The feed redesign"). With WHS_CAST_BY_IO only.
#define WHS_CAST_BUDDY_VC 512u

// Ring gather (wh_sdpa_gather.cpp): K and V^T arrive while the op runs; head h's landing counter (L1 of the counter
// core, line h of 16 bytes) counts its senders' increments, the barrier line the launches started on the other chips.
#define WHS_RING_BARRIER 31u
#define WHS_GATHER_BATCH 4u    // 2-tile packets a gather batch (two in flight)
#define WHS_HEADER_BYTES 256u  // a packet header slot in the gather's staging
#define WHS_GATHER_BYTES (2u * WHS_GATHER_BATCH * 2u * WHS_TILE + (2u * WHS_GATHER_BATCH + 1u) * WHS_HEADER_BYTES)

static inline uint32_t whs_key_blocks(uint32_t valid_keys) { return (valid_keys + WHS_ROWS - 1u) / WHS_ROWS; }
