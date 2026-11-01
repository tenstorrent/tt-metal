// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm writer (BRISC / NoC1): GammaSliceLoad (once) + exchange_partials_block (k > 1) + store_block.
// Per block (k > 1): exchange_send -> exchange_recv_partials -> [block 0: finish the gamma load] -> exchange_recv_inv.
//
// GammaSliceLoad — this member's W slice of gamma into cb_gamma, "expanded" so every row of tile j holds
//   gamma[32j .. 32j+31] (lets pass 2 multiply with a plain DEST-reuse mul). Loaded here, not on the reader,
//   so the x stream starts at once on every Tensix core; compute waits for it only before its first pass 2.
//   RM gamma  : one contiguous read of the slice into tile 0's area, RISC-V scatter of the 2 x 16-datum
//               row-0 pieces per tile (descending j, so the staging bytes are consumed before overwritten).
//   TILE gamma: per tile, face 0 row 0 and face 1 row 0 straight from DRAM (64 B reads; a bfloat8_b tile's
//               64 B exponent section rides along with the first read and is re-patterned so every face
//               row carries row 0's shared exponent).
//   Both      : rows 1..15 of faces 0/1 and faces 2/3 are replicated by local NoC copies (doubling, one
//               barrier per stage) — no RISC-V whole-tile store loop.
//   Interleaved / sharded k = 1 (RMS_GAMMA_MCAST): read + expanded once per W slice and multicast to the slice's
//               group.
//   Streamed (RMS_GAMMA_CHUNK = G, fallback when the resident slice does not fit L1): per block, the slice is
//               re-loaded in G-tile chunks through a 2-chunk ring (always G pages pushed, so a chunk never wraps
//               the ring); pass 2 consumes one chunk per G-tile window.
//
// Gamma timing (Refinement 5): k == 1 / multicast gamma / blocks > GAMMA_AFTER_SEND_MAX_ROWS tile-rows — the load
//   runs at kernel start. Per-core load with short blocks (gamma_after_send) — the DRAM reads are issued at kernel
//   start; the barrier + expand run only after block 0's partials are out and the owner duties are done.
// Signal fence (signal_after_flush): every semaphore increment that publishes data writes follows a write flush,
//   not a barrier (same NoC + VC + destination => in-order delivery); the barrier at kernel exit retires the acks.
// exchange_partials_block — symmetric unicast all-gather, one round per block:
//   every member m writes its block_rows column-0 partial tiles into slot (r*k + m) of EVERY member's
//   cb_gathered_sumsq (its own included), barriers, then bumps each member's arrival semaphore by 1.
//   The slot base is this core's own cb_gathered write pointer: every member declares the identical CB
//   list and pushes identical counts, so the pointer is the same on all members (lockstep). Depth is 2
//   blocks: a peer can only reach block b+2 after consuming my partial of b+1, which I send only after my
//   compute has popped gathered(b) (exchange(b+1) follows store(b) in this kernel).
//   The semaphore is monotonic: block b is complete at k*(b+1) arrivals; it is never reset.
// exchange_partials_block — owner topology (combine_owner, k > K_SYM_MAX_*; interleaved k > 8, WIDTH / BLOCK_SHARDED;
//   interleaved groups with k > grid_x span grid rows — every peer is addressed by its own RT (x, y) pair):
//   tile-row r of the block is owned by member r % k. Every member writes its partial of row r into slot
//   ((r / k) * k + m) of the owner's cb_gathered_sumsq and bumps the owner's arrival semaphore once per row;
//   every member reserves/pushes the same ceil(rows / k) * k gathered pages per block (lockstep slot base).
//   An owner then returns its combined inv_rms tiles into slot r of EVERY member's cb_inv_rms (inv semaphore).
//   No slot is reused before it is consumed: a member's partial(b+1) exists only after its compute applied
//   inv_rms(b), i.e. after every owner of block b finished combine(b) and every member popped inv_rms(b-1).
// Payload (PARTIAL_FACES_ONLY): a partial / inv_rms tile is meaningful in column 0 only, which lives in faces
//   0 and 2 — ship those two faces, not the whole tile (the combine's Skip reduce and the Col-broadcast
//   normalize never read faces 1/3).
// store_block
//   TILE sharded: cb_out IS the output shard (zero-copy) — compute packs straight into it; the writer only
//         waits for the block's pages so the program cannot retire before the shard is written.
//   RM sharded: per (chunk, tile-row) the untilized stick pieces are written back to the local output shard.
//   TILE: per tile-row, in units of <= out_unit_tiles tiles: wait, write, flush, pop. A unit is also clamped
//         to the cb_out ring end (tracked in out_ring_pos) — its pages are read linearly from the read pointer,
//         and core_w_tiles need not be a multiple of out_unit_tiles.
//   RM  : chunk-major (matches compute's per-chunk untilize): per (chunk, tile-row) one stick-piece write
//         per valid stick; full chunks through write_sticks_after_untilize, the ragged chunk raw.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#ifdef RMS_GAMMA_MCAST
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#endif
#include "rms_norm_common.hpp"
#if defined(GS_NC_ROLE) && defined(RMS_MASK_TAIL)
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#endif

using namespace rms_norm_cb;

constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(0);
constexpr uint32_t W = get_compile_time_arg_val(1);
constexpr uint32_t H = get_compile_time_arg_val(2);
constexpr uint32_t image_row_tiles = get_compile_time_arg_val(3);
constexpr uint32_t rm_chunk_w_tiles = get_compile_time_arg_val(4);
constexpr uint32_t out_page_bytes = get_compile_time_arg_val(5);
constexpr uint32_t elem_bytes = get_compile_time_arg_val(6);
constexpr uint32_t w_split_cores = get_compile_time_arg_val(7);
constexpr uint32_t out_unit_tiles = get_compile_time_arg_val(8);
constexpr uint32_t sem_id = get_compile_time_arg_val(9);
constexpr uint32_t out_cb_pages = get_compile_time_arg_val(10);     // cb_out capacity (TILE: out_depth * out_unit)
constexpr uint32_t NOC_XY_ARG_BASE = get_compile_time_arg_val(11);  // RT index of the first peer (x, y) pair
constexpr bool combine_owner = get_compile_time_arg_val(12) != 0;
constexpr bool partial_faces_only = get_compile_time_arg_val(13) != 0;
constexpr uint32_t sem_inv_id = get_compile_time_arg_val(14);
constexpr uint32_t x_row_pitch = get_compile_time_arg_val(15);
constexpr uint32_t gamma_page_bytes = get_compile_time_arg_val(16);
constexpr uint32_t gamma_elem_bytes = get_compile_time_arg_val(17);
constexpr bool gamma_is_tile = get_compile_time_arg_val(18) != 0;
// Shared-exponent section ahead of the datums in one gamma tile: 64 B for bfloat8_b (one exponent byte per
// 16-datum face row, face-major), 0 for float32 / bfloat16. Datums (bfp8: mantissa bytes) follow in face order.
constexpr uint32_t gamma_exp_bytes = get_compile_time_arg_val(19);
// k > 1: load the resident gamma slice after block 0's partials are sent (else at kernel start).
constexpr bool gamma_after_send = get_compile_time_arg_val(20) != 0;
// Signal fence: before a semaphore increment, wait only until the preceding data writes have LEFT this core
// (flush) instead of for their completion acks (barrier). Safe: the writes and noc_semaphore_inc travel on the
// same NoC and the same unicast-write VC to the same destination, so they arrive in program order. The
// outstanding acks are retired once by the write barrier at kernel exit.
constexpr bool signal_after_flush = get_compile_time_arg_val(21) != 0;
constexpr auto out_args = TensorAccessorArgs<22>();
[[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;

#ifdef RMS_HAS_GAMMA
// Replicate row 0 of faces 0/1 of `num_tiles` gamma tiles into every row of every face (local NoC copies).
constexpr uint32_t g_tile_bytes = gamma_exp_bytes + TILE_DIM * TILE_DIM * gamma_elem_bytes;

// Block-float gamma: every face row of tile j must carry row 0's shared exponent — faces 0/2 take face 0
// row 0's, faces 1/3 face 1 row 0's. 16 word stores per tile (the pattern, not a whole-tile fill).
FORCE_INLINE void expand_gamma_exponents(uint32_t base, uint32_t num_tiles) {
    static_assert(
        gamma_exp_bytes == 0 || gamma_exp_bytes == 4 * FACE_DIM, "bfp gamma: expected one exponent byte per face row");
    if constexpr (gamma_exp_bytes > 0) {
        constexpr uint32_t words_per_face = FACE_DIM / sizeof(uint32_t);
        for (uint32_t j = 0; j < num_tiles; ++j) {
            volatile tt_l1_ptr uint8_t* exp8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base + j * g_tile_bytes);
            volatile tt_l1_ptr uint32_t* exp32 =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + j * g_tile_bytes);
            const uint32_t e_left = exp8[0] * 0x01010101u;          // face 0 row 0
            const uint32_t e_right = exp8[FACE_DIM] * 0x01010101u;  // face 1 row 0
            for (uint32_t w = 0; w < words_per_face; ++w) {
                exp32[0 * words_per_face + w] = e_left;
                exp32[1 * words_per_face + w] = e_right;
                exp32[2 * words_per_face + w] = e_left;
                exp32[3 * words_per_face + w] = e_right;
            }
        }
    }
}

FORCE_INLINE void expand_gamma_rows(uint32_t base, uint32_t num_tiles) {
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * gamma_elem_bytes;
    constexpr uint32_t face_row_bytes = FACE_DIM * gamma_elem_bytes;
    expand_gamma_exponents(base, num_tiles);
    for (uint32_t rows = 1; rows < FACE_DIM; rows <<= 1) {
        const uint32_t bytes = rows * face_row_bytes;
        for (uint32_t j = 0; j < num_tiles; ++j) {
            const uint32_t tile = base + j * g_tile_bytes + gamma_exp_bytes;
            noc_async_read(get_noc_addr(tile), tile + bytes, bytes);                            // face 0
            noc_async_read(get_noc_addr(tile + face_bytes), tile + face_bytes + bytes, bytes);  // face 1
        }
        noc_async_read_barrier();
    }
    for (uint32_t j = 0; j < num_tiles; ++j) {  // faces 0,1 -> faces 2,3 (rows 16..31)
        const uint32_t tile = base + j * g_tile_bytes + gamma_exp_bytes;
        noc_async_read(get_noc_addr(tile), tile + 2 * face_bytes, 2 * face_bytes);
    }
    noc_async_read_barrier();
}

constexpr uint32_t GAMMA_READ_ALIGN = 64;  // BH DRAM read alignment (also >= one 16-datum face row)

// RM gamma: bytes of lead-in below the 64 B-aligned read start of the slice.
FORCE_INLINE uint32_t gamma_rm_lead(uint32_t gamma_col0) {
    return (gamma_col0 * gamma_elem_bytes) & (GAMMA_READ_ALIGN - 1);
}

// issue_gamma_reads: the slice's row-0 DRAM reads into `base` (no barrier — finish_gamma_slice waits for them).
template <typename Accessor>
FORCE_INLINE void issue_gamma_reads(
    const Accessor& g_acc,
    uint32_t base,
    uint32_t w_tile_start,
    uint32_t core_w_tiles,
    uint32_t gamma_col0,
    uint32_t gamma_cols) {
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * gamma_elem_bytes;
    if constexpr (gamma_is_tile) {
        // Per tile, two 64 B-aligned reads: [exponent section (bfp only) + face 0 row 0], [face 1 row 0].
        constexpr uint32_t d0 = gamma_exp_bytes;
        for (uint32_t j = 0; j < core_w_tiles; ++j) {
            const uint32_t tile = base + j * g_tile_bytes;
            noc_async_read(g_acc.get_noc_addr(w_tile_start + j, 0), tile, d0 + GAMMA_READ_ALIGN);
            noc_async_read(
                g_acc.get_noc_addr(w_tile_start + j, d0 + face_bytes), tile + d0 + face_bytes, GAMMA_READ_ALIGN);
        }
    } else {
        const uint32_t byte0 = gamma_col0 * gamma_elem_bytes;
        const uint32_t lead = gamma_rm_lead(gamma_col0);
        noc_async_read(g_acc.get_noc_addr(0, byte0 - lead), base, lead + gamma_cols * gamma_elem_bytes);
    }
}

// finish_gamma_slice: wait for the row-0 reads, (RM) scatter row 0 into the tile faces, expand rows 1..31.
FORCE_INLINE void finish_gamma_slice(uint32_t base, uint32_t core_w_tiles, uint32_t gamma_col0) {
    noc_async_read_barrier();
    if constexpr (!gamma_is_tile) {
        constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * gamma_elem_bytes;
        const uint32_t lead = gamma_rm_lead(gamma_col0);
        // Scatter row 0: datums [32j, 32j+16) -> tile j face 0 row 0, [32j+16, 32j+32) -> face 1 row 0.
        constexpr uint32_t words_per_face_row = FACE_DIM * gamma_elem_bytes / sizeof(uint32_t);
        for (uint32_t jj = core_w_tiles; jj > 0; --jj) {
            const uint32_t j = jj - 1;
            volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + lead + j * TILE_DIM * gamma_elem_bytes);
            volatile tt_l1_ptr uint32_t* dst0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + j * g_tile_bytes);
            volatile tt_l1_ptr uint32_t* dst1 =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + j * g_tile_bytes + face_bytes);
            uint32_t tmp[2 * words_per_face_row];
            for (uint32_t w = 0; w < 2 * words_per_face_row; ++w) {
                tmp[w] = src[w];
            }
            for (uint32_t w = 0; w < words_per_face_row; ++w) {
                dst0[w] = tmp[w];
                dst1[w] = tmp[words_per_face_row + w];
            }
        }
    }
    expand_gamma_rows(base, core_w_tiles);
}

// Fill `base` (cb_gamma's write pointer) with this member's expanded gamma slice; the caller owns the CB
// reserve/push. gamma_col0 / gamma_cols: first global column and column count of the slice (RM gamma). The
// slice start need not be tile- or 64 B-aligned (RM shards are 16 B granular): the read starts at the 64 B-
// aligned offset below it and the scatter skips the lead bytes.
template <typename Accessor>
FORCE_INLINE void fill_gamma_slice(
    const Accessor& g_acc,
    uint32_t base,
    uint32_t w_tile_start,
    uint32_t core_w_tiles,
    uint32_t gamma_col0,
    uint32_t gamma_cols) {
    issue_gamma_reads(g_acc, base, w_tile_start, core_w_tiles, gamma_col0, gamma_cols);
    finish_gamma_slice(base, core_w_tiles, gamma_col0);
}
#endif

#ifdef RMS_HAS_GAMMA
// GammaSliceLoad: this member's expanded gamma slice into cb_gamma (program lifetime; compute first needs it
// at pass 2, so this overlaps the reader's x stream and pass 1).
//   RMS_GAMMA_MCAST (interleaved): the Tensix cores sharing a W slice form one multicast group (host Mcast);
//     its sender reads + expands the slice once and multicasts the expanded tiles to the group — the slice is
//     read from DRAM once per W slice instead of once per Tensix core. One-shot (no handshake): cb_gamma is
//     written once per launch and nothing else touches its L1.
//   otherwise: every Tensix core reads + expands its own slice.
//   Split in two so the DRAM reads can be in flight while the writer serves the combine:
//   begin() reserves cb_gamma and issues this core's row-0 reads (per-core load, or the multicast sender);
//   end() waits for them, expands, multicasts (sender) / receives (receiver) and pushes. load() = both.
#ifdef RMS_GAMMA_MCAST
constexpr auto gamma_mcast = dataflow_kernel_lib::McastArgs<
    get_named_compile_time_arg_val("gamma_mcast_ct_offset"),
    get_named_compile_time_arg_val("gamma_mcast_rt_offset")>();
#endif

template <typename Accessor>
struct GammaSliceLoad {
    const Accessor& g_acc;
    uint32_t w_tile_start;
    uint32_t core_w_tiles;
    uint32_t push_tiles;
    uint32_t col0;
    uint32_t cols;
    uint32_t base = 0;

    // This core reads the slice from DRAM itself (per-core load, or the slice group's multicast sender).
    FORCE_INLINE bool reads_dram() const {
#ifdef RMS_GAMMA_MCAST
        return gamma_mcast.can_send();
#else
        return true;
#endif
    }

    FORCE_INLINE void begin() {
        MaybeDeviceZoneScope("wr_gamma_issue");
        cb_reserve_back(cb_gamma, push_tiles);
        base = get_write_ptr(cb_gamma);
        if (reads_dram()) {
            issue_gamma_reads(g_acc, base, w_tile_start, core_w_tiles, col0, cols);
        }
    }

    FORCE_INLINE void end() {
        MaybeDeviceZoneScope("wr_gamma_finish");  // barrier + expand (+ multicast send / receive wait)
#ifdef RMS_GAMMA_MCAST
        Noc noc;
        // optional_*: a W slice owned by a single Tensix core (one row-group) has a sender and no receivers.
        if (auto sender = gamma_mcast.optional_sender(noc)) {
            finish_gamma_slice(base, core_w_tiles, col0);
            sender->send(base, base, core_w_tiles * g_tile_bytes);  // zero fan-out: src == dst, no transfer
        } else if (auto receiver = gamma_mcast.optional_receiver(noc)) {
            receiver->receive();
        }
#else
        finish_gamma_slice(base, core_w_tiles, col0);
#endif
        cb_push_back(cb_gamma, push_tiles);
    }
};
#endif

#ifdef RMS_HAS_GAMMA
// ===== gamma_stream_l1 bench variants (perf_experiments/gamma_stream_l1) =====
// GS_STATEFUL: the expand's local NoC copies use the one-packet stateful read API — every copy of a stage has
//   the same size and source core (this one), so the cmd buffer's size / source-coordinate registers are written
//   once per stage and each copy only writes its two L1 addresses (vs a full register set per noc_async_read).
// GS_COMPACT: the row-0 slab of this member's whole gamma slice is read from DRAM ONCE per program into a small
//   resident L1 region (cb_gamma_row: TILE gamma — per tile the same two 64 B-aligned pieces the per-chunk path
//   reads, d0 + 128 B; RM gamma — the contiguous slice row), at kernel start (overlaps pass 1). Each chunk's tiles
//   are then placed from L1 (local copies) instead of DRAM, then expanded as before.
FORCE_INLINE void expand_rows_any(uint32_t base, uint32_t num_tiles) {
#ifdef GS_STATEFUL
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * gamma_elem_bytes;
    constexpr uint32_t face_row_bytes = FACE_DIM * gamma_elem_bytes;
    expand_gamma_exponents(base, num_tiles);
    const uint64_t self = get_noc_addr(0);
    for (uint32_t rows = 1; rows < FACE_DIM; rows <<= 1) {
        const uint32_t bytes = rows * face_row_bytes;
        noc_async_read_one_packet_set_state(self, bytes);
        for (uint32_t j = 0; j < num_tiles; ++j) {
            const uint32_t tile = base + j * g_tile_bytes + gamma_exp_bytes;
            noc_async_read_one_packet_with_state(tile, tile + bytes);
            noc_async_read_one_packet_with_state(tile + face_bytes, tile + face_bytes + bytes);
        }
        noc_async_read_barrier();
    }
    noc_async_read_one_packet_set_state(self, 2 * face_bytes);
    for (uint32_t j = 0; j < num_tiles; ++j) {
        const uint32_t tile = base + j * g_tile_bytes + gamma_exp_bytes;
        noc_async_read_one_packet_with_state(tile, tile + 2 * face_bytes);
    }
    noc_async_read_barrier();
#else
    expand_gamma_rows(base, num_tiles);
#endif
}

#if defined(GS_COMPACT) || defined(GS_RES_LOCAL)
constexpr uint32_t gs_slab_tile_bytes = gamma_exp_bytes + 2 * GAMMA_READ_ALIGN;  // TILE gamma: per-tile row-0 slab

// Resident row slab: issue the DRAM reads (no barrier) into cb_gamma_row's L1.
template <typename Accessor>
FORCE_INLINE uint32_t gs_issue_row_slab(
    const Accessor& g_acc,
    uint32_t w_tile_start,
    uint32_t core_w_tiles,
    uint32_t col0,
    uint32_t valid_cols,
    uint32_t parity,  // GS_SPLIT: only the tiles of chunks with (chunk index & 1) == parity; 2 = all
    uint32_t chunk) {
    cb_reserve_back(cb_gamma_row, 1);  // private L1 region (both RISC-Vs: disjoint tile slabs), never pushed
    const uint32_t row = get_write_ptr(cb_gamma_row);
    if constexpr (gamma_is_tile) {
        constexpr uint32_t d0 = gamma_exp_bytes;
        constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * gamma_elem_bytes;
        for (uint32_t j = 0; j < core_w_tiles; ++j) {
            if (parity < 2 && ((j / chunk) & 1) != parity) {
                continue;
            }
            const uint32_t slab = row + j * gs_slab_tile_bytes;
            noc_async_read(g_acc.get_noc_addr(w_tile_start + j, 0), slab, d0 + GAMMA_READ_ALIGN);
            noc_async_read(
                g_acc.get_noc_addr(w_tile_start + j, d0 + face_bytes), slab + d0 + GAMMA_READ_ALIGN, GAMMA_READ_ALIGN);
        }
    } else {
        const uint32_t byte0 = col0 * gamma_elem_bytes;
        const uint32_t lead = gamma_rm_lead(col0);
        noc_async_read(g_acc.get_noc_addr(0, byte0 - lead), row, lead + valid_cols * gamma_elem_bytes);
    }
    return row;
}

// Place row 0 of chunk tiles [c0, c0 + n) from the resident slab into `base`, then expand.
FORCE_INLINE void gs_fill_from_slab(uint32_t row, uint32_t base, uint32_t c0, uint32_t n, uint32_t col0) {
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * gamma_elem_bytes;
    if constexpr (gamma_is_tile) {
        constexpr uint32_t d0 = gamma_exp_bytes;
        const uint64_t self = get_noc_addr(0);
        // piece 0: [exp section (bfp) + face 0 row 0 (+ row 1 junk, overwritten by the expand)]
        noc_async_read_one_packet_set_state(self, d0 + GAMMA_READ_ALIGN);
        for (uint32_t j = 0; j < n; ++j) {
            noc_async_read_one_packet_with_state(row + (c0 + j) * gs_slab_tile_bytes, base + j * g_tile_bytes);
        }
        noc_async_read_one_packet_set_state(self, GAMMA_READ_ALIGN);
        for (uint32_t j = 0; j < n; ++j) {
            noc_async_read_one_packet_with_state(
                row + (c0 + j) * gs_slab_tile_bytes + d0 + GAMMA_READ_ALIGN, base + j * g_tile_bytes + d0 + face_bytes);
        }
        noc_async_read_barrier();
    } else {
        const uint32_t lead = gamma_rm_lead(col0);
#ifdef GS_RM_SCATTER_RISCV
        // RISC-V scatter straight from the resident row (no staging, so no descending-order constraint).
        constexpr uint32_t words_per_face_row = FACE_DIM * gamma_elem_bytes / sizeof(uint32_t);
        for (uint32_t j = 0; j < n; ++j) {
            volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row + lead + (c0 + j) * TILE_DIM * gamma_elem_bytes);
            volatile tt_l1_ptr uint32_t* dst0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + j * g_tile_bytes);
            volatile tt_l1_ptr uint32_t* dst1 =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + j * g_tile_bytes + face_bytes);
            for (uint32_t w = 0; w < words_per_face_row; ++w) {
                dst0[w] = src[w];
                dst1[w] = src[words_per_face_row + w];
            }
        }
#else
        // Local NoC copies of the two 16-datum row-0 pieces per tile (16 B-aligned: lead is a 16 B multiple).
        constexpr uint32_t frb = FACE_DIM * gamma_elem_bytes;
        const uint64_t self = get_noc_addr(0);
        noc_async_read_one_packet_set_state(self, frb);
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t src = row + lead + (c0 + j) * TILE_DIM * gamma_elem_bytes;
            const uint32_t dst = base + j * g_tile_bytes;
            noc_async_read_one_packet_with_state(src, dst);
            noc_async_read_one_packet_with_state(src + frb, dst + face_bytes);
        }
        noc_async_read_barrier();
#endif
    }
}
#endif

#if defined(GS_COMPACT) || defined(GS_RES_LOCAL)
// Acquire the resident row slab (returns its L1 base).
//   GS_SLAB_SEM, BRISC (RMS_GAMMA_MCAST): the HEIGHT group's sender reads the WHOLE slab once and multicasts it
//     (a few KB-tens of KB instead of every Tensix core issuing ~2 DRAM reads per gamma tile at kernel start); then
//     the local flag semaphore GS_SLAB_SEM is raised for this core's NCRISC.
//   GS_SLAB_SEM, NCRISC (GS_NC_ROLE): wait for that flag — the slab is already in this core's L1.
//   otherwise: this RISC-V reads the tiles of its own chunk parity itself.
template <typename Accessor>
FORCE_INLINE uint32_t gs_slab_acquire(
    const Accessor& g_acc,
    uint32_t w_tile_start,
    uint32_t core_w_tiles,
    uint32_t col0,
    uint32_t valid_cols,
    uint32_t parity,
    uint32_t chunk) {
#if defined(GS_SLAB_SEM) && defined(GS_NC_ROLE)
    cb_reserve_back(cb_gamma_row, 1);
    const uint32_t row = get_write_ptr(cb_gamma_row);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(GS_SLAB_SEM)), 1);
    return row;
#elif defined(GS_SLAB_SEM) && defined(RMS_GAMMA_MCAST)
    cb_reserve_back(cb_gamma_row, 1);
    uint32_t row = get_write_ptr(cb_gamma_row);
    {
        MaybeDeviceZoneScope("gs_slab_mcast");
        Noc noc;
        if (auto sender = gamma_mcast.optional_sender(noc)) {
            row = gs_issue_row_slab(g_acc, w_tile_start, core_w_tiles, col0, valid_cols, 2, chunk);
            noc_async_read_barrier();
            uint32_t slab_bytes;
            if constexpr (gamma_is_tile) {
                slab_bytes = core_w_tiles * gs_slab_tile_bytes;
            } else {
                slab_bytes = ((gamma_rm_lead(col0) + valid_cols * gamma_elem_bytes + 63) / 64) * 64;
            }
            sender->send(row, row, slab_bytes);
        } else if (auto receiver = gamma_mcast.optional_receiver(noc)) {
            receiver->receive();
        }
    }
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(GS_SLAB_SEM)), 1);
    return row;
#else
    const uint32_t row = gs_issue_row_slab(g_acc, w_tile_start, core_w_tiles, col0, valid_cols, parity, chunk);
    noc_async_read_barrier();
    return row;
#endif
}
#endif

#ifdef GS_RES_LOCAL
// Resident slice, local expand (no multicast of expanded tiles): tiles [c0, c0 + n) into cb_g's next n pages,
// placed from the resident row slab + expanded, pushed at once (compute waits progressively).
template <uint32_t cb_g>
FORCE_INLINE void gs_res_chunk(uint32_t row, uint32_t core_w_tiles, uint32_t col0, uint32_t c0, uint32_t n) {
    cb_reserve_back(cb_g, n);
    if (c0 < core_w_tiles) {
        const uint32_t valid = (core_w_tiles - c0) < n ? (core_w_tiles - c0) : n;
        const uint32_t base = get_write_ptr(cb_g);
        gs_fill_from_slab(row, base, c0, valid, col0);
        expand_rows_any(base, valid);
    }
    cb_push_back(cb_g, n);
}
#endif

#ifdef RMS_GAMMA_CHUNK

// stream_gamma_chunk: gamma tiles [c0, c0 + G) of this member's slice into the next G pages of the cb_gamma ring
// (always G pages pushed, so a chunk never wraps the ring). Gap tiles past core_w_tiles (a ragged shard column's
// padding / the RM chunk pitch) are pushed unwritten — their output is don't-care.
template <uint32_t cb_g, typename Accessor>
FORCE_INLINE void stream_gamma_chunk(
    const Accessor& g_acc,
    uint32_t w_tile_start,
    uint32_t core_w_tiles,
    uint32_t col0,
    uint32_t valid_cols,
    uint32_t c0,
    [[maybe_unused]] uint32_t row) {
    constexpr uint32_t G = RMS_GAMMA_CHUNK;
    {
        MaybeDeviceZoneScope("gs_ring_wait");  // back-pressure: compute has not freed a ring slot
        cb_reserve_back(cb_g, G);
    }
#ifdef GS_ABLATE_FILL
    if (false) {  // ablation: ring handshake only, no gamma payload (compute-bound floor of pass 2 with gamma)
#else
    if (c0 < core_w_tiles) {
#endif
        const uint32_t valid = (core_w_tiles - c0) < G ? (core_w_tiles - c0) : G;
        const uint32_t base = get_write_ptr(cb_g);
#ifdef GS_COMPACT
        {
            MaybeDeviceZoneScope("gs_place");  // row 0 from the resident slab (local)
            gs_fill_from_slab(row, base, c0, valid, col0);
        }
#else
        const uint32_t cols =
            (valid_cols - c0 * TILE_DIM) < valid * TILE_DIM ? (valid_cols - c0 * TILE_DIM) : valid * TILE_DIM;
        {
            MaybeDeviceZoneScope("gs_place");  // row 0 from DRAM: issue + barrier (+ RM scatter)
            issue_gamma_reads(g_acc, base, w_tile_start + c0, valid, col0 + c0 * TILE_DIM, cols);
            noc_async_read_barrier();
            if constexpr (!gamma_is_tile) {
                constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * gamma_elem_bytes;
                const uint32_t lead = gamma_rm_lead(col0 + c0 * TILE_DIM);
                constexpr uint32_t words_per_face_row = FACE_DIM * gamma_elem_bytes / sizeof(uint32_t);
                for (uint32_t jj = valid; jj > 0; --jj) {
                    const uint32_t j = jj - 1;
                    volatile tt_l1_ptr uint32_t* src =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + lead + j * TILE_DIM * gamma_elem_bytes);
                    volatile tt_l1_ptr uint32_t* dst0 =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + j * g_tile_bytes);
                    volatile tt_l1_ptr uint32_t* dst1 =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + j * g_tile_bytes + face_bytes);
                    uint32_t tmp[2 * words_per_face_row];
                    for (uint32_t w = 0; w < 2 * words_per_face_row; ++w) {
                        tmp[w] = src[w];
                    }
                    for (uint32_t w = 0; w < words_per_face_row; ++w) {
                        dst0[w] = tmp[w];
                        dst1[w] = tmp[words_per_face_row + w];
                    }
                }
            }
        }
#endif
        {
            MaybeDeviceZoneScope("gs_expand");  // rows 1..31 (local copies, one barrier per stage)
            expand_rows_any(base, valid);
        }
    }
    cb_push_back(cb_g, G);
}
#endif
#endif  // RMS_HAS_GAMMA (gamma_stream_l1 helpers)

// One partial / inv_rms tile to a peer: faces 0 and 2 (column 0) or the whole tile.
FORCE_INLINE void send_acc_tile(uint32_t src, uint64_t dst, uint32_t tile_bytes) {
    if constexpr (partial_faces_only) {
        const uint32_t face_bytes = tile_bytes / 4;
        noc_async_write(src, dst, face_bytes);
        noc_async_write(src + 2 * face_bytes, dst + 2 * face_bytes, face_bytes);
    } else {
        noc_async_write(src, dst, tile_bytes);
    }
}

FORCE_INLINE uint64_t peer_addr(uint32_t peer, uint32_t l1_addr) {
    return get_noc_addr(
        get_arg_val<uint32_t>(NOC_XY_ARG_BASE + 2 * peer),
        get_arg_val<uint32_t>(NOC_XY_ARG_BASE + 2 * peer + 1),
        l1_addr);
}

FORCE_INLINE void fence_before_signal() {
    if constexpr (signal_after_flush) {
        noc_async_writes_flushed();
    } else {
        noc_async_write_barrier();
    }
}

// Exchange state carried across blocks (k > 1). The semaphores are monotonic: never reset.
struct ExchangeState {
    uint32_t member_idx;
    uint32_t sem_addr;
    uint32_t sem_inv_addr;
    uint32_t tile_bytes;
    uint32_t slot_base;  // this block's cb_gathered_sumsq slot base (reserved by the send half)
    uint32_t partials_expected;
    uint32_t inv_expected;
};

FORCE_INLINE uint32_t owned_rows(uint32_t rows, uint32_t member_idx) {
    constexpr uint32_t k = w_split_cores;
    return member_idx < rows ? (rows - member_idx + k - 1) / k : 0;
}

// exchange send half: this member's partials of the block out to their receivers (no cross-core wait).
//   symmetric: slot (r*k + m) of EVERY member's cb_gathered_sumsq; owner: slot (r/k)*k + m of row r's owner r % k.
// The gathered slots are reserved here (the slot base is this core's own write pointer — lockstep on all members).
FORCE_INLINE void exchange_send(uint32_t rows, ExchangeState& st) {
    constexpr uint32_t k = w_split_cores;
    const uint32_t tb = st.tile_bytes;
    {
        MaybeDeviceZoneScope("wr_wait_partial");  // starved on compute's pass 1 + collapse
        cb_wait_front(cb_partial_sumsq, rows);
    }
    MaybeDeviceZoneScope("wr_xchg_send");
    const uint32_t src = get_read_ptr(cb_partial_sumsq);
    if constexpr (combine_owner) {
        cb_reserve_back(cb_gathered_sumsq, ((rows + k - 1) / k) * k);
        st.slot_base = get_write_ptr(cb_gathered_sumsq);
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t slot = (r / k) * k + st.member_idx;
            send_acc_tile(src + r * tb, peer_addr(r % k, st.slot_base + slot * tb), tb);
        }
        fence_before_signal();
        const uint32_t n_owners = rows < k ? rows : k;
        for (uint32_t o = 0; o < n_owners; ++o) {
            noc_semaphore_inc(peer_addr(o, st.sem_addr), (rows - o + k - 1) / k);
        }
    } else {
        cb_reserve_back(cb_gathered_sumsq, k * rows);
        st.slot_base = get_write_ptr(cb_gathered_sumsq);
        for (uint32_t m = 0; m < k; ++m) {
            for (uint32_t r = 0; r < rows; ++r) {
                send_acc_tile(src + r * tb, peer_addr(m, st.slot_base + (r * k + st.member_idx) * tb), tb);
            }
        }
        fence_before_signal();
        for (uint32_t m = 0; m < k; ++m) {
            noc_semaphore_inc(peer_addr(m, st.sem_addr), 1);
        }
    }
    cb_pop_front(cb_partial_sumsq, rows);
}

// exchange receive, part 1: wait for this member's gathered partials -> cb_gathered_sumsq (compute combines);
//   owner topology: then return the combined inv_rms of each owned row r into slot r of every member's cb_inv_rms.
FORCE_INLINE void exchange_recv_partials(uint32_t rows, ExchangeState& st) {
    constexpr uint32_t k = w_split_cores;
    if constexpr (combine_owner) {
        const uint32_t tb = st.tile_bytes;
        const uint32_t owned = owned_rows(rows, st.member_idx);
        st.partials_expected += owned * k;
        {
            MaybeDeviceZoneScope("wr_wait_peers");  // rendezvous: the slowest member's partial
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st.sem_addr), st.partials_expected);
        }
        cb_push_back(cb_gathered_sumsq, ((rows + k - 1) / k) * k);
        MaybeDeviceZoneScope("wr_inv_return");  // occupancy: waits on compute's owner combine + k-way return

        cb_reserve_back(cb_inv_rms, rows);
        const uint32_t inv_base = get_write_ptr(cb_inv_rms);
        if (owned > 0) {
            // One owned tile at a time from the CB front: members own different row counts per block, so the
            // read pointer is not block-aligned and a multi-tile linear walk could cross the ring end.
            for (uint32_t j = 0; j < owned; ++j) {
                const uint32_t r = j * k + st.member_idx;
                cb_wait_front(cb_inv_owned, 1);
                const uint32_t inv_src = get_read_ptr(cb_inv_owned);
                for (uint32_t m = 0; m < k; ++m) {
                    send_acc_tile(inv_src, peer_addr(m, inv_base + r * tb), tb);
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_inv_owned, 1);
            }
            fence_before_signal();
            for (uint32_t m = 0; m < k; ++m) {
                noc_semaphore_inc(peer_addr(m, st.sem_inv_addr), owned);
            }
        }
    } else {
        st.partials_expected += k;
        {
            MaybeDeviceZoneScope("wr_wait_peers");  // rendezvous: the slowest member's partial
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st.sem_addr), st.partials_expected);
        }
        cb_push_back(cb_gathered_sumsq, k * rows);
    }
}

// exchange receive, part 2 (owner topology): wait for every row's inv_rms of the block -> cb_inv_rms.
FORCE_INLINE void exchange_recv_inv(uint32_t rows, ExchangeState& st) {
    if constexpr (combine_owner) {
        st.inv_expected += rows;
        MaybeDeviceZoneScope("wr_wait_inv");  // rendezvous: every owner's inv_rms return
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st.sem_inv_addr), st.inv_expected);
        cb_push_back(cb_inv_rms, rows);
    }
}

// Perf ablation hooks (perf probes only; never set by the op): RMS_ABLATE_EMPTY returns at once (fixed-cost
// floor of the same program on the same Tensix cores); RMS_ABLATE_DM drops the interleaved TILE x / out NoC
// payload but keeps every CB handshake and barrier.
#ifdef RMS_ABLATE_DM
constexpr bool ablate_dm = true;
#else
constexpr bool ablate_dm = false;
#endif

#ifdef GS_NC_ROLE
// gamma_stream_l1 GS_SPLIT, NCRISC side (TILE sharded only — the reader's job there is a single cb_x publish):
// publish the resident shard to compute (the reader's duty), then stream the ODD gamma chunks into cb_gamma2
// on NoC0, in parallel with BRISC's even chunks on NoC1.
void kernel_main() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const uint32_t block_row_tiles = get_arg_val<uint32_t>(3);
    const uint32_t last_block_row_tiles = get_arg_val<uint32_t>(4);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(5);
    const uint32_t core_w_tiles = get_arg_val<uint32_t>(6);
    const uint32_t valid_cols = get_arg_val<uint32_t>(9);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(11);
    const uint32_t col0 = get_arg_val<uint32_t>(12);
#ifdef RMS_MASK_TAIL
    constexpr uint32_t w_valid_in_last_tile = W - TILE_DIM * (tensor_w_tiles - 1);
    if (w_tile_start + core_w_tiles == tensor_w_tiles) {
        dataflow_kernel_lib::prepare_reduce_mask<cb_tail_mask, ckernel::ReduceDim::REDUCE_ROW>(w_valid_in_last_tile);
    }
#endif
    {
        const uint32_t shard_rows = (num_blocks - 1) * block_row_tiles + last_block_row_tiles;
        cb_reserve_back(cb_x, shard_rows * x_row_pitch);
        cb_push_back(cb_x, shard_rows * x_row_pitch);
    }
    const auto g_acc = TensorAccessor(gamma_args, gamma_addr, gamma_page_bytes);
#ifdef GS_RES_LOCAL
    {
        MaybeDeviceZoneScope("gs_res_load");  // resident slice, odd chunks: slab reads + local expand
        constexpr uint32_t Gr = GS_RES_CHUNK;
        const uint32_t total = x_row_pitch;
        const uint32_t row = gs_slab_acquire(g_acc, w_tile_start, core_w_tiles, col0, valid_cols, 1, Gr);
        for (uint32_t c0 = Gr; c0 < total; c0 += 2 * Gr) {
            gs_res_chunk<cb_gamma2>(row, core_w_tiles, col0, c0, (total - c0) < Gr ? (total - c0) : Gr);
        }
    }
    return;
#else
    constexpr uint32_t gamma_chunk = RMS_GAMMA_CHUNK;
    const uint32_t norm_w = x_row_pitch;
#ifdef GS_COMPACT
    uint32_t gs_row;
    {
        MaybeDeviceZoneScope("gs_row_load");
        gs_row = gs_slab_acquire(g_acc, w_tile_start, core_w_tiles, col0, valid_cols, 1, RMS_GAMMA_CHUNK);
    }
#else
    const uint32_t gs_row = 0;
#endif
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        MaybeDeviceZoneScope("wr_gamma_stream");
        for (uint32_t c0 = gamma_chunk; c0 < norm_w; c0 += 2 * gamma_chunk) {
            stream_gamma_chunk<cb_gamma2>(g_acc, w_tile_start, core_w_tiles, col0, valid_cols, c0, gs_row);
        }
    }
#endif
}
#else
void kernel_main() {
#ifdef RMS_ABLATE_EMPTY
    return;
#endif
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const uint32_t block_row_tiles = get_arg_val<uint32_t>(3);
    const uint32_t last_block_row_tiles = get_arg_val<uint32_t>(4);
    [[maybe_unused]] const uint32_t w_tile_start = get_arg_val<uint32_t>(5);
    const uint32_t core_w_tiles = get_arg_val<uint32_t>(6);
    [[maybe_unused]] const uint32_t chunks_this = get_arg_val<uint32_t>(7);
    [[maybe_unused]] const uint32_t member_idx = get_arg_val<uint32_t>(8);
    [[maybe_unused]] const uint32_t valid_cols = get_arg_val<uint32_t>(9);     // RM sharded: valid slice columns
    [[maybe_unused]] const uint32_t valid_sticks = get_arg_val<uint32_t>(10);  // RM sharded: valid shard sticks
    [[maybe_unused]] const uint32_t gamma_addr = get_arg_val<uint32_t>(11);
    [[maybe_unused]] const uint32_t col0 = get_arg_val<uint32_t>(12);  // first global column of the slice

#if defined(RMS_HAS_GAMMA) && defined(RMS_GAMMA_CHUNK)
    const auto g_acc = TensorAccessor(gamma_args, gamma_addr, gamma_page_bytes);
#if defined(RMS_RM)
    const uint32_t norm_w = chunks_this * rm_chunk_w_tiles;  // RM: pass 2 walks chunks_this C-tile windows
#elif defined(RMS_SHARDED)
    const uint32_t norm_w = x_row_pitch;  // sharded TILE: pass 2 walks the whole shard row
#else
    const uint32_t norm_w = core_w_tiles;
#endif
#ifdef GS_COMPACT
    uint32_t gs_row;
    {
        MaybeDeviceZoneScope("gs_row_load");  // once per program: row-0 slab DRAM reads + barrier
#ifdef GS_SPLIT
        gs_row = gs_slab_acquire(g_acc, w_tile_start, core_w_tiles, col0, valid_cols, 0, RMS_GAMMA_CHUNK);
#else
        gs_row = gs_slab_acquire(g_acc, w_tile_start, core_w_tiles, col0, valid_cols, 2, RMS_GAMMA_CHUNK);
#endif
    }
#else
    const uint32_t gs_row = 0;
#endif
#endif
    // Resident gamma slice (once per program). k == 1, multicast gamma, or !gamma_after_send: loaded at kernel start,
    // overlapping the reader's x stream and pass 1. Per-core load with k > 1 and gamma_after_send: only the DRAM reads
    // are issued at kernel start;
    // the barrier + expand (+ multicast) run after block 0's partials have left (symmetric) / after this member's
    // owner duties — the combine's critical path (partials out, owner combine + inv_rms return) no longer waits
    // on the gamma expand, which instead overlaps the wait for the other members.
#if defined(RMS_HAS_GAMMA) && defined(GS_RES_LOCAL)
    // gamma_stream_l1 GS_RES_LOCAL (sharded k == 1): every Tensix core expands its own resident slice locally,
    // chunk by chunk, from a resident row slab (no multicast of the 2 KB/tile expanded slice); compute waits
    // progressively. TILE: all chunks now (GS_SPLIT: even chunks; NCRISC does the odd ones). RM: prefetch
    // GS_RES_RM_PREFETCH chunks now, the rest after each chunk's sticks drain (compute blocks on cb_out_rm).
    const auto g_res_acc = TensorAccessor(gamma_args, gamma_addr, gamma_page_bytes);
    constexpr uint32_t gs_rc = GS_RES_CHUNK;
#ifdef RMS_RM
    const uint32_t gs_total = chunks_this * rm_chunk_w_tiles;
    constexpr uint32_t gs_first_n = GS_RES_RM_PREFETCH;
    constexpr uint32_t gs_step = gs_rc;
    constexpr uint32_t gs_parity = 2;
#else
    const uint32_t gs_total = x_row_pitch;
    constexpr uint32_t gs_first_n = 0xFFFFFFFF;
#ifdef GS_SPLIT
    constexpr uint32_t gs_step = 2 * gs_rc;
    constexpr uint32_t gs_parity = 0;
#else
    constexpr uint32_t gs_step = gs_rc;
    constexpr uint32_t gs_parity = 2;
#endif
#endif
    uint32_t gs_res_row;
    {
        MaybeDeviceZoneScope("gs_res_load");  // slab reads + the first (TILE: all own) chunks' local expand
        gs_res_row = gs_slab_acquire(g_res_acc, w_tile_start, core_w_tiles, col0, valid_cols, gs_parity, gs_rc);
        for (uint32_t i = 0, c0 = 0; i < gs_first_n && c0 < gs_total; ++i, c0 += gs_step) {
            gs_res_chunk<cb_gamma>(
                gs_res_row, core_w_tiles, col0, c0, (gs_total - c0) < gs_rc ? (gs_total - c0) : gs_rc);
        }
    }
    auto finish_resident_gamma = [](uint32_t) {};
#elif defined(RMS_HAS_GAMMA) && !defined(RMS_GAMMA_CHUNK)
    const auto g_res_acc = TensorAccessor(gamma_args, gamma_addr, gamma_page_bytes);
#ifdef RMS_RM
    const uint32_t gamma_push_tiles = chunks_this * rm_chunk_w_tiles;
#else
    const uint32_t gamma_push_tiles = core_w_tiles;
#endif
    GammaSliceLoad<decltype(g_res_acc)> gamma_res{
        g_res_acc, w_tile_start, core_w_tiles, gamma_push_tiles, col0, valid_cols};
    gamma_res.begin();
#ifdef RMS_GAMMA_MCAST
    // Multicast: receivers wait on the sender, so a late sender would stall its whole slice group — keep it early.
    constexpr bool gamma_late = false;
#else
    constexpr bool gamma_late = gamma_after_send && w_split_cores > 1;
#endif
    if constexpr (!gamma_late) {
        gamma_res.end();
    }
    auto finish_resident_gamma = [&](uint32_t blk) {
        if constexpr (gamma_late) {
            if (blk == 0) {
                gamma_res.end();
            }
        }
    };
#else
    auto finish_resident_gamma = [](uint32_t) {};
#endif

#ifndef RMS_SHARDED
    const auto out_acc = TensorAccessor(out_args, out_addr, out_page_bytes);
#endif

    [[maybe_unused]] ExchangeState xs{member_idx, 0, 0, 0, 0, 0, 0};
    if constexpr (w_split_cores > 1) {
        xs.sem_addr = get_semaphore(sem_id);
        xs.tile_bytes = get_tile_size(cb_partial_sumsq);
        if constexpr (combine_owner) {
            xs.sem_inv_addr = get_semaphore(sem_inv_id);
        }
    }

    [[maybe_unused]] uint32_t out_ring_pos = 0;  // TILE: cb_out page position of the read pointer
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t rows = (blk + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;

        if constexpr (w_split_cores > 1) {
            exchange_send(rows, xs);
            exchange_recv_partials(rows, xs);
            finish_resident_gamma(blk);
            exchange_recv_inv(rows, xs);
        }

#if defined(RMS_HAS_GAMMA) && defined(RMS_GAMMA_CHUNK)
        // Streamed gamma for this block. TILE (shard-backed cb_out: compute never blocks on output) — every chunk now.
        // RM — only the ring depth now, then chunk c + depth after chunk c's sticks are drained: compute blocks on
        // cb_out_rm until this kernel drains it, so pushing every chunk first would deadlock on the gamma ring.
        constexpr uint32_t gamma_chunk = RMS_GAMMA_CHUNK;
#ifdef RMS_RM
        constexpr uint32_t gamma_prefetch = RMS_GAMMA_STREAM_DEPTH;
#else
        constexpr uint32_t gamma_prefetch = 0xFFFFFFFF;
#endif
        {
            MaybeDeviceZoneScope("wr_gamma_stream");  // occupancy: chunk loads + back-pressure on the ring
#ifdef GS_SPLIT
            constexpr uint32_t gs_step = 2 * gamma_chunk;  // even chunks here; odd chunks on NCRISC (GS_NC_ROLE)
#else
            constexpr uint32_t gs_step = gamma_chunk;
#endif
            for (uint32_t i = 0, c0 = 0; i < gamma_prefetch && c0 < norm_w; ++i, c0 += gs_step) {
                stream_gamma_chunk<cb_gamma>(g_acc, w_tile_start, core_w_tiles, col0, valid_cols, c0, gs_row);
            }
        }
#endif
        // ---- store_block ----
        MaybeDeviceZoneScope("wr_store");  // occupancy: waits on pass 2 output + issue/flush of the writes
#if defined(RMS_SHARDED) && defined(RMS_RM)
        constexpr uint32_t chunk_row_bytes = rm_chunk_w_tiles * TILE_DIM * elem_bytes;
        for (uint32_t c = 0; c < chunks_this; ++c) {
            const uint32_t c0 = c * rm_chunk_w_tiles * TILE_DIM;
            const uint32_t cols =
                (valid_cols - c0) < rm_chunk_w_tiles * TILE_DIM ? (valid_cols - c0) : rm_chunk_w_tiles * TILE_DIM;
            const uint32_t bytes = cols * elem_bytes;
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t s0 = (blk * block_row_tiles + r) * TILE_DIM;
                const uint32_t nsticks = (valid_sticks - s0) < TILE_DIM ? (valid_sticks - s0) : TILE_DIM;
                cb_wait_front(cb_out_rm, rm_chunk_w_tiles);
                const uint32_t l1 = get_read_ptr(cb_out_rm);
                const uint32_t dst = out_addr + s0 * out_page_bytes + c0 * elem_bytes;
                for (uint32_t s = 0; s < nsticks; ++s) {
                    noc_async_write(l1 + s * chunk_row_bytes, get_noc_addr(dst + s * out_page_bytes), bytes);
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_out_rm, rm_chunk_w_tiles);
            }
#if defined(RMS_HAS_GAMMA) && defined(GS_RES_LOCAL)
            if (blk == 0 && c + GS_RES_RM_PREFETCH < chunks_this) {
                MaybeDeviceZoneScope("gs_res_refill");
                const uint32_t c0n = (c + GS_RES_RM_PREFETCH) * gs_rc;
                gs_res_chunk<cb_gamma>(gs_res_row, core_w_tiles, col0, c0n, gs_rc);
            }
#endif
#if defined(RMS_HAS_GAMMA) && defined(RMS_GAMMA_CHUNK)
            // Chunk c's gamma was popped before its sticks reached cb_out_rm: refill the ring with chunk c + depth
            // (loaded while compute normalizes chunk c + 1, whose gamma is already resident).
            if (c + RMS_GAMMA_STREAM_DEPTH < chunks_this) {
                stream_gamma_chunk<cb_gamma>(
                    g_acc,
                    w_tile_start,
                    core_w_tiles,
                    col0,
                    valid_cols,
                    (c + RMS_GAMMA_STREAM_DEPTH) * gamma_chunk,
                    gs_row);
            }
#endif
        }
#elif defined(RMS_SHARDED)
        cb_wait_front(cb_out, rows * x_row_pitch);
        cb_pop_front(cb_out, rows * x_row_pitch);
#elif defined(RMS_RM)
        constexpr uint32_t chunk_row_bytes = rm_chunk_w_tiles * TILE_DIM * elem_bytes;
        for (uint32_t c = 0; c < chunks_this; ++c) {
            const uint32_t chunk_tiles = (core_w_tiles - c * rm_chunk_w_tiles) < rm_chunk_w_tiles
                                             ? (core_w_tiles - c * rm_chunk_w_tiles)
                                             : rm_chunk_w_tiles;
            const uint32_t col0 = (w_tile_start + c * rm_chunk_w_tiles) * TILE_DIM;
            const uint32_t col_end_tiles = (w_tile_start + c * rm_chunk_w_tiles + chunk_tiles) * TILE_DIM;
            const uint32_t col_end = col_end_tiles < W ? col_end_tiles : W;
            const uint32_t bytes = (col_end - col0) * elem_bytes;
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t t = row_start + blk * block_row_tiles + r;
                const uint32_t image = t / image_row_tiles;
                const uint32_t tr = t - image * image_row_tiles;
                const uint32_t first_stick = image * H + tr * TILE_DIM;
                const uint32_t valid_rows = (H - tr * TILE_DIM) < TILE_DIM ? (H - tr * TILE_DIM) : TILE_DIM;
                if (bytes == chunk_row_bytes) {
                    dataflow_kernel_lib::write_sticks_after_untilize<cb_out_rm>(
                        out_acc, valid_rows, bytes, first_stick, col0 * elem_bytes);
                } else {
                    cb_wait_front(cb_out_rm, rm_chunk_w_tiles);
                    const uint32_t l1 = get_read_ptr(cb_out_rm);
                    for (uint32_t s = 0; s < valid_rows; ++s) {
                        noc_async_write(
                            l1 + s * chunk_row_bytes, out_acc.get_noc_addr(first_stick + s, col0 * elem_bytes), bytes);
                    }
                    noc_async_writes_flushed();
                    cb_pop_front(cb_out_rm, rm_chunk_w_tiles);
                }
            }
        }
#else
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t t = row_start + blk * block_row_tiles + r;
            const uint32_t first_tile = t * tensor_w_tiles + w_tile_start;
            for (uint32_t j0 = 0; j0 < core_w_tiles;) {
                uint32_t n = (core_w_tiles - j0) < out_unit_tiles ? (core_w_tiles - j0) : out_unit_tiles;
                n = (out_cb_pages - out_ring_pos) < n ? (out_cb_pages - out_ring_pos) : n;
                cb_wait_front(cb_out, n);
                const uint32_t l1 = get_read_ptr(cb_out);
                for (uint32_t i = 0; i < n; ++i) {
                    if constexpr (!ablate_dm) {
                        noc_async_write(
                            l1 + i * out_page_bytes, out_acc.get_noc_addr(first_tile + j0 + i), out_page_bytes);
                    }
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_out, n);
                j0 += n;
                out_ring_pos = (out_ring_pos + n == out_cb_pages) ? 0 : out_ring_pos + n;
            }
        }
#endif
    }
    noc_async_write_barrier();
    if constexpr (w_split_cores > 1) {
        noc_async_atomic_barrier();
    }
}
#endif  // GS_NC_ROLE
