// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm writer (BRISC / NoC1): GammaSliceLoad (once) + exchange_partials_block (k > 1) + store_block.
// Per block (k > 1): exchange_send -> exchange_recv_partials -> [block 0: finish the gamma load] -> exchange_recv_inv.
//
// GammaSliceLoad — this member's W slice of gamma into cb_gamma. Compute multiplies by gamma ROW-broadcast
//   (MulGammaRowBcast), so a cb_gamma page only needs row 0 of faces 0/1 = gamma[32j .. 32j+31]: pages are
//   [bfp exponent section +] faces 0/1 (half a tile), nothing is expanded. Loaded here, not on the reader,
//   so the x stream starts at once on every Tensix core; compute waits for it only before its first pass 2.
//   RM gamma  : 64 B-aligned slice (lead == 0): per tile ONE DRAM read of the 32-datum row into face 0 rows
//               0..1, then one local L1 copy of its upper half (face 0 row 1) into face 1 row 0 (a DRAM read keeps
//               its source offset mod 64 B, so a bf16 face-1 half cannot land at face 1 directly). Unaligned slice
//               (RM shards are 16 B granular): one contiguous read + RISC-V scatter of the 2 row-0 pieces per tile
//               (descending j, so the staging bytes are consumed before overwritten).
//   TILE gamma: per tile, face 0 row 0 and face 1 row 0 straight from DRAM (64 B reads; a bfloat8_b tile's
//               64 B exponent section rides along with the first read — row 0's exponents are already right).
//   Interleaved / sharded k = 1 (RMS_GAMMA_MCAST): read once per W slice and multicast to the slice's
//               group.
//   Streamed (RMS_GAMMA_CHUNK = G, fallback when the resident slice does not fit L1): per block, the slice is
//               re-loaded in G-tile chunks through a 2-chunk ring (always G pages pushed, so a chunk never wraps
//               the ring); pass 2 consumes one chunk per G-tile window.
//
// Gamma timing (Refinement 5): k == 1 / multicast gamma / blocks > GAMMA_AFTER_SEND_MAX_ROWS tile-rows — the load
//   runs at kernel start. Per-core load with short blocks (gamma_after_send) — the DRAM reads are issued at kernel
//   start; the barrier [+ RM face-1 copy] runs only after block 0's partials are out and the owner duties are done.
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
// Combine pipeline (combine_pipeline: sharded TILE owner combine, shard cut into sub-blocks — compute runs
//   front(t+1), combine(t), apply(t-1)): a two-round exchange window. Sub-block i uses the partial / inv semaphore
//   pair of parity i & 1 (each monotonic) and row owners rotate per sub-block (row r of i owned by (i*s + r) % k).
//   Order: send(0), W1(0); per i: send(i+1), W3(i) (return inv_rms), W1(i+1) (wait partials), W4(i) (wait inv_rms).
//   send(i+2) follows W4(i), i.e. every owner finished W1(i) / combine(i): a parity's counts never mix blocks i and
//   i+2 and the 2-block cb_gathered ring slot of i is free before i+2 lands; an owner's W3(i+2) needs this member's
//   send(i+2), so inv counts of i and i+2 never mix either. The single-owner inv_rms multicast is off on this path
//   (a rotated one-tile-row sub-block's owner need not be member 0). cb_out (the output shard) is waited once at
//   the end.
// Payload (PARTIAL_FACES_ONLY): a partial / inv_rms tile is meaningful in column 0 only, which lives in faces
//   0 and 2 — ship those two faces, not the whole tile (the combine's Skip reduce and the Col-broadcast
//   normalize never read faces 1/3).
// store_block
//   TILE sharded: cb_out IS the output shard (zero-copy) — compute packs straight into it; the writer only
//         waits for the block's pages so the program cannot retire before the shard is written.
//   RM sharded: per (chunk, tile-row) the untilized stick pieces are written back to the local output shard
//         (one write per tile-row when the shard pitch equals the staging pitch, else stateful per-stick writes).
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
#if defined(RMS_GAMMA_MCAST) || defined(RMS_INV_MCAST)
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#endif
#include "rms_norm_common.hpp"

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
constexpr uint32_t out_cb_pages = get_compile_time_arg_val(10);  // cb_out capacity (TILE: out_depth * out_unit)
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
constexpr uint32_t gamma_l1_page_bytes = get_compile_time_arg_val(22);  // cb_gamma page (faces 0/1 of a gamma tile)
constexpr bool combine_pipeline = get_compile_time_arg_val(23) != 0;
constexpr uint32_t sem_par_id = get_compile_time_arg_val(24);      // combine pipeline: odd-sub-block partials semaphore
constexpr uint32_t sem_inv_par_id = get_compile_time_arg_val(25);  // combine pipeline: odd-sub-block inv_rms semaphore
constexpr auto out_args = TensorAccessorArgs<26>();
[[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;

#ifdef RMS_HAS_GAMMA
// L1 stride of one cb_gamma page: [exponent section +] faces 0/1 (the row-broadcast unpack reads nothing else).
constexpr uint32_t g_tile_bytes = gamma_l1_page_bytes;
static_assert(
    gamma_l1_page_bytes == gamma_exp_bytes + 2 * FACE_DIM * FACE_DIM * gamma_elem_bytes, "cb_gamma page = faces 0/1");

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
        if (lead == 0) {
            // one DRAM read of tile j's 32-datum row into face 0 rows 0..1 (finish moves the face-1 half). The last
            // tile may read past W: its padding columns are don't-care output.
            constexpr uint32_t row_bytes = TILE_DIM * gamma_elem_bytes;
            for (uint32_t j = 0; j < core_w_tiles; ++j) {
                noc_async_read(g_acc.get_noc_addr(0, byte0 + j * row_bytes), base + j * g_tile_bytes, row_bytes);
            }
            return;
        }
        noc_async_read(g_acc.get_noc_addr(0, byte0 - lead), base, lead + gamma_cols * gamma_elem_bytes);
    }
}

// finish_gamma_slice: wait for the row-0 reads; RM: put row 0's face-1 half in face 1 (local copy / scatter).
FORCE_INLINE void finish_gamma_slice(uint32_t base, uint32_t core_w_tiles, uint32_t gamma_col0) {
    noc_async_read_barrier();
    if constexpr (!gamma_is_tile) {
        constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * gamma_elem_bytes;
        constexpr uint32_t words_per_face_row = FACE_DIM * gamma_elem_bytes / sizeof(uint32_t);
        constexpr uint32_t frow = FACE_DIM * gamma_elem_bytes;
        const uint32_t lead = gamma_rm_lead(gamma_col0);
        if (lead == 0) {
            for (uint32_t j = 0; j < core_w_tiles; ++j) {  // face 0 row 1 (the face-1 half of row 0) -> face 1 row 0
                const uint32_t tile = base + j * g_tile_bytes;
                noc_async_read(get_noc_addr(tile + frow), tile + face_bytes, frow);
            }
            noc_async_read_barrier();
            return;
        }
        // Scatter row 0: datums [32j, 32j+16) -> tile j face 0 row 0, [32j+16, 32j+32) -> face 1 row 0.
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
}

// Fill `base` (cb_gamma's write pointer) with this member's gamma slice; the caller owns the CB
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
// GammaSliceLoad: this member's gamma slice into cb_gamma (program lifetime; compute first needs it
// at pass 2, so this overlaps the reader's x stream and pass 1).
//   RMS_GAMMA_MCAST (interleaved): the Tensix cores sharing a W slice form one multicast group (host Mcast);
//     its sender reads the slice once and multicasts the faces-0/1 pages to the group — the slice is
//     read from DRAM once per W slice instead of once per Tensix core. One-shot (no handshake): cb_gamma is
//     written once per launch and nothing else touches its L1.
//   otherwise: every Tensix core reads its own slice.
//   Split in two so the DRAM reads can be in flight while the writer serves the combine:
//   begin() reserves cb_gamma and issues this core's row-0 reads (per-core load, or the multicast sender);
//   end() waits for them, finishes RM rows, multicasts (sender) / receives (receiver) and pushes. load() = both.
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
        MaybeDeviceZoneScope("wr_gamma_finish");  // barrier [+ RM face-1 copy] (+ multicast send / receive wait)
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

#ifdef RMS_GAMMA_CHUNK
// stream_gamma_chunk: gamma tiles [c0, c0 + G) of this member's slice into the next G pages of the cb_gamma ring
// (always G pages pushed, so a chunk never wraps the ring). Gap tiles past core_w_tiles (a ragged shard column's
// padding / the RM chunk pitch) are pushed unwritten — their output is don't-care.
template <typename Accessor>
FORCE_INLINE void stream_gamma_chunk(
    const Accessor& g_acc,
    uint32_t w_tile_start,
    uint32_t core_w_tiles,
    uint32_t col0,
    uint32_t valid_cols,
    uint32_t c0) {
    constexpr uint32_t G = RMS_GAMMA_CHUNK;
    cb_reserve_back(cb_gamma, G);
    if (c0 < core_w_tiles) {
        const uint32_t valid = (core_w_tiles - c0) < G ? (core_w_tiles - c0) : G;
        const uint32_t cols =
            (valid_cols - c0 * TILE_DIM) < valid * TILE_DIM ? (valid_cols - c0 * TILE_DIM) : valid * TILE_DIM;
        fill_gamma_slice(g_acc, get_write_ptr(cb_gamma), w_tile_start + c0, valid, col0 + c0 * TILE_DIM, cols);
    }
    cb_push_back(cb_gamma, G);
}
#endif

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
    uint32_t ofs = 0;       // owner rotation of the receive block (row r owned by member (r + ofs) % k)
    uint32_t send_ofs = 0;  // owner rotation of the block being sent
};

// First owned row of member m under rotation ofs: (m - ofs) mod k.
FORCE_INLINE uint32_t first_owned_row(uint32_t member_idx, uint32_t ofs) {
    constexpr uint32_t k = w_split_cores;
    return (member_idx + k - ofs) % k;
}

FORCE_INLINE uint32_t owned_rows(uint32_t rows, uint32_t member_idx, uint32_t ofs = 0) {
    constexpr uint32_t k = w_split_cores;
    const uint32_t r0 = first_owned_row(member_idx, ofs);
    return r0 < rows ? (rows - r0 + k - 1) / k : 0;
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
        const uint32_t ofs = st.send_ofs;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t owner = (r + ofs) % k;
            const uint32_t j = (r + ofs) / k - (owner < ofs ? 1 : 0);  // index among the owner's rows (ofs 0: r / k)
            send_acc_tile(src + r * tb, peer_addr(owner, st.slot_base + (j * k + st.member_idx) * tb), tb);
        }
        fence_before_signal();
        // Row r < min(rows, k) is the first row of its owner (r + ofs) % k, which owns (rows - r + k - 1) / k rows.
        const uint32_t n_owners = rows < k ? rows : k;
        for (uint32_t r = 0; r < n_owners; ++r) {
            noc_semaphore_inc(peer_addr((r + ofs) % k, st.sem_addr), (rows - r + k - 1) / k);
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

// inv_rms return of a single-owner block (RMS_INV_MCAST, Perf 1): a block of ONE tile-row has one owner, member 0,
// which returns inv_rms to its k members with one multicast (kernel_lib Mcast, fixed sender = member 0, Flag signal,
// no handshake, loopback to itself) instead of k unicasts + k semaphore increments: BH p150b, WIDTH k = 28 / 32 decode
// 8.9 -> 7.6 us / 8.2 -> 6.6 us. Blocks with >= 2 owners keep the unicast return: several owners multicasting into
// the same members at once serialize (measured 2.1x slower on BLOCK 7168x1024, perf_experiments/inv_mcast_return).
// Slot reuse is safe without a handshake: a member sends its partial of block b+1 only after it consumed inv(b).
#ifdef RMS_INV_MCAST
constexpr auto inv_mcast = dataflow_kernel_lib::McastArgs<
    get_named_compile_time_arg_val("inv_mcast_ct_offset"),
    get_named_compile_time_arg_val("inv_mcast_rt_offset")>();
constexpr bool inv_mcast_on = true;
#else
constexpr bool inv_mcast_on = false;
#endif
static_assert(!(combine_pipeline && inv_mcast_on), "combine pipeline: rotated owners - inv_rms multicast must be off");

// W1 (owner topology): wait for this member's owned partials of the block, publish them to compute.
FORCE_INLINE void owner_wait_partials(uint32_t rows, ExchangeState& st) {
    constexpr uint32_t k = w_split_cores;
    st.partials_expected += owned_rows(rows, st.member_idx, st.ofs) * k;
    {
        MaybeDeviceZoneScope("wr_wait_peers");  // rendezvous: the slowest member's partial
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st.sem_addr), st.partials_expected);
    }
    cb_push_back(cb_gathered_sumsq, ((rows + k - 1) / k) * k);
}

// W3 (owner topology): return the combined inv_rms of each owned row r into slot r of every member's cb_inv_rms.
FORCE_INLINE void owner_return_inv(uint32_t rows, ExchangeState& st) {
    constexpr uint32_t k = w_split_cores;
    const uint32_t tb = st.tile_bytes;
    const uint32_t r0 = first_owned_row(st.member_idx, st.ofs);
    const uint32_t owned = owned_rows(rows, st.member_idx, st.ofs);
    MaybeDeviceZoneScope("wr_inv_return");  // occupancy: waits on compute's owner combine + k-way return
    cb_reserve_back(cb_inv_rms, rows);
    const uint32_t inv_base = get_write_ptr(cb_inv_rms);
#ifdef RMS_INV_MCAST
    if (rows == 1) {      // never rotated (static_assert above): member 0 owns the block's only row
        if (owned > 0) {  // member 0: the block's only owner
            cb_wait_front(cb_inv_owned, 1);
            Noc noc;
            // faces 0..2 are contiguous (face 1 is don't-care): one send covers column 0. optional_sender: a writer
            // kernel group holding no member 0 (reversed NoC pairing on the DRAM-adjacent columns) has no sender
            // pipe compiled in; member 0 always sits in the group whose writer NoC is the channel's.
            if (auto sender = inv_mcast.optional_sender(noc)) {
                sender->send(get_read_ptr(cb_inv_owned), inv_base, partial_faces_only ? 3 * (tb / 4) : tb);
            }
            cb_pop_front(cb_inv_owned, 1);
        }
        return;
    }
#endif
    if (owned > 0) {
        // One owned tile at a time from the CB front: members own different row counts per block, so the
        // read pointer is not block-aligned and a multi-tile linear walk could cross the ring end.
        for (uint32_t j = 0; j < owned; ++j) {
            const uint32_t r = r0 + j * k;
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
}

// exchange receive, part 1: wait for this member's gathered partials -> cb_gathered_sumsq (compute combines);
//   owner topology: then return the combined inv_rms of each owned row r into slot r of every member's cb_inv_rms.
FORCE_INLINE void exchange_recv_partials(uint32_t rows, ExchangeState& st) {
    constexpr uint32_t k = w_split_cores;
    if constexpr (combine_owner) {
        owner_wait_partials(rows, st);
        owner_return_inv(rows, st);
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
        MaybeDeviceZoneScope("wr_wait_inv");  // rendezvous: every owner's inv_rms return
        if (inv_mcast_on && rows == 1) {
#ifdef RMS_INV_MCAST
            // member 0's multicast of the block's only row (Flag). The sender itself does not wait: its loopback
            // copy is fenced (write barrier) inside send().
            Noc noc;
            if (auto receiver = inv_mcast.optional_receiver(noc); receiver && !inv_mcast.can_send()) {
                receiver->receive();
            }
#endif
        } else {
            st.inv_expected += rows;  // the unicast return's semaphore counts only unicast-returned rows
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st.sem_inv_addr), st.inv_expected);
        }
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
#endif
    // Resident gamma slice (once per program). k == 1, multicast gamma, or !gamma_after_send: loaded at kernel start,
    // overlapping the reader's x stream and pass 1. Per-core load with k > 1 and gamma_after_send: only the DRAM reads
    // are issued at kernel start;
    // the barrier [+ RM face-1 copy] (+ multicast) run after block 0's partials have left (symmetric) / after this
    // member's owner duties — the combine's critical path (partials out, owner combine + inv_rms return) no longer
    // waits on the gamma finish, which instead overlaps the wait for the other members.
#if defined(RMS_HAS_GAMMA) && !defined(RMS_GAMMA_CHUNK)
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

#if defined(RMS_SHARDED) && !defined(RMS_RM) && !defined(RMS_GAMMA_CHUNK)
    if constexpr (combine_pipeline && w_split_cores > 1 && combine_owner) {
        // Combine pipeline: two-round exchange window (see the header). Parity p = i & 1 selects the semaphore
        // pair and its expected counts.
        auto rows_of = [&](uint32_t i) { return (i + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles; };
        auto ofs_of = [&](uint32_t i) { return (i * block_row_tiles) % w_split_cores; };
        const uint32_t sem_p[2] = {xs.sem_addr, get_semaphore(sem_par_id)};
        const uint32_t sem_inv_p[2] = {xs.sem_inv_addr, get_semaphore(sem_inv_par_id)};
        uint32_t pexp[2] = {0, 0};
        uint32_t iexp[2] = {0, 0};
        auto par = [&](uint32_t i, auto&& fn) {
            const uint32_t p = i & 1;
            xs.sem_addr = sem_p[p];
            xs.sem_inv_addr = sem_inv_p[p];
            xs.partials_expected = pexp[p];
            xs.inv_expected = iexp[p];
            fn();
            pexp[p] = xs.partials_expected;
            iexp[p] = xs.inv_expected;
        };
        auto send = [&](uint32_t i) {
            par(i, [&] {
                xs.send_ofs = ofs_of(i);
                exchange_send(rows_of(i), xs);
            });
        };
        auto wait_partials = [&](uint32_t i) {
            par(i, [&] {
                xs.ofs = ofs_of(i);
                owner_wait_partials(rows_of(i), xs);
            });
        };
        send(0);
        wait_partials(0);
        for (uint32_t i = 0; i < num_blocks; ++i) {
            if (i + 1 < num_blocks) {
                send(i + 1);
            }
            par(i, [&] {
                xs.ofs = ofs_of(i);
                owner_return_inv(rows_of(i), xs);
            });
            finish_resident_gamma(i);  // gamma_after_send: after sub-block 0's owner duties (no-op otherwise)
            if (i + 1 < num_blocks) {
                wait_partials(i + 1);
            }
            par(i, [&] { exchange_recv_inv(rows_of(i), xs); });
        }
        MaybeDeviceZoneScope("wr_store");  // cb_out IS the output shard: wait until compute has written all of it
        const uint32_t shard_pages = ((num_blocks - 1) * block_row_tiles + last_block_row_tiles) * x_row_pitch;
        cb_wait_front(cb_out, shard_pages);
        cb_pop_front(cb_out, shard_pages);
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        return;
    }
#endif
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
            for (uint32_t i = 0, c0 = 0; i < gamma_prefetch && c0 < norm_w; ++i, c0 += gamma_chunk) {
                stream_gamma_chunk(g_acc, w_tile_start, core_w_tiles, col0, valid_cols, c0);
            }
        }
#endif
        // ---- store_block ----
        MaybeDeviceZoneScope("wr_store");  // occupancy: waits on pass 2 output + issue/flush of the writes
#if defined(RMS_SHARDED) && defined(RMS_RM)
        constexpr uint32_t chunk_row_bytes = rm_chunk_w_tiles * TILE_DIM * elem_bytes;
        static_assert(chunk_row_bytes <= NOC_MAX_BURST_SIZE, "a stick piece must be one NoC packet");
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
                if (out_page_bytes == chunk_row_bytes && bytes == chunk_row_bytes) {
                    // Shard pitch == staging pitch: the tile-row's sticks are one contiguous run.
                    noc_async_write(l1, get_noc_addr(dst), nsticks * chunk_row_bytes);
                } else {
                    // Stateful one-packet writes: NoC coordinates + length programmed once per tile-row.
                    noc_async_write_one_packet_set_state(get_noc_addr(dst), bytes);
                    for (uint32_t s = 0; s < nsticks; ++s) {
                        noc_async_write_one_packet_with_state(l1 + s * chunk_row_bytes, dst + s * out_page_bytes);
                    }
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_out_rm, rm_chunk_w_tiles);
            }
#if defined(RMS_HAS_GAMMA) && defined(RMS_GAMMA_CHUNK)
            // Chunk c's gamma was popped before its sticks reached cb_out_rm: refill the ring with chunk c + depth
            // (loaded while compute normalizes chunk c + 1, whose gamma is already resident).
            if (c + RMS_GAMMA_STREAM_DEPTH < chunks_this) {
                stream_gamma_chunk(
                    g_acc, w_tile_start, core_w_tiles, col0, valid_cols, (c + RMS_GAMMA_STREAM_DEPTH) * gamma_chunk);
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
