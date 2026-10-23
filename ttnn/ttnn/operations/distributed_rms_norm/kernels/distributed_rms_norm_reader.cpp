// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm reader (compute cores, NCRISC / NoC0) — op_design.md -> Block schedule / Dataflow Strategy.
//
// Prologue: cb_reduce_scaler <- 1.0 (calculate_and_prepare_reduce_scaler<SUM, REDUCE_ROW>); the pad tiles [V, WS) of
// every tile-row slot of the x ring are zeroed once (a ragged R3-X slice), so the square pass sees zeros there.
// Per block (= wave) w of R_b tile-rows (R_b = 0 for a group with fewer rows: only the root's semaphore handshake
// with its ports runs, so every port's per-wave thresholds stay uniform):
//   load_block            interleaved: R_b tile-rows of this core's W-slice -> cb_x_tiles (WS pages per tile-row, one
//                         barrier). Sharded (X_SHARD): cb_x_tiles IS the input shard (zero-copy) — the block's
//                         R_b * WS resident pages are handed to compute with no data movement (the core's own shard
//                         is never read through a TensorAccessor)
//                         W-pad (W_local % 32 != 0, the core whose slice holds tile Wt-1): columns [w_last_valid, 32)
//                         of that tile of every block row are zeroed in L1 before the push (interleaved: after the read
//                         barrier; sharded: in the resident shard, whose tile padding is the tensor's own and not
//                         guaranteed zero), so pad columns never enter Σx²
//   Ownership: the block's rows are split over the group's OWNERS roots (1 = a single root, every R1-X core and every
//   interleaved R3-X row-group; > 1 = R3-RS, WIDTH_SHARDED row-groups): root k owns rows [k*q, k*q + n_k),
//   q = ceil(R_b / OWNERS) (host owner_rows). "root" below = an owner with n_k > 0; an owner with none still counts
//   at its ports.
//   core_combine_block    R3-X: every member writes its partials of owner k's rows (faces 0/2) into slots
//                         [my_index*q, +n_k) of k's cb_gather region (block % gather_depth) and increments k's gather
//                         semaphore of the block's parity; an owner waits for the G_w - 1 others, pushes G_w * q pages
//                         (row stride q)
//   publish_stats_block   root: column 0 of each of its device-partial tiles -> the 32 contiguous wire words at the
//                         base of its own slot page (t*G + p) of cb_stats_slots (no_col0_expand: the wire form = tile
//                         face 0 rows 0-1 = compute's form, 128 B); the words are NoC-written into landing[w][p] (at
//                         the root's offset) of each sending port of the root's link, then SEM_OWN_STATS on those ports
//                         is incremented. Written into stats-ring region (published % stats_depth)
//   gather_slots_block    root: each notifying port signals its own landed semaphore once per landed entry (lamp L4);
//                         as soon as entry i of wave w is in (semaphore >= w * n_in + i + 1) its slot's 128 B rows are
//                         read from that port straight to the page bases (no barrier: reads of later entries are
//                         issued as they land); no RISC-V expansion (compute consumes the wire form: transpose_dest +
//                         one SFPU lane move put inv in column 0); once all G - 1 are in, push the whole G * Q region
//   slot forward (FWD_INV: single-root R3-X, Perf 1 replicated_finalize) instead of an inv distribution: the root
//                         forwards the RAW 128 B slot rows — its own slot at publish, each peer slot as soon as its
//                         read completed — to the same cb_stats_slots page bases of every member, as LINKED multicasts
//                         per member rectangle closed by a multicast semaphore SET (SEM_OWN_FWD = blocks published;
//                         SEM_SLOT_FWD = (block + 1) << 24 | mask of forwarded peer slots). Every member reserves its
//                         region, waits for every slot's bit, pushes G * Q, and its compute finalizes its own
//                         inv (the same fixed g order: inv is bit-identical group-wide). No inv crosses the NoC
//   distribute_inv_block  R3-RS (several owners): owner k copies its finalized rows' column 0 into its own cb_inv_rms
//                         pages [k*q, +n_k) (+ the compact 128 B tail) and unicasts the tails to the same pages of
//                         every other member + SEM_INV. Every member reserves R_b, waits, expands the other owners'
//                         tails to column 0, pushes R_b
// Schedule: front(w) = load + combine + publish (+ own-slot forward), back(w) = gather_slots (root: + slot forward;
// FWD_INV member: forwarded-slot gather) + distribute_inv (R3-RS). Back to back
// (front(w), back(w)) or, with the WAVE_PIPELINE look-ahead (stats_depth = 2, num_waves > 1), front(0) then
// [front(w+1), back(w)]: wave w+1's DRAM reads / combine / publish run while wave w's stats are on the Fabric.
//
// Raw NoC / CB calls (op_design.md -> "Helpers considered and rejected"): no kernel_lib helper reads TILE pages, moves
// 32 column-0 words between a tile and a 128 B vector, or gathers tiles to a root.
// The slot forward (FWD_INV) is raw noc_async_write_multicast(linked) + noc_semaphore_set_multicast, bypassing the
// kernel_lib McastFamily (Perf 1, measured: landed -> inv on every member 3.42 / 3.23 -> 2.78 / 2.57 us per wave,
// overlapping waves 12.18 -> 10.67 us bench): Counter data-ready pays a write barrier before its atomic per round,
// and Flag data-ready only writes a fixed VALID word, so neither can carry a growing per-block slot mask that closes
// a linked data chain. Do not "fix" this back to McastFamily without re-measuring.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "distributed_rms_norm_combine.hpp"
#include "tools/profiler/kernel_profiler.hpp"

namespace {

constexpr uint32_t cb_x_tiles = 0;
constexpr uint32_t cb_reduce_scaler = 1;
constexpr uint32_t cb_partial_col = 3;
constexpr uint32_t cb_gather = 4;
constexpr uint32_t cb_device_partial = 5;
constexpr uint32_t cb_stats_slots = 6;
constexpr uint32_t cb_inv_root = 7;
constexpr uint32_t cb_inv_rms = 8;

constexpr uint32_t FACE_ELEMS = 16 * 16;
constexpr uint32_t TAIL_WORD = 1024 - 32;  // face 3, rows 30-31: the compact wire form of a stats page
constexpr uint32_t ROW_BYTES = 32 * 4;

// Word index of element row i, column 0, of a Float32 tile (faces 0 / 2 hold column 0).
constexpr uint32_t col0_word(uint32_t i) { return i < 16 ? 16 * i : 512 + 16 * (i - 16); }

// Word moves between a tile's column 0 and the 32-word compact tail of a page (the reader's own CB pages). Loads are
// plain (the caller runs l1_acquire() first: L1 fence + compiler barrier) and batched 8 at a time into registers before
// the stores, so the in-order core overlaps the load latencies instead of stalling once per word; stores are volatile
// so they reach L1 before any NoC read of them.
template <uint32_t i0>
FORCE_INLINE void col0_batch(const uint32_t* s, volatile tt_l1_ptr uint32_t* d, bool tail_too) {
    uint32_t v[8];
#pragma GCC unroll 8
    for (uint32_t k = 0; k < 8; ++k) {
        v[k] = s[col0_word(i0 + k)];
    }
#pragma GCC unroll 8
    for (uint32_t k = 0; k < 8; ++k) {
        d[col0_word(i0 + k)] = v[k];
    }
    if (tail_too) {
#pragma GCC unroll 8
        for (uint32_t k = 0; k < 8; ++k) {
            d[TAIL_WORD + i0 + k] = v[k];
        }
    }
}

// Column 0 of tile `src` -> tile positions + compact tail of page `dst`.
FORCE_INLINE void col0_to_page(uint32_t src, uint32_t dst) {
    const uint32_t* s = reinterpret_cast<const uint32_t*>(src);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    col0_batch<0>(s, d, true);
    col0_batch<8>(s, d, true);
    col0_batch<16>(s, d, true);
    col0_batch<24>(s, d, true);
}

// no_col0_expand: the stats slots' wire form IS their compute form — 32 contiguous fp32 words at the page base (tile
// face 0 rows 0-1: element i at row i / 16, column i % 16). Compute sums / finalizes those two SFPU vectors and moves
// them to column 0 itself (transpose_dest + one SFPU lane move), so no RISC-V expansion runs after a slot lands.
// Column 0 of tile `src` -> the 32 contiguous wire words at page `dst` (the root's own slot, publish).
template <uint32_t i0>
FORCE_INLINE void wire_batch(const uint32_t* s, volatile tt_l1_ptr uint32_t* d) {
    uint32_t v[8];
#pragma GCC unroll 8
    for (uint32_t k = 0; k < 8; ++k) {
        v[k] = s[col0_word(i0 + k)];
    }
#pragma GCC unroll 8
    for (uint32_t k = 0; k < 8; ++k) {
        d[i0 + k] = v[k];
    }
}
FORCE_INLINE void col0_to_wire(uint32_t src, uint32_t dst) {
    const uint32_t* s = reinterpret_cast<const uint32_t*>(src);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    wire_batch<0>(s, d);
    wire_batch<8>(s, d);
    wire_batch<16>(s, d);
    wire_batch<24>(s, d);
}

template <uint32_t i0>
FORCE_INLINE void tail_batch(const uint32_t* s, volatile tt_l1_ptr uint32_t* d) {
    uint32_t v[8];
#pragma GCC unroll 8
    for (uint32_t k = 0; k < 8; ++k) {
        v[k] = s[i0 + k];
    }
#pragma GCC unroll 8
    for (uint32_t k = 0; k < 8; ++k) {
        d[col0_word(i0 + k)] = v[k];
    }
}

// Compact tail of page `page` (landed over the NoC) -> its column-0 tile positions.
FORCE_INLINE void tail_to_col0(uint32_t page) {
    const uint32_t* s = reinterpret_cast<const uint32_t*>(page) + TAIL_WORD;
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page);
    tail_batch<0>(s, d);
    tail_batch<8>(s, d);
    tail_batch<16>(s, d);
    tail_batch<24>(s, d);
}

// Element-sized zero stores over [addr, addr + n_elems * es) (partial face rows, <= 15 elements; base op rms_norm).
// es = 1: a Bfp8_b mantissa byte (sign + 7-bit mantissa, no implicit one) — a zero byte is an exact 0 whatever the
// row's shared exponent.
template <uint32_t es>
FORCE_INLINE void store_zeros(uint32_t addr, uint32_t n_elems) {
    if constexpr (es == 4) {
        volatile tt_l1_ptr uint32_t* q = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
        for (uint32_t i = 0; i < n_elems; ++i) {
            q[i] = 0;
        }
    } else if constexpr (es == 1) {
        volatile tt_l1_ptr uint8_t* q = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(addr);
        for (uint32_t i = 0; i < n_elems; ++i) {
            q[i] = 0;
        }
    } else {
        volatile tt_l1_ptr uint16_t* q = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
        for (uint32_t i = 0; i < n_elems; ++i) {
            q[i] = 0;
        }
    }
}

// Zero columns [w_valid, 32) of the TILE-layout tile at byte `tile_offset` of `cb`'s reserved window (faces 0/1 = rows
// 0-15 cols 0-15 / 16-31, faces 2/3 = rows 16-31; base op rms_norm). Whole pad faces go through the NoC zero loopback,
// partial face rows through element stores. The caller issues write_zeros_l1_barrier() before the push.
// data_off: bytes ahead of the element data (Bfp8_b: the 64 shared-exponent bytes; 0 for Float32 / Float16_b) — only
// the elements (mantissas) are zeroed, the exponents are left as read.
template <uint32_t es, uint32_t data_off>
FORCE_INLINE void zero_tile_w_padding(Noc& noc, CircularBuffer& cb, uint32_t tile_offset, uint32_t w_valid) {
    constexpr uint32_t face_bytes = FACE_ELEMS * es;
    constexpr uint32_t face_row_bytes = 16 * es;
    tile_offset += data_off;
    const uint32_t tile_addr = cb.get_write_ptr() + tile_offset;
    uint32_t c0 = w_valid;  // first pad column inside the partial faces (0..15)
    uint32_t f0 = 0;        // first face of the partial pair (0: faces 0/2, 1: faces 1/3)
    if (w_valid <= 16) {
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = tile_offset + 1 * face_bytes});
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = tile_offset + 3 * face_bytes});
    } else {
        c0 = w_valid - 16;
        f0 = 1;
    }
    if (c0 < 16) {
        for (uint32_t f = f0; f < 4; f += 2) {
            for (uint32_t r = 0; r < 16; ++r) {
                store_zeros<es>(tile_addr + f * face_bytes + r * face_row_bytes + c0 * es, 16 - c0);
            }
        }
    }
}

FORCE_INLINE uint64_t noc_xy(uint32_t packed, uint32_t addr) {
    return get_noc_addr(packed >> 16, packed & 0xFFFF, addr);
}

// Acquire L1 bytes another agent (NoC read / remote write / compute pack) just produced, before the plain
// (non-volatile) loads of col0_batch / tail_batch read them. invalidate_l1_cache() is a bare `asm("fence")` on
// Blackhole: a HARDWARE fence but NOT a compiler barrier (no "memory" clobber), and the NoC barrier / semaphore waits
// only order volatile accesses. Without the clobber the compiler may hoist the plain loads above the read barrier and
// read the page before the NoC read lands: measured as stale cross-device partials (the previous call's, or zeros on a
// fresh landing) in wave 0 of R1 with num_waves = 2 (~40 of 108 rows on the p = 1 devices, every call).
FORCE_INLINE void l1_acquire() {
    invalidate_l1_cache();
    asm volatile("" ::: "memory");
}

// distribute_inv_block's send: the compact tails of this owner's my_rows finalized rows (pages at `own`) to the same
// pages of every other member, barrier, then one SEM_INV increment per member. The rotated order (members
// my_index+1 .. G_w-1, then 0 .. my_index-1: drms_combine::rotated) is walked as two contiguous runs of the member
// list, and Q == 1 (one row per owner) is a compile-time single write: the G_w - 1 iteration loop is a pointer walk
// (55 members: 3.28 us; with a per-member rotated() and a runtime row loop it measured 3.77 us).
template <uint32_t G_w, uint32_t T32, uint32_t Q>
FORCE_INLINE void send_inv_tails(
    uint32_t own, uint32_t my_rows, uint32_t my_index, uint32_t member_idx, uint32_t inv_sem_addr) {
    const uint32_t rows = Q == 1 ? 1 : my_rows;
    auto send_to = [&](uint32_t k) {
        const uint32_t m = get_arg_val<uint32_t>(member_idx + k);
        for (uint32_t t = 0; t < rows; ++t) {
            const uint32_t tail = own + t * T32 + TAIL_WORD * 4;
            noc_async_write(tail, noc_xy(m, tail), ROW_BYTES);
        }
    };
    for (uint32_t k = my_index + 1; k < G_w; ++k) {
        send_to(k);
    }
    for (uint32_t k = 0; k < my_index; ++k) {
        send_to(k);
    }
    noc_async_write_barrier();
    for (uint32_t k = my_index + 1; k < G_w; ++k) {
        noc_semaphore_inc(noc_xy(get_arg_val<uint32_t>(member_idx + k), inv_sem_addr), 1);
    }
    for (uint32_t k = 0; k < my_index; ++k) {
        noc_semaphore_inc(noc_xy(get_arg_val<uint32_t>(member_idx + k), inv_sem_addr), 1);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t num_waves = get_compile_time_arg_val(1);
    constexpr uint32_t WS = get_compile_time_arg_val(2);
    constexpr uint32_t G = get_compile_time_arg_val(6);
    constexpr uint32_t G_w = get_compile_time_arg_val(7);
    constexpr bool r3 = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t x_depth = get_compile_time_arg_val(10);
    constexpr uint32_t gather_depth = get_compile_time_arg_val(11);
    constexpr uint32_t sem_gather_id = get_compile_time_arg_val(12);
    constexpr uint32_t sem_inv_id = get_compile_time_arg_val(13);
    constexpr uint32_t sem_own_stats_id = get_compile_time_arg_val(15);
    constexpr uint32_t sem_counted_id = get_compile_time_arg_val(17);
    constexpr bool x_shard = get_compile_time_arg_val(18) != 0;
    constexpr uint32_t OWNERS = get_compile_time_arg_val(21);  // R3-RS owners (1 = a single root)
    constexpr uint32_t Q = R / OWNERS;                         // rows an owner holds per full block (OWNERS | R)
    constexpr uint32_t w_last_valid = get_compile_time_arg_val(22);  // valid columns of tile Wt-1 (32: W aligned)
    // cb_stats_slots regions of G * Q pages (host _depths); > 1 <=> the WAVE_PIPELINE look-ahead (num_waves > 1)
    constexpr uint32_t stats_depth = get_compile_time_arg_val(24);
    constexpr bool pipelined = stats_depth > 1;
    constexpr uint32_t sem_gather_odd_id = get_compile_time_arg_val(25);
    constexpr uint32_t sem_own_fwd_id = get_compile_time_arg_val(26);   // FWD_INV member: own slot forwarded (count)
    constexpr uint32_t sem_slot_fwd_id = get_compile_time_arg_val(27);  // FWD_INV member: (block+1) << 24 | slot mask
    constexpr auto input_args = TensorAccessorArgs<28>();  // after the 28 common CT args (host COMMON_CT_ARGS)
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    // FWD_INV (Perf 1 replicated_finalize): a single-root R3-X row-group forwards the raw stat slots to its members and
    // every member finalizes its own inv. R3-RS (several owners) keeps the owners' unicast inv all-gather.
    constexpr bool fwd_inv = r3 && OWNERS == 1;
    static_assert(!fwd_inv || G <= 24, "SEM_SLOT_FWD packs one bit per slot below the block tag (bits 24..31)");

    constexpr uint32_t T_in = get_tile_size(cb_x_tiles);
    constexpr uint32_t T32 = get_tile_size(cb_stats_slots);

    size_t a = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(a++);
    const uint32_t start_row = get_arg_val<uint32_t>(a++);
    const uint32_t rows = get_arg_val<uint32_t>(a++);
    const uint32_t w0 = get_arg_val<uint32_t>(a++);
    const uint32_t V = get_arg_val<uint32_t>(a++);
    const bool is_root = get_arg_val<uint32_t>(a++) != 0;  // a root (R3-RS: an owner, index < OWNERS)
    const uint32_t my_index = get_arg_val<uint32_t>(a++);
    const uint32_t p = get_arg_val<uint32_t>(a++);
    const uint32_t landing = get_arg_val<uint32_t>(a++);
    const uint32_t seg_stride = get_arg_val<uint32_t>(a++);
    const uint32_t n_pub = get_arg_val<uint32_t>(a++);
    const uint32_t n_notify = get_arg_val<uint32_t>(a++);
    const uint32_t pub_idx = a;
    a += n_pub;
    const uint32_t notify_idx = a;  // n_notify entries: the landed semaphore id of each notifying port (per direction)
    a += n_notify;
    // Per notifying port k: n_in_k, then the slots landing there in entry order (the port notifies per entry).
    const uint32_t entries_idx = a;
    for (uint32_t k = 0; k < n_notify; ++k) {
        a += 1 + get_arg_val<uint32_t>(a);
    }
    const uint32_t slot_idx = a;  // G entries: packed coords of the port holding slot q (q != p)
    a += G;
    const uint32_t offset_idx = a;  // num_waves entries: this root's byte offset in its link's segment
    a += num_waves;
    const uint32_t member_idx = a;  // R3: the G_w packed member coords, index order (member 0 = owner 0)
    auto member = [member_idx](uint32_t k) { return get_arg_val<uint32_t>(member_idx + k); };
    a += r3 ? G_w : 0;
    // FWD_INV root: the multicast rectangles covering its members (itself excluded), in this kernel's NoC order:
    // x_start, y_start, x_end, y_end, num_dests each (members and R1-X: none).
    const uint32_t n_rect = get_arg_val<uint32_t>(a++);
    const uint32_t rect_idx = a;

    const auto input = TensorAccessor(input_args, in_addr);
    Noc noc;
    // Element bytes / bytes ahead of the elements of an input tile: Float32 4 / 0, Float16_b 2 / 0, Bfp8_b 1 / 64 (the
    // shared exponents; 1088 = 64 + 1024 * 1).
    constexpr uint32_t es_in = T_in / (32 * 32);
    constexpr uint32_t data_off_in = T_in - es_in * 32 * 32;
    // This slice holds the row's last tile, whose W-pad columns must be zeroed before the square pass.
    const bool zero_w_pad = w_last_valid < 32 && w0 + V == Wt;

    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_reduce_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    if (!x_shard && V < WS) {
        // Ragged slice: zero the pad tiles of every tile-row slot of the x ring once; the reader never writes them.
        CircularBuffer x_cb(cb_x_tiles);
        constexpr uint32_t ring_rows = x_depth * R;
        x_cb.reserve_back(ring_rows * WS);
        for (uint32_t row = 0; row < ring_rows; ++row) {
            noc.async_write_zeros(x_cb, (WS - V) * T_in, {.offset_bytes = (row * WS + V) * T_in});
        }
        noc.write_zeros_l1_barrier();
    }

    // Captured before any push: identical on every compute core (same CB layout everywhere).
    const uint32_t gather_base = get_write_ptr(cb_gather);
    volatile tt_l1_ptr uint32_t* sem_inv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_inv_id));
    const uint32_t own_stats_sem_addr = get_semaphore(sem_own_stats_id);
    const uint32_t inv_sem_addr = get_semaphore(sem_inv_id);
    volatile tt_l1_ptr uint32_t* sem_counted =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_counted_id));
    // SEM_OWN_STATS on a sending port is cumulative over waves: before this root's wave-w increment (w >= 1), every
    // sending port must have counted wave w-1, or a fast root's wave-w increment (an empty block increments at once)
    // could stand in for a slow root's still-unpublished wave-(w-1) rows.
    auto wait_counted = [&](uint32_t w) {
        if (w > 0 && n_pub > 0) {
            DeviceZoneScopedN("rd_counted_wait");
            noc_semaphore_wait_min(sem_counted, w * n_pub);
        }
    };

    const uint32_t stats_base = get_write_ptr(cb_stats_slots);  // region r of the slot ring: + r * G * Q pages

    // ---- FWD_INV slot forward (root) ----
    // Per member rectangle: the 128 B tails of `n` pages (page0 + i * stride) as LINKED multicasts, then the multicast
    // semaphore SET of `value` to the same rectangle. The set closes that rectangle's linked chain, so it can never
    // overtake the tails (no write barrier, also with several rectangles: each chain is ordered on its own). Every
    // member's cb_stats_slots region sits at the same address (same CB layout, one G * Q push per non-empty block).
    // The set's source is a fresh 16 B staging word of the (otherwise unused) cb_inv_root page: a word is never
    // rewritten while a multicast may still read it (256 words, flushed on wrap). Values only grow, and one source's
    // multicasts to one rectangle arrive in issue order, so a member never sees a semaphore go back.
    const uint32_t stage_base = fwd_inv ? get_write_ptr(cb_inv_root) : 0;
    uint32_t n_staged = 0;
    auto forward_and_signal = [&](uint32_t page0, uint32_t stride, uint32_t n, uint32_t sem_addr, uint32_t value) {
        DeviceZoneScopedN("rd_fwd");
        constexpr uint32_t STAGE_WORDS = 256;  // 16 B words in one fp32 page (T32 = 4 KiB)
        if (n_staged != 0 && n_staged % STAGE_WORDS == 0) {
            noc_async_writes_flushed();  // the ring wraps: every earlier set has read its word
        }
        const uint32_t src = stage_base + (n_staged++ % STAGE_WORDS) * 16;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src) = value;
        for (uint32_t r = 0; r < n_rect; ++r) {
            const uint32_t t = rect_idx + r * 5;
            const uint32_t x0 = get_arg_val<uint32_t>(t), y0 = get_arg_val<uint32_t>(t + 1);
            const uint32_t x1 = get_arg_val<uint32_t>(t + 2), y1 = get_arg_val<uint32_t>(t + 3);
            const uint32_t nd = get_arg_val<uint32_t>(t + 4);
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t wire = page0 + i * stride;  // no_col0_expand: the wire words sit at the page base
                noc_async_write_multicast(wire, get_noc_multicast_addr(x0, y0, x1, y1, wire), ROW_BYTES, nd, true);
            }
            noc_semaphore_set_multicast(src, get_noc_multicast_addr(x0, y0, x1, y1, sem_addr), nd);
        }
    };
    // The gather semaphore of block b's parity (one semaphore without the look-ahead: gather_depth = 1).
    auto gather_sem_addr_of = [&](uint32_t b) {
        return get_semaphore((b % gather_depth) == 0 ? sem_gather_id : sem_gather_odd_id);
    };

    // Geometry of block (= wave) w; identical in front / back and across the row-group.
    struct Block {
        uint32_t R_b;         // tile-rows of the block (0: this group has no rows in wave w)
        uint32_t q;           // rows per owner
        uint32_t num_owners;  // owners holding rows
        uint32_t my_first;    // first row this core owns
        uint32_t my_rows;     // rows this core owns (0: not an owner of this block)
    };
    auto block = [&](uint32_t w) {
        Block b;
        const uint32_t first = w * R;
        b.R_b = rows > first ? (rows - first < R ? rows - first : R) : 0;
        // Ownership of this block's rows (host owner_rows): owner k sums / publishes / finalizes [k*q, k*q + n_k).
        b.q = (b.R_b + OWNERS - 1) / OWNERS;
        b.num_owners = b.R_b ? (b.R_b + b.q - 1) / b.q : 0;
        b.my_first = my_index * b.q;
        b.my_rows = (is_root && b.my_first < b.R_b) ? (b.R_b - b.my_first < b.q ? b.R_b - b.my_first : b.q) : 0;
        return b;
    };
    // An owner with nothing to publish in wave w (an empty or ragged block): the ports still count one increment.
    auto publish_nothing = [&](uint32_t w) {
        wait_counted(w);
        for (uint32_t k = 0; k < n_pub; ++k) {
            noc_semaphore_inc(noc_xy(get_arg_val<uint32_t>(pub_idx + k), own_stats_sem_addr), 1);
        }
    };

    uint32_t blocks_done = 0;              // non-empty blocks fronted (identical across a row-group)
    uint32_t gather_expected[2] = {0, 0};  // SEM_GATHER arrivals consumed, per block parity (G_w - 1 per owned block)
    uint32_t inv_expected = 0;             // SEM_INV arrivals this member has consumed (one per other owner per block)
    uint32_t published = 0;  // blocks whose own stats this root published (stats ring region = published % depth)
    uint32_t gathered = 0;   // blocks whose slots this root gathered + pushed (same region sequence)

    // ---- front(w): load_block, core_combine_block, publish_stats_block ----
    auto front = [&](uint32_t w) {
        const Block b = block(w);
        if (b.R_b == 0) {
            if (is_root) {
                publish_nothing(w);
            }
            return;
        }
        const uint32_t R_b = b.R_b;
        const uint32_t q = b.q;
        const uint32_t first = w * R;

        // ---- load_block ----
        {
            DeviceZoneScopedN("rd_load");
            if constexpr (x_shard) {
                // The block is already resident in the shard: hand it over (compute squares its V valid tiles only).
                cb_reserve_back(cb_x_tiles, R_b * WS);
                if constexpr (w_last_valid < 32) {
                    if (zero_w_pad) {
                        CircularBuffer x_cb(cb_x_tiles);
                        for (uint32_t t = 0; t < R_b; ++t) {
                            zero_tile_w_padding<es_in, data_off_in>(noc, x_cb, (t * WS + V - 1) * T_in, w_last_valid);
                        }
                        noc.write_zeros_l1_barrier();
                    }
                }
                cb_push_back(cb_x_tiles, R_b * WS);
            } else {
                for (uint32_t t = 0; t < R_b; ++t) {
                    const uint32_t page0 = (start_row + first + t) * Wt + w0;
                    cb_reserve_back(cb_x_tiles, WS);
                    const uint32_t l1 = get_write_ptr(cb_x_tiles);
                    for (uint32_t j = 0; j < V; ++j) {
                        noc_async_read_tile(page0 + j, input, l1 + j * T_in);
                    }
                    noc_async_read_barrier();
                    if constexpr (w_last_valid < 32) {
                        if (zero_w_pad) {
                            CircularBuffer x_cb(cb_x_tiles);
                            zero_tile_w_padding<es_in, data_off_in>(noc, x_cb, (V - 1) * T_in, w_last_valid);
                            noc.write_zeros_l1_barrier();
                        }
                    }
                    cb_push_back(cb_x_tiles, WS);
                }
            }
        }

        // ---- core_combine_block (R3-X) ----
        // Every member sends its partials of owner k's rows (contiguous in cb_partial_col) to slots
        // [my_index*q, +n_k) of k's cb_gather region (blocks_done % gather_depth; contributor-major, row stride q;
        // its own by a local write), barrier, then increments every OTHER owner's gather semaphore of the block's
        // parity. An owner waits the G_w - 1 others and pushes G_w*q pages.
        // Slot safety: a member's partials of block b+gather_depth exist only after its compute consumed inv of
        // block b+gather_depth-2 (look-ahead) / b+gather_depth-1, which every owner sent after its compute had
        // consumed its gather of block b.
        if constexpr (r3) {
            {
                DeviceZoneScopedN("rd_wait_partial");
                cb_wait_front(cb_partial_col, R_b);
            }
            const uint32_t par = blocks_done % gather_depth;
            const uint32_t src = get_read_ptr(cb_partial_col);
            const uint32_t slot = gather_base + (par * G_w * Q + my_index * q) * T32;
            if (b.my_rows > 0) {
                cb_reserve_back(cb_gather, G_w * q);
            }
            for (uint32_t kk = 0; kk < b.num_owners; ++kk) {
                const uint32_t k = drms_combine::rotated(kk, my_index, b.num_owners);
                const uint32_t kf = k * q;
                const uint32_t n = R_b - kf < q ? R_b - kf : q;
                if (k == my_index) {
                    noc_async_write(src + kf * T32, get_noc_addr(slot), n * T32);
                } else {
                    const uint32_t mk = member(k);
                    drms_combine::combine_write_tiles<T32>(src + kf * T32, mk >> 16, mk & 0xFFFF, slot, n);
                }
            }
            noc_async_write_barrier();  // the partials have landed before any owner can count them
            const uint32_t gsem = gather_sem_addr_of(blocks_done);
            for (uint32_t k = 0; k < b.num_owners; ++k) {
                if (k != my_index) {
                    noc_semaphore_inc(noc_xy(member(k), gsem), 1);
                }
            }
            if (b.my_rows > 0) {
                gather_expected[par] += G_w - 1;
                {
                    DeviceZoneScopedN("rd_gather_wait");
                    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gsem), gather_expected[par]);
                }
                cb_push_back(cb_gather, G_w * q);
            }
            cb_pop_front(cb_partial_col, R_b);
        }
        ++blocks_done;

        if (is_root && b.my_rows == 0) {
            publish_nothing(w);  // an owner with no rows in this block (R3-RS, ragged block)
        }
        if (b.my_rows > 0) {
            // ---- publish_stats_block (this root's my_rows rows) -> stats ring region `published % depth` ----
            const uint32_t my_rows = b.my_rows;
            {
                DeviceZoneScopedN("rd_wait_devpart");
                cb_wait_front(cb_device_partial, my_rows);
            }
            l1_acquire();
            // Every region free but the one still waiting for its slots (look-ahead): the region written here.
            cb_reserve_back(cb_stats_slots, stats_depth * G * Q);
            const uint32_t slots = stats_base + (published % stats_depth) * G * Q * T32;
            const uint32_t seg_off = get_arg_val<uint32_t>(offset_idx + w);
            const uint32_t part = get_read_ptr(cb_device_partial);
            for (uint32_t t = 0; t < my_rows; ++t) {
                const uint32_t page = slots + (t * G + p) * T32;
                col0_to_wire(part + t * T32, page);
                const uint32_t dst = landing + (w * G + p) * seg_stride + seg_off + t * ROW_BYTES;
                for (uint32_t k = 0; k < n_pub; ++k) {
                    noc_async_write(page, noc_xy(get_arg_val<uint32_t>(pub_idx + k), dst), ROW_BYTES);
                }
            }
            noc_async_write_barrier();
            wait_counted(w);
            for (uint32_t k = 0; k < n_pub; ++k) {
                noc_semaphore_inc(noc_xy(get_arg_val<uint32_t>(pub_idx + k), own_stats_sem_addr), 1);
            }
            cb_pop_front(cb_device_partial, my_rows);
            ++published;
            if constexpr (fwd_inv) {
                // the own slot's wire words (written by col0_to_wire above) to every member: SEM_OWN_FWD = published
                forward_and_signal(slots + p * T32, G * T32, my_rows, get_semaphore(sem_own_fwd_id), published);
            }
        }
    };

    // ---- back(w): gather_slots_block, distribute_inv_block ----
    auto back = [&](uint32_t w) {
        const Block b = block(w);
        if (b.R_b == 0) {
            return;
        }
        const uint32_t R_b = b.R_b;
        const uint32_t my_rows = b.my_rows;
        if (my_rows > 0) {
            // ---- gather_slots_block ----
            // Per-entry (lamp L4): each notifying port k signals its own semaphore once per landed entry, in entry
            // order, so entry i of wave w is in its landing once that semaphore reaches w * n_in_k + i + 1. A slot's
            // reads are issued as soon as it lands (no barrier), and every batch whose reads have all completed is
            // forwarded (FWD_INV) and expanded while later slots are still on the Fabric. The sum's order is fixed by
            // the page layout (slot g at page t*G + g), not by arrival: inv stays bit-identical.
            DeviceZoneScopedN("rd_gather");
            const uint32_t slots = stats_base + (gathered % stats_depth) * G * Q * T32;
            const uint32_t seg_off = get_arg_val<uint32_t>(offset_idx + w);
            uint32_t got[2] = {0, 0};  // entries of wave w issued so far, per notifying port (<= 2: FWD / BWD)
            uint32_t remaining = G - 1;
            uint32_t inflight[G];
            uint32_t n_inflight = 0;
            uint32_t fwd_mask = 0;  // peer slots forwarded so far (FWD_INV)
            while (remaining > 0 || n_inflight > 0) {
                if (remaining > 0) {
                    invalidate_l1_cache();
                    uint32_t e = entries_idx;
                    for (uint32_t k = 0; k < n_notify; ++k) {
                        const uint32_t n_in_k = get_arg_val<uint32_t>(e);
                        volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            get_semaphore(get_arg_val<uint32_t>(notify_idx + k)));
                        const uint32_t seen = *landed;
                        while (got[k] < n_in_k && seen >= w * n_in_k + got[k] + 1) {
                            const uint32_t qq = get_arg_val<uint32_t>(e + 1 + got[k]);
                            ++got[k];
                            --remaining;
                            const uint32_t port = get_arg_val<uint32_t>(slot_idx + qq);
                            const uint32_t src = landing + (w * G + qq) * seg_stride + seg_off;
                            for (uint32_t t = 0; t < my_rows; ++t) {
                                noc_async_read(
                                    noc_xy(port, src + t * ROW_BYTES), slots + (t * G + qq) * T32, ROW_BYTES);
                            }
                            inflight[n_inflight++] = qq;
                        }
                        e += 1 + n_in_k;
                    }
                }
                if (n_inflight > 0 && (remaining == 0 || ncrisc_noc_reads_flushed(noc_index))) {
                    noc_async_read_barrier();  // immediate unless the last slots are still in flight
                    if constexpr (fwd_inv) {
                        // one linked chain per slot (its my_rows rows, page stride G), the mask set closing each
                        for (uint32_t j = 0; j < n_inflight; ++j) {
                            fwd_mask |= 1u << inflight[j];
                            forward_and_signal(
                                slots + inflight[j] * T32,
                                G * T32,
                                my_rows,
                                get_semaphore(sem_slot_fwd_id),
                                ((gathered + 1) << 24) | fwd_mask);
                        }
                    }
                    // no_col0_expand: the landed wire words are already in compute's form (no rd_slot_expand)
                    n_inflight = 0;
                }
            }
            cb_push_back(cb_stats_slots, G * Q);  // a whole region (rows >= my_rows unused): uniform ring quanta
            ++gathered;
        } else if constexpr (fwd_inv) {
            // ---- FWD_INV member: the root forwards every slot's tails into this core's region ----
            // Own slot p: ready once SEM_OWN_FWD >= block + 1. Peer slots: SEM_SLOT_FWD = (block + 1) << 24 | mask;
            // a tag above block + 1 means the root already forwards a later block, so every slot of this one is in.
            // Region safety: the root writes region (block % depth) of this core only at publish / gather of that
            // block, i.e. after its core-combine gather counted this core's partial of the block — which this core's
            // compute produced after finalizing (and popping) the region's previous block.
            DeviceZoneScopedN("rd_mbr_gather");
            cb_reserve_back(cb_stats_slots, G * Q);
            const uint32_t slots = stats_base + (gathered % stats_depth) * G * Q * T32;
            volatile tt_l1_ptr uint32_t* own_fwd =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_own_fwd_id));
            volatile tt_l1_ptr uint32_t* slot_fwd =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_slot_fwd_id));
            constexpr uint32_t all = (1u << G) - 1;
            const uint32_t tag = gathered + 1;
            // no_col0_expand: the forwarded wire words are compute's form already — only wait for every slot's bit
            uint32_t ready = 0;
            while (ready != all) {
                invalidate_l1_cache();
                ready = *own_fwd >= tag ? (1u << p) : 0;
                const uint32_t v = *slot_fwd;
                ready |= (v >> 24) > tag ? all : ((v >> 24) == tag ? (v & 0xFFFFFF) : 0);
            }
            cb_push_back(cb_stats_slots, G * Q);
            ++gathered;
        }

        // ---- distribute_inv_block (R3-RS: several owners) ----
        // inv crosses the NoC in its compact wire form (128 B per tile-row, the page's face-3 tail; every member
        // expands it to column 0 itself). Owner k copies its finalized rows into its own cb_inv_rms pages
        // [k*q, +n_k) (column 0 + tail), writes the tails to the same pages of every other member (every member's
        // cb_inv_rms sits at the same address: same CB layout, same push sequence), then increments their SEM_INV.
        // Every member reserves R_b pages, waits for the other owners and expands their tails. (A single-root
        // row-group, FWD_INV, has no inv distribution: every member finalized its own.)
        if constexpr (r3 && !fwd_inv) {
            const uint32_t my_first = b.my_first;
            cb_reserve_back(cb_inv_rms, R_b);
            const uint32_t dst = get_write_ptr(cb_inv_rms);
            if (my_rows > 0) {
                {
                    DeviceZoneScopedN("rd_wait_invroot");
                    cb_wait_front(cb_inv_root, my_rows);
                }
                const uint32_t src = get_read_ptr(cb_inv_root);
                const uint32_t own = dst + my_first * T32;
                {
                    DeviceZoneScopedN("rd_dist_copy");
                    l1_acquire();
                    for (uint32_t t = 0; t < my_rows; ++t) {
                        col0_to_page(src + t * T32, own + t * T32);
                    }
                }
                {
                    DeviceZoneScopedN("rd_dist_send");
                    send_inv_tails<G_w, T32, Q>(own, my_rows, my_index, member_idx, inv_sem_addr);
                }
                cb_pop_front(cb_inv_root, my_rows);
            }
            inv_expected += b.num_owners - (my_rows > 0 ? 1 : 0);
            {
                DeviceZoneScopedN("rd_inv_wait");
                noc_semaphore_wait_min(sem_inv, inv_expected);
            }
            l1_acquire();
            for (uint32_t t = 0; t < R_b; ++t) {
                if (t < my_first || t >= my_first + my_rows) {
                    tail_to_col0(dst + t * T32);
                }
            }
            cb_push_back(cb_inv_rms, R_b);
        }
    };

    if constexpr (pipelined) {
        // One-block look-ahead: wave w+1's load / combine / publish run while wave w's stats are on the Fabric.
        front(0);
        for (uint32_t w = 0; w < num_waves; ++w) {
            if (w + 1 < num_waves) {
                front(w + 1);
            }
            back(w);
        }
    } else {
        for (uint32_t w = 0; w < num_waves; ++w) {
            front(w);
            back(w);
        }
    }
    noc_async_write_barrier();  // FWD_INV: the last forwards are acknowledged before the program ends
    noc_async_atomic_barrier();
}
