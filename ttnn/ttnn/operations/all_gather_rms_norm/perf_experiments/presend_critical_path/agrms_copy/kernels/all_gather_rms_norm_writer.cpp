// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm member writer (BRISC / NoC1): load_gamma_slice (once) + exchange_partials_block (k > 1) +
// store_block. Ported from the base op's writer (rms_norm_writer.cpp: GammaSliceLoad, exchange_send /
// exchange_recv_partials, the TILE store loop).
//
// load_gamma_slice: gamma pages [g0, g0 + n) (regime A: the member's landing = output tiles; regime N: every dense
//   output tile) into the compact cb_gamma pages compute multiplies ROW-broadcast (row 0 of faces 0/1 only):
//   TILE gamma (1,1,1,W): per tile, face 0 row 0 and face 1 row 0 straight from DRAM (64 B reads).
//   ROW_MAJOR sticks (1,1,ceil(W/32),32): page t IS gamma[32t .. 32t+31]: one 64 B read of the stick into face 0
//     (rows 0..1 for bf16) + a local copy of its upper half into face 1 row 0 (bf16), or two 64 B reads straight into
//     face 0 / face 1 row 0 (fp32).
// exchange_partials_block (symmetric, k <= K_SYM_MAX): every member writes its rows' column-0 partials (faces 0/2)
//   into slot (r*k + m) of EVERY member's cb_gathered_sumsq (lockstep slot base = this core's own write pointer),
//   flushes, bumps each member's partials semaphore of the block's PARITY by 1; then waits for k * (blocks of that
//   parity so far) arrivals and publishes the slots.
//   Why two semaphores: a peer can run one block ahead (its partial(b+1) only needs every partial(b)), so on ONE
//   cumulative counter a fast peer's increment for b+1 can stand in for a slow peer's increment for b that is still in
//   flight, and the combine of b reads that peer's slot before it is written (stale partial -> wrong inv_rms; seen
//   on the golden suite after a program switch). A peer can never be two blocks ahead (partial(b+2) needs this
//   member's partial(b+1), produced only after combine(b)), so a per-parity count only completes with all k
//   partials of block b.
// store_block: per tile-row, in units of <= out_unit tiles (clamped to the cb_out ring end): wait, write, flush, pop.
//   Native sharded placement (out_native, Refinement 1): cb_out_tiles IS the resident output shard (zero-copy);
//   compute packs every row at its final place, so the store is only the consumer credit (wait + pop, no write).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "all_gather_rms_norm_common.hpp"

using namespace agrms_cb;

constexpr uint32_t out_page_bytes = get_compile_time_arg_val(0);
constexpr uint32_t out_w_tiles = get_compile_time_arg_val(1);
constexpr uint32_t w_split_cores = get_compile_time_arg_val(2);  // k
constexpr uint32_t out_unit = get_compile_time_arg_val(3);
constexpr uint32_t out_cb_pages = get_compile_time_arg_val(4);
constexpr uint32_t sem_partials_id = get_compile_time_arg_val(5);
constexpr bool partial_faces_only = get_compile_time_arg_val(6) != 0;
constexpr bool gamma_is_tile = get_compile_time_arg_val(7) != 0;
constexpr uint32_t gamma_elem_bytes = get_compile_time_arg_val(8);
constexpr uint32_t gamma_l1_page_bytes = get_compile_time_arg_val(9);  // compact cb_gamma page stride
constexpr uint32_t gamma_tail_pages = get_compile_time_arg_val(10);
constexpr uint32_t gamma_dram_page_bytes = get_compile_time_arg_val(11);
constexpr bool regime_n = get_compile_time_arg_val(12) != 0;
constexpr uint32_t block_row_tiles = get_compile_time_arg_val(13);
constexpr uint32_t n_groups = get_compile_time_arg_val(14);
constexpr bool combine_owner = get_compile_time_arg_val(15) != 0;  // k > K_SYM_MAX: per-row owner combine
constexpr uint32_t row_gather = get_compile_time_arg_val(16);      // owner: gather tiles per owned row, ceil(k / 32)
constexpr uint32_t sem_inv_id = get_compile_time_arg_val(17);      // owner: inv_rms returns arrived
constexpr uint32_t gathered_cb_pages = get_compile_time_arg_val(18);
constexpr uint32_t sem_partials_odd_id = get_compile_time_arg_val(19);  // partials of odd blocks (see below)
constexpr uint32_t GAMMA_READ_ALIGN = get_compile_time_arg_val(20);     // DRAM read alignment (>= one face row)
constexpr bool out_native = get_compile_time_arg_val(21) != 0;          // cb_out_tiles backed on the output shard
// Local work (Refinement 5): symmetric exchange of ROW-form partials (row 0 = the 32 row sums, as the owner combine
// ships them): two face rows into row member_idx of the row's gather tile on every peer, not faces 0/2.
constexpr bool row_partials = get_compile_time_arg_val(22) != 0;
constexpr auto out_args = TensorAccessorArgs<23>();
[[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

constexpr uint32_t FACE_DIM = 16;

#ifdef AGRMS_HAS_GAMMA
template <typename Accessor>
FORCE_INLINE void load_gamma_slice(const Accessor& g_acc, uint32_t g0, uint32_t n) {
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * gamma_elem_bytes;
    constexpr uint32_t half_row_bytes = FACE_DIM * gamma_elem_bytes;  // one 16-datum face row
    cb_reserve_back(cb_gamma, n + gamma_tail_pages);
    const uint32_t base = get_write_ptr(cb_gamma);
    for (uint32_t j = 0; j < n; ++j) {
        const uint32_t page = base + j * gamma_l1_page_bytes;
        if constexpr (gamma_is_tile) {
            noc_async_read(g_acc.get_noc_addr(g0 + j, 0), page, GAMMA_READ_ALIGN);
            noc_async_read(g_acc.get_noc_addr(g0 + j, face_bytes), page + face_bytes, GAMMA_READ_ALIGN);
        } else if constexpr (half_row_bytes % GAMMA_READ_ALIGN == 0) {
            // fp32: each half of the stick is itself 64 B-aligned in DRAM: straight into face 0 / face 1 row 0
            noc_async_read(g_acc.get_noc_addr(g0 + j, 0), page, half_row_bytes);
            noc_async_read(g_acc.get_noc_addr(g0 + j, half_row_bytes), page + face_bytes, half_row_bytes);
        } else {
            // bf16: the 64 B stick lands in face 0 rows 0..1; row 1 (the upper half) is moved to face 1 below
            noc_async_read(g_acc.get_noc_addr(g0 + j, 0), page, 2 * half_row_bytes);
        }
    }
    noc_async_read_barrier();
    if constexpr (!gamma_is_tile && half_row_bytes % GAMMA_READ_ALIGN != 0) {
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t page = base + j * gamma_l1_page_bytes;
            noc_async_read(get_noc_addr(page + half_row_bytes), page + face_bytes, half_row_bytes);
        }
        noc_async_read_barrier();
    }
    cb_push_back(cb_gamma, n + gamma_tail_pages);
}
#endif

FORCE_INLINE uint64_t peer_addr(uint32_t peers_idx, uint32_t m, uint32_t l1_addr) {
    const uint32_t xy = get_arg_val<uint32_t>(peers_idx + m);
    return get_noc_addr(xy & 0xFFFF, xy >> 16, l1_addr);
}

// One partial tile to a peer: faces 0 and 2 (column 0) or the whole tile.
FORCE_INLINE void send_acc_tile(uint32_t src, uint64_t dst, uint32_t tile_bytes) {
    if constexpr (partial_faces_only) {
        const uint32_t face_bytes = tile_bytes / 4;
        noc_async_write(src, dst, face_bytes);
        noc_async_write(src + 2 * face_bytes, dst + 2 * face_bytes, face_bytes);
    } else {
        noc_async_write(src, dst, tile_bytes);
    }
}

// ---- Owner combine (k > K_SYM_MAX), ported from the base op's writer (exchange_send owner branch,
// owner_wait_partials, owner_return_inv, exchange_recv_inv). Row r of a block (rotation ofs) is owned by member
// (r + ofs) % k. A member's partial of row r arrives from compute in ROW form (row 0 = its 32 row sums) and ships as
// two face rows into row (m % 32) of gather tile m / 32 of the owner's slot for r; every member reserves / pushes
// ceil(rows / k) * row_gather gathered pages per block (lockstep slot base). The owner returns inv_rms of each owned
// row into slot r of every member's cb_inv_rms (faces 0/2) and bumps their sem_inv.
constexpr uint32_t TILE_DIM = 32;
FORCE_INLINE uint32_t gather_row_offset(uint32_t member, uint32_t tile_bytes) {
    const uint32_t face_bytes = tile_bytes / 4;
    const uint32_t frow = tile_bytes / 64;  // one 16-datum face row
    const uint32_t i = member % TILE_DIM;
    return (member / TILE_DIM) * tile_bytes + (i < FACE_DIM ? i * frow : 2 * face_bytes + (i - FACE_DIM) * frow);
}

// Rows k % 32 .. 31 of an owned row's LAST gather tile are written by no member: zero them once (local NoC reads of
// the hardware zeros region). Every block reserves a multiple of row_gather pages and the ring holds a multiple of
// it, so a last gather tile always sits on a ring page p with p % row_gather == row_gather - 1.
FORCE_INLINE void zero_fill_range(uint64_t zeros, uint32_t a, uint32_t end) {
    for (; a < end; a += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, a, (end - a) < MEM_ZEROS_SIZE ? (end - a) : MEM_ZEROS_SIZE);
    }
}
FORCE_INLINE void zero_unused_gather_rows() {
    constexpr uint32_t row_unused0 = w_split_cores % TILE_DIM;
    if constexpr ((combine_owner || row_partials) && row_unused0 != 0) {
        const uint32_t tb = get_tile_size(cb_gathered_sumsq);
        const uint32_t fb = tb / 4;
        const uint32_t frow = tb / 64;
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        const uint32_t base = get_write_ptr(cb_gathered_sumsq);
        for (uint32_t p = row_gather - 1; p < gathered_cb_pages; p += row_gather) {
            const uint32_t tile = base + p * tb;
            if constexpr (row_unused0 < FACE_DIM) {
                zero_fill_range(zeros, tile + row_unused0 * frow, tile + fb);
                zero_fill_range(zeros, tile + fb + row_unused0 * frow, tile + 2 * fb);
                zero_fill_range(zeros, tile + 2 * fb, tile + 4 * fb);
            } else {
                zero_fill_range(zeros, tile + 2 * fb + (row_unused0 - FACE_DIM) * frow, tile + 3 * fb);
                zero_fill_range(zeros, tile + 3 * fb + (row_unused0 - FACE_DIM) * frow, tile + 4 * fb);
            }
        }
        noc_async_read_barrier();
    }
}

struct OwnerExchange {
    uint32_t peers_idx;
    uint32_t member_idx;
    uint32_t sem_addr_par[2];  // partials semaphores by block parity (see exchange_partials_block above)
    uint32_t sem_inv_addr;
    uint32_t tb;
    uint32_t partials_expected[2] = {0, 0};
    uint32_t inv_expected = 0;

    FORCE_INLINE void block(uint32_t rows, uint32_t ofs, uint32_t parity) {
        const uint32_t sem_addr = sem_addr_par[parity];
        constexpr uint32_t k = w_split_cores;
        const uint32_t face_bytes = tb / 4;
        const uint32_t frow = tb / 64;
        // send: this member's row partials to their owners
        cb_wait_front(cb_partial_sumsq, rows);
        const uint32_t src = get_read_ptr(cb_partial_sumsq);
        const uint32_t slots = ((rows + k - 1) / k) * row_gather;
        cb_reserve_back(cb_gathered_sumsq, slots);
        const uint32_t slot_base = get_write_ptr(cb_gathered_sumsq);
        const uint32_t row_off = gather_row_offset(member_idx, tb);
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t owner = (r + ofs) % k;
            const uint32_t j = (r + ofs) / k - (owner < ofs ? 1 : 0);
            const uint64_t dst = peer_addr(peers_idx, owner, slot_base + j * row_gather * tb + row_off);
            noc_async_write(src + r * tb, dst, frow);                            // face 0 row 0
            noc_async_write(src + r * tb + face_bytes, dst + face_bytes, frow);  // face 1 row 0
        }
        noc_async_writes_flushed();  // same NoC / VC / destination: the increments land after the data
        const uint32_t n_owners = rows < k ? rows : k;
        for (uint32_t r = 0; r < n_owners; ++r) {
            noc_semaphore_inc(peer_addr(peers_idx, (r + ofs) % k, sem_addr), (rows - r + k - 1) / k);
        }
        cb_pop_front(cb_partial_sumsq, rows);
        // owner: wait for the k partials of every owned row, publish them to compute
        const uint32_t r0 = (member_idx + k - ofs) % k;
        const uint32_t owned = r0 < rows ? (rows - r0 + k - 1) / k : 0;
        partials_expected[parity] += owned * k;
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr), partials_expected[parity]);
        cb_push_back(cb_gathered_sumsq, slots);
        // owner: return the combined inv_rms of each owned row r into slot r of every member's cb_inv_rms
        cb_reserve_back(cb_inv_rms, rows);
        const uint32_t inv_base = get_write_ptr(cb_inv_rms);
        for (uint32_t jj = 0; jj < owned; ++jj) {
            const uint32_t r = r0 + jj * k;
            cb_wait_front(cb_inv_owned, 1);
            const uint32_t inv_src = get_read_ptr(cb_inv_owned);
            for (uint32_t m = 0; m < k; ++m) {
                send_acc_tile(inv_src, peer_addr(peers_idx, m, inv_base + r * tb), tb);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_inv_owned, 1);
        }
        if (owned > 0) {
            for (uint32_t m = 0; m < k; ++m) {
                noc_semaphore_inc(peer_addr(peers_idx, m, sem_inv_addr), owned);
            }
        }
        // every member: wait for every row's inv_rms of the block, publish it to compute
        inv_expected += rows;
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_inv_addr), inv_expected);
        cb_push_back(cb_inv_rms, rows);
    }
};

void kernel_main() {
#if defined(AGRMS_ABLATE_EMPTY) || defined(AGRMS_ABLATE_TRANSPORT)
    return;
#endif
    uint32_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t grp = get_arg_val<uint32_t>(a++);
    const uint32_t n_rows = get_arg_val<uint32_t>(a++);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(a++);
    const uint32_t core_w = get_arg_val<uint32_t>(a++);
    const uint32_t member_idx = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t gamma_addr = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t out_pad = get_arg_val<uint32_t>(a++);  // native: pad pages per output row
    const uint32_t peers_idx = a;  // k packed (x | y << 16) NoC coords of the group's members

    // output tiles this member writes per row: regime A its landing (= output) tiles, regime N the dense row
    const uint32_t t_first = regime_n ? 0 : w_tile_start;
    const uint32_t t_count = regime_n ? out_w_tiles : core_w;

#ifdef AGRMS_HAS_GAMMA
    {
        const auto g_acc = TensorAccessor(gamma_args, gamma_addr, gamma_dram_page_bytes);
        load_gamma_slice(g_acc, t_first, t_count);
    }
#endif

    const auto out_acc = TensorAccessor(out_args, out_addr, out_page_bytes);
    [[maybe_unused]] const uint32_t sem_par[2] = {get_semaphore(sem_partials_id), get_semaphore(sem_partials_odd_id)};
    [[maybe_unused]] uint32_t partials_expected[2] = {0, 0};
    [[maybe_unused]] const uint32_t acc_tile_bytes = get_tile_size(cb_partial_sumsq);
    [[maybe_unused]] OwnerExchange owner_x{
        peers_idx, member_idx, {sem_par[0], sem_par[1]}, get_semaphore(sem_inv_id), acc_tile_bytes};
    zero_unused_gather_rows();

    uint32_t out_ring_pos = 0;
    for (uint32_t i0 = 0; i0 < n_rows; i0 += block_row_tiles) {
        const uint32_t rows = (n_rows - i0) < block_row_tiles ? (n_rows - i0) : block_row_tiles;
        [[maybe_unused]] const uint32_t parity = (i0 / block_row_tiles) & 1u;

        // ---- exchange_partials_block (symmetric all-gather of the k members' partials) ----
        {
            AGRMS_ZONE("AG_W_XCHG");
            if constexpr (w_split_cores > 1 && combine_owner) {
                owner_x.block(rows, i0 % w_split_cores, parity);  // owners rotate per block (compute: same ofs)
            } else if constexpr (w_split_cores > 1) {
                constexpr uint32_t k = w_split_cores;
                {
                    AGRMS_ZONE("AG_W_PWAIT");
                    cb_wait_front(cb_partial_sumsq, rows);
                }
                AGRMS_ZONE("AG_W_XSEND");
                const uint32_t src = get_read_ptr(cb_partial_sumsq);
                constexpr uint32_t slots_per_row = row_partials ? row_gather : k;
                cb_reserve_back(cb_gathered_sumsq, slots_per_row * rows);
                const uint32_t slot_base = get_write_ptr(cb_gathered_sumsq);
                if constexpr (row_partials) {
                    // row r of the block: gather tile(s) r * row_gather, row member_idx (faces 0/1 row 0 -> 2 writes)
                    const uint32_t face_bytes = acc_tile_bytes / 4;
                    const uint32_t frow = acc_tile_bytes / 64;
                    const uint32_t row_off = gather_row_offset(member_idx, acc_tile_bytes);
                    AGRMS_ZONE("AG_W_XW");
                    // one write-state per peer (its NoC coords + the face-row size), then 2 * rows stateful writes
                    for (uint32_t m = 0; m < k; ++m) {
                        noc_async_write_one_packet_set_state(peer_addr(peers_idx, m, slot_base), frow);
                        for (uint32_t r = 0; r < rows; ++r) {
                            const uint32_t s0 = src + r * acc_tile_bytes;
                            const uint32_t d0 = slot_base + r * row_gather * acc_tile_bytes + row_off;
                            noc_async_write_one_packet_with_state(s0, d0);
                            noc_async_write_one_packet_with_state(s0 + face_bytes, d0 + face_bytes);
                        }
                    }
                } else {
                    for (uint32_t m = 0; m < k; ++m) {
                        for (uint32_t r = 0; r < rows; ++r) {
                            send_acc_tile(
                                src + r * acc_tile_bytes,
                                peer_addr(peers_idx, m, slot_base + (r * k + member_idx) * acc_tile_bytes),
                                acc_tile_bytes);
                        }
                    }
                }
                // same NoC, VC and destination: the increments arrive after the data (flush, not barrier)
                {
                    AGRMS_ZONE("AG_W_XF");
                    noc_async_writes_flushed();
                }
                const uint32_t sem_addr = sem_par[parity];
                {
                    AGRMS_ZONE("AG_W_XI");
                    for (uint32_t m = 0; m < k; ++m) {
                        noc_semaphore_inc(peer_addr(peers_idx, m, sem_addr), 1);
                    }
                }
                cb_pop_front(cb_partial_sumsq, rows);
                partials_expected[parity] += k;
                {
                    AGRMS_ZONE("AG_W_SWAIT");
                    noc_semaphore_wait_min(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr), partials_expected[parity]);
                }
                cb_push_back(cb_gathered_sumsq, slots_per_row * rows);
            }
        }
        AGRMS_ZONE("AG_W_STORE");

        // ---- store_block ----
        if constexpr (out_native) {
            const uint32_t n = rows * (t_count + out_pad);
            cb_wait_front(cb_out_tiles, n);
            cb_pop_front(cb_out_tiles, n);
            continue;
        }
        for (uint32_t i = i0; i < i0 + rows; ++i) {
            const uint32_t r = grp + i * n_groups;
            const uint32_t first_page = r * out_w_tiles + t_first;
            for (uint32_t j0 = 0; j0 < t_count;) {
                uint32_t n = (t_count - j0) < out_unit ? (t_count - j0) : out_unit;
                n = (out_cb_pages - out_ring_pos) < n ? (out_cb_pages - out_ring_pos) : n;
                cb_wait_front(cb_out_tiles, n);
                const uint32_t l1 = get_read_ptr(cb_out_tiles);
                for (uint32_t j = 0; j < n; ++j) {
                    noc_async_write(l1 + j * out_page_bytes, out_acc.get_noc_addr(first_page + j0 + j), out_page_bytes);
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_out_tiles, n);
                j0 += n;
                out_ring_pos = (out_ring_pos + n == out_cb_pages) ? 0 : out_ring_pos + n;
            }
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
