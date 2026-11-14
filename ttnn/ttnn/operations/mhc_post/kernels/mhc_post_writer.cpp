// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post writer (RISCV_0, NoC1).
// store_block: per unit (token-tile row t, column tile c) the n X' tiles X'_j -> page t*n*Ct + j*Ct + c, consumed
// per compute block (j-major, unit-minor). Writes stream (flush per block, one barrier at the end).
// collapse: the pass's y tiles -> page t*Ct + c.
//
// R2 combine (op_design.md "Split chooser" / "Block schedule"), per token-tile row t of my row group:
//   partial: merge the row-summed X'^2 (column 0 of cb_sumsq_col) into column 24 of the X'@W partial, wait until
//     leader(t) announced its slots free for row t (bit t%32 of SEM_READY_BITS), NoC-write the tile into my slot
//     of the leader's cb_leader_slots, inc the leader's SEM_LEADER_COUNT.
//   leader(t) = chunk member (t mod chunk_size), LAGGED one row (after my own row t+1, so the leader never waits on
//     the chunk's slowest member): hand the slots to compute; send the folded chunk sum into root(t)'s
//     cb_root_slots, inc SEM_ROOT_COUNT; then re-reserve the slots (compute popped them) and announce my next led
//     row (t + chunk_size) to the chunk's members.
//   root(t) = member ((t - t_start) mod Q), DEFERRED past the root's own rows so that no core's row loop ever
//     waits on a root: every rooted row k has its own region of cb_root_slots (no reuse, no announcement); at the
//     end the root hands them to compute, writes stats; collapse: permutes the stats tile into SFPU lane layout for
//     the hc step and the hc result back into a row tile; writes hc. All semaphores are back to 0 at kernel end.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t FACE_WORDS = 256;
constexpr uint32_t FACE_DIM = 16;

// fp32 word of element (row, col) of a 32x32 tile (face-major, 16x16 row-major faces)
inline uint32_t tile_word(uint32_t row, uint32_t col) {
    return ((row / FACE_DIM) * 2 + col / FACE_DIM) * FACE_WORDS + (row % FACE_DIM) * FACE_DIM + col % FACE_DIM;
}
// fp32 word of SFPU vector v (dst_reg[v]), lane l: vector v = face v/8, row group (v%8)/2 (4 rows), column parity
// v%2; lane l = row 4*rg + l/8, column 2*(l%8) + parity. Any fixed lane order works: the hc step is lane-wise.
inline uint32_t lane_word(uint32_t v, uint32_t l) {
    const uint32_t f = v / 8, rg = (v % 8) / 2, par = v % 2;
    return f * FACE_WORDS + (4 * rg + l / 8) * FACE_DIM + 2 * (l % 8) + par;
}
// Move the 32 tokens of one value between a row-layout tile column (`col`: token l at row l) and its SFPU lane
// vector (`vec`: lane l). ToLanes: col -> vec, else vec -> col. Loads are issued 8 at a time ahead of their stores
// so the RISC's L1 load latency overlaps (a plain load-store loop costs ~20 cycles a word).
template <bool ToLanes>
inline void permute_vector(uint32_t* col, uint32_t* vec) {
#pragma GCC unroll 4
    for (uint32_t g = 0; g < 4; ++g) {  // lanes 8g..8g+7 = tokens 8g..8g+7 = one row of the vector
        uint32_t* c = col + (g >> 1) * 2 * FACE_WORDS + (g & 1) * 8 * FACE_DIM;  // token 8g's row
        uint32_t* v = vec + g * FACE_DIM;
        uint32_t r[8];
        if constexpr (ToLanes) {
#pragma GCC unroll 8
            for (uint32_t k = 0; k < 8; ++k) {
                r[k] = c[k * FACE_DIM];
            }
#pragma GCC unroll 8
            for (uint32_t k = 0; k < 8; ++k) {
                v[2 * k] = r[k];
            }
        } else {
#pragma GCC unroll 8
            for (uint32_t k = 0; k < 8; ++k) {
                r[k] = v[2 * k];
            }
#pragma GCC unroll 8
            for (uint32_t k = 0; k < 8; ++k) {
                c[k * FACE_DIM] = r[k];
            }
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_mixed_out_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t n = get_compile_time_arg_val(1);
    constexpr uint32_t block_rows = get_compile_time_arg_val(2);
    constexpr uint32_t block_cols = get_compile_time_arg_val(3);
    constexpr uint32_t c_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t out_ring_tiles = get_compile_time_arg_val(5);  // cb_mixed_out_tiles capacity
    constexpr bool has_w = get_compile_time_arg_val(6) != 0;
    constexpr bool collapse = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t group_q = get_compile_time_arg_val(8);
    constexpr uint32_t fanin = get_compile_time_arg_val(9);
    constexpr uint32_t cb_proj_out = get_compile_time_arg_val(10);
    constexpr uint32_t cb_sumsq_col = get_compile_time_arg_val(11);
    constexpr uint32_t cb_leader_slots = get_compile_time_arg_val(12);
    constexpr uint32_t cb_chunk_sum = get_compile_time_arg_val(13);
    constexpr uint32_t cb_root_slots = get_compile_time_arg_val(14);
    constexpr uint32_t cb_stats = get_compile_time_arg_val(15);
    constexpr uint32_t cb_hc_in = get_compile_time_arg_val(16);
    constexpr uint32_t cb_hc_out = get_compile_time_arg_val(17);
    constexpr uint32_t cb_y_out = get_compile_time_arg_val(18);
    constexpr uint32_t cb_hc_tile = get_compile_time_arg_val(19);
    constexpr uint32_t sem_leader_count_id = get_compile_time_arg_val(20);
    constexpr uint32_t sem_root_count_id = get_compile_time_arg_val(21);
    constexpr uint32_t sem_ready_bits_id = get_compile_time_arg_val(22);
    constexpr uint32_t ng = get_compile_time_arg_val(23);  // n(n+2): stats column of sum X'^2, hc width
    constexpr auto out_args = TensorAccessorArgs<24>();
    [[maybe_unused]] constexpr auto stats_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto hc_args = TensorAccessorArgs<stats_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto y_args = TensorAccessorArgs<hc_args.next_compile_time_args_offset()>();
    constexpr uint32_t num_chunks = (group_q + fanin - 1) / fanin;
    constexpr uint32_t COORD_BASE = 4;  // common args: out, stats, hc, y addresses, then the core table

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t t_count = get_arg_val<uint32_t>(1);
    const uint32_t c_start = get_arg_val<uint32_t>(2);
    const uint32_t c_count = get_arg_val<uint32_t>(3);
    [[maybe_unused]] const uint32_t group = get_arg_val<uint32_t>(4);
    [[maybe_unused]] const uint32_t member_q = get_arg_val<uint32_t>(5);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);

    const uint32_t out_page = get_tile_size(cb_mixed_out_tiles);
    const auto out_acc = TensorAccessor(out_args, out_addr, out_page);
    // A block window may wrap the ring (a ragged pass publishes fewer tiles): address it modulo the ring.
    const uint32_t ring_base = get_read_ptr(cb_mixed_out_tiles);  // nothing consumed yet: the CB base
    uint32_t ring_slot = 0;

    // ---------------- R2 state ----------------
    [[maybe_unused]] uint32_t f32_page = 0, leader_slots_base = 0, root_slots_base = 0;
    [[maybe_unused]] uint32_t sem_leader = 0, sem_root = 0, sem_ready = 0;
    [[maybe_unused]] uint32_t my_chunk = 0, chunk_first = 0, chunk_size = 0, my_pos = 0;
    auto member_noc = [&](uint32_t q, uint32_t addr) {
        const uint32_t xy = get_common_arg_val<uint32_t>(COORD_BASE + group * group_q + q);
        return get_noc_addr(xy & 0xFFFF, xy >> 16, addr);
    };
    // announce to my chunk that my leader slots are free for row `row`
    auto announce = [&](uint32_t row) {
        for (uint32_t m = 0; m < chunk_size; ++m) {
            noc_semaphore_inc(member_noc(chunk_first + m, sem_ready), 1u << (row % 32));
        }
    };
    auto is_leader = [&](uint32_t t) { return chunk_first + (t % chunk_size) == member_q; };
    // leader(t) duty, lagged one row (run after my own row t+1's stores): the chunk's partials -> compute (they
    // have long landed); the folded chunk sum -> slot `chunk` of region (t - t_start)/Q of root(t)'s
    // cb_root_slots; re-reserve the slots (compute popped them) and announce my next led row.
    auto lead = [&](uint32_t t) {
        const uint32_t root_q = (t - t_start) % group_q;  // root(t)
        volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_leader);
        noc_semaphore_wait_min(cnt, chunk_size);
        noc_semaphore_set(cnt, 0);             // no further incs before my next announcement
        cb_push_back(cb_leader_slots, fanin);  // reserved when the row was announced
        cb_wait_front(cb_chunk_sum, 1);
        noc_async_write(
            get_read_ptr(cb_chunk_sum),
            member_noc(root_q, root_slots_base + (((t - t_start) / group_q) * num_chunks + my_chunk) * f32_page),
            f32_page);
        noc_async_write_barrier();
        noc_semaphore_inc(member_noc(root_q, sem_root), 1);
        cb_pop_front(cb_chunk_sum, 1);
        if (t + chunk_size < t_start + t_count) {
            cb_reserve_back(cb_leader_slots, fanin);  // compute has folded (popped) this row's slots
            announce(t + chunk_size);
        }
    };
    if constexpr (has_w) {
        f32_page = get_tile_size(cb_proj_out);
        leader_slots_base = get_write_ptr(cb_leader_slots);  // same address on every core (one CB config)
        root_slots_base = get_write_ptr(cb_root_slots);
        sem_leader = get_semaphore(sem_leader_count_id);
        sem_root = get_semaphore(sem_root_count_id);
        sem_ready = get_semaphore(sem_ready_bits_id);
        my_chunk = member_q / fanin;
        chunk_first = my_chunk * fanin;
        chunk_size = (group_q - chunk_first) < fanin ? (group_q - chunk_first) : fanin;
        my_pos = member_q - chunk_first;
        // my first led row: the first t >= t_start with t mod chunk_size == my_pos
        const uint32_t first_led = t_start + ((my_pos + chunk_size - t_start % chunk_size) % chunk_size);
        if (first_led < t_start + t_count) {
            cb_reserve_back(cb_leader_slots, fanin);
            announce(first_led);
        }
    }

    for (uint32_t rb = 0; rb < t_count; rb += block_rows) {
        const uint32_t rows_in_block = (t_count - rb) < block_rows ? (t_count - rb) : block_rows;
        for (uint32_t r = 0; r < rows_in_block; ++r) {
            const uint32_t t = t_start + rb + r;
            const uint32_t x_row = t * n * c_tiles;
            for (uint32_t cb0 = 0; cb0 < c_count; cb0 += block_cols) {
                const uint32_t cols_in_pass = (c_count - cb0) < block_cols ? (c_count - cb0) : block_cols;
                // Same block decomposition as the compute: a full pass is one block; a ragged pass blocks of 1 (plain)
                // or, with W, one full block whose filler units (u >= cols_in_pass) are dropped here.
                const uint32_t b = (has_w || cols_in_pass == block_cols) ? block_cols : 1;
                const uint32_t valid = has_w ? cols_in_pass : b;
                for (uint32_t u0 = 0; u0 < cols_in_pass; u0 += b) {
                    const uint32_t c = c_start + cb0 + u0;
                    cb_wait_front(cb_mixed_out_tiles, b * n);
                    for (uint32_t j = 0; j < n; ++j) {  // block order: j-major, then unit
                        for (uint32_t u = 0; u < b; ++u) {
                            if (u < valid) {
                                noc_async_write(
                                    ring_base + ring_slot * out_page,
                                    out_acc.get_noc_addr(x_row + j * c_tiles + c + u),
                                    out_page);
                            }
                            ring_slot = (ring_slot + 1 == out_ring_tiles) ? 0 : ring_slot + 1;
                        }
                    }
                    noc_async_writes_flushed();  // the block's bytes have left L1: free the slots, no ack wait
                    for (uint32_t k = 0; k < b * n; ++k) {  // single-tile pops: a pop may not cross the ring end
                        cb_pop_front(cb_mixed_out_tiles, 1);
                    }
                }
                if constexpr (collapse) {  // the pass's y tiles
                    const uint32_t y_page = get_tile_size(cb_y_out);
                    const auto y_acc = TensorAccessor(y_args, get_common_arg_val<uint32_t>(3), y_page);
                    for (uint32_t u = 0; u < block_cols; ++u) {  // filler units' y (u >= cols_in_pass) dropped
                        cb_wait_front(cb_y_out, 1);
                        if (u < cols_in_pass) {
                            noc_async_write(
                                get_read_ptr(cb_y_out), y_acc.get_noc_addr(t * c_tiles + c_start + cb0 + u), y_page);
                            noc_async_writes_flushed();
                        }
                        cb_pop_front(cb_y_out, 1);
                    }
                }
            }

            if constexpr (has_w) {
                if (t > t_start && is_leader(t - 1)) {  // before partial(t): a 1-member chunk announces row t there
                    lead(t - 1);
                }
                // ---- partial(t): merge sum X'^2 into column ng, send to leader(t) ----
                cb_wait_front(cb_proj_out, 1);
                cb_wait_front(cb_sumsq_col, 1);
                volatile tt_l1_ptr uint32_t* proj =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_proj_out));
                volatile tt_l1_ptr uint32_t* sq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_sumsq_col));
                for (uint32_t row = 0; row < 32; ++row) {
                    proj[tile_word(row, ng)] = sq[tile_word(row, 0)];
                }
                const uint32_t leader_q = chunk_first + (t % chunk_size);
                const uint32_t bit = 1u << (t % 32);
                volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_ready);
                while ((*ready & bit) == 0) {
                    invalidate_l1_cache();
                }
                noc_semaphore_inc(get_noc_addr(sem_ready), 0u - bit);  // consume the announcement (atomic)
                noc_async_write(
                    get_read_ptr(cb_proj_out), member_noc(leader_q, leader_slots_base + my_pos * f32_page), f32_page);
                noc_async_write_barrier();
                noc_semaphore_inc(member_noc(leader_q, sem_leader), 1);
                cb_pop_front(cb_proj_out, 1);
                cb_pop_front(cb_sumsq_col, 1);
            }
        }
    }
    if constexpr (has_w) {
        if (t_count > 0 && is_leader(t_start + t_count - 1)) {  // the lagged duty of my last row
            lead(t_start + t_count - 1);
        }
    }

    // ---- root work, deferred past my own rows (never on the grid's critical path): for every row t I root
    // (t - t_start = member_q + k*Q), the chunk sums landed in region k of cb_root_slots -> compute; stats (+ hc)
    // -> DRAM ----
    if constexpr (has_w) {
        const uint32_t rooted = member_q < t_count ? (t_count - member_q + group_q - 1) / group_q : 0;
        if (rooted > 0) {
            volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_root);
            noc_semaphore_wait_min(cnt, rooted * num_chunks);
            noc_semaphore_set(cnt, 0);
            cb_reserve_back(cb_root_slots, rooted * num_chunks);
            cb_push_back(cb_root_slots, rooted * num_chunks);
        }
        const auto stats_acc = TensorAccessor(stats_args, get_common_arg_val<uint32_t>(1), f32_page);
        [[maybe_unused]] const auto hc_acc = TensorAccessor(hc_args, get_common_arg_val<uint32_t>(2), f32_page);
        for (uint32_t k = 0; k < rooted; ++k) {
            const uint32_t t = t_start + member_q + k * group_q;
            cb_wait_front(cb_stats, 1);
            noc_async_write(get_read_ptr(cb_stats), stats_acc.get_noc_addr(t), f32_page);
            if constexpr (collapse) {
                // stats tile (token l = row, value v = column) -> SFPU lane layout (value v = vector v, token = lane)
                uint32_t* st = reinterpret_cast<uint32_t*>(get_read_ptr(cb_stats));
                cb_reserve_back(cb_hc_in, 1);
                uint32_t* lanes = reinterpret_cast<uint32_t*>(get_write_ptr(cb_hc_in));
                for (uint32_t v = 0; v <= ng; ++v) {
                    permute_vector<true>(st + tile_word(0, v), lanes + lane_word(v, 0));
                }
                cb_push_back(cb_hc_in, 1);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_stats, 1);
            if constexpr (collapse) {
                cb_wait_front(cb_hc_out, 1);
                uint32_t* lanes = reinterpret_cast<uint32_t*>(get_read_ptr(cb_hc_out));
                uint32_t* hct = reinterpret_cast<uint32_t*>(get_write_ptr(cb_hc_tile));
                for (uint32_t v = 0; v < ng; ++v) {  // padding columns ng.. are left as they are
                    permute_vector<false>(hct + tile_word(0, v), lanes + lane_word(v, 0));
                }
                noc_async_write(get_write_ptr(cb_hc_tile), hc_acc.get_noc_addr(t), f32_page);
                noc_async_write_barrier();
                cb_pop_front(cb_hc_out, 1);
            }
        }
    }
    noc_async_write_barrier();
}
