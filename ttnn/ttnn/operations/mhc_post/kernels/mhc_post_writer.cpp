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
//   root(t) = member ((t - t_start) mod Q). Without norm its work is DEFERRED past the root's own rows so that no
//     core's row loop ever waits on a root: every rooted row k has its own region of cb_root_slots (no reuse, no
//     announcement); at the end the root hands them to compute, writes stats; collapse: permutes the stats tile
//     into SFPU lane layout for the hc step and the hc result back into a row tile; writes hc. Knob (norm,
//     root_lag > 0, Refinement 3): LAGGED root_lag rows — virtual row i runs the root work of row i - root_lag, so
//     rstd returns within the row loop. All semaphores are back to 0 at kernel end.
// norm: the partial's column ng+1 carries sum y_b^2 (second page of cb_sumsq_col). The root's hc step leaves
//   rstd in lane vector ng+1; the root copies it to token order (128 B) and unicasts it into slot (t - t_start) of
//   every group member's cb_rstd_land, then incs their SEM_RSTD_COUNT. Deferred (default) the member's reader
//   expands every row's rstd once all landed, and the writer writes y after the root work. Lagged: the root sets bit
//   (t - t_start) % 32 instead and the member's writer expands the row's rstd norm_lag rows after it, then writes
//   its y — so the y writes and the feed publication stream with the rows.
// feed (R4): y tile (t, c) of column block p = c / Cb goes to the feed output D at rows [me*T, ..) of block me
//   when p == me (written locally; the host passes me = G when T % 32 != 0, where my rows do not start on a D tile
//   boundary and the ports place my own block too), else to y_output (staging, at its natural page) for the
//   ports; after each of my
//   rows (behind a write barrier) I write my row count into my slot of every port's cb_feed_poll (a per-worker
//   count: a shared counter could not tell "every worker published row i" from "fast workers published rows i and
//   i+1"); the ports arm me (SEM_FEED_ROWS) once their slots are zeroed.

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
    [[maybe_unused]] constexpr uint32_t block_rows = get_compile_time_arg_val(2);
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
    constexpr bool norm = get_compile_time_arg_val(24) != 0;
    constexpr uint32_t cb_rstd_land = get_compile_time_arg_val(25);
    constexpr uint32_t cb_rstd_send = get_compile_time_arg_val(26);
    constexpr uint32_t sem_rstd_count_id = get_compile_time_arg_val(27);
    constexpr bool feed = get_compile_time_arg_val(28) != 0;
    constexpr uint32_t sem_feed_rows_id = get_compile_time_arg_val(29);
    constexpr uint32_t cb_rstd_col = get_compile_time_arg_val(30);
    // norm schedule knob: root_lag == 0 deferred roots + norm tail; > 0 root work of row t after my row t + root_lag,
    // y of row t after my row t + norm_lag (see the compute)
    constexpr uint32_t root_lag = get_compile_time_arg_val(31);
    constexpr uint32_t norm_lag = get_compile_time_arg_val(32);
    constexpr bool lagged = norm && root_lag > 0;
    constexpr uint32_t sumsq_pages = norm ? 2 : 1;  // per row: sum X'^2 [, sum y_b^2]
    constexpr uint32_t rstd_bytes = 32 * sizeof(uint32_t);
    constexpr uint32_t cb_feed_poll = get_compile_time_arg_val(33);  // feed: the ports' published-row slots
    constexpr uint32_t cb_rstd_lane = get_compile_time_arg_val(34);  // norm: the root's rstd in lane layout
    constexpr auto out_args = TensorAccessorArgs<35>();
    [[maybe_unused]] constexpr auto stats_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto hc_args = TensorAccessorArgs<stats_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto y_args = TensorAccessorArgs<hc_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto d_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr uint32_t num_chunks = (group_q + fanin - 1) / fanin;
    // common args: out, stats, hc, y addresses; feed: D address, me, G, Cb, Ht; then the core table
    constexpr uint32_t FEED_BASE = 4;
    constexpr uint32_t FEED_PORTS = FEED_BASE + 5;  // feed: port count, then MAX_FEED_PORTS port coordinates
    constexpr uint32_t MAX_FEED_PORTS = 4;
    constexpr uint32_t COORD_BASE = feed ? FEED_PORTS + 1 + MAX_FEED_PORTS : FEED_BASE;

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

    // y tiles of one column pass of row t (block_cols tiles; the filler units' y, u >= cols, are dropped)
    [[maybe_unused]] auto write_y_pass = [&](uint32_t t, uint32_t cb0, uint32_t cols) {
        const uint32_t y_page = get_tile_size(cb_y_out);
        const auto y_acc = TensorAccessor(y_args, get_common_arg_val<uint32_t>(3), y_page);
        [[maybe_unused]] uint32_t me = 0, grp = 0, cb_tiles = 1, ht = 1, d_row = 0;
        if constexpr (feed) {
            me = get_common_arg_val<uint32_t>(FEED_BASE + 1);
            grp = get_common_arg_val<uint32_t>(FEED_BASE + 2);
            cb_tiles = get_common_arg_val<uint32_t>(FEED_BASE + 3);
            ht = get_common_arg_val<uint32_t>(FEED_BASE + 4);
            const uint32_t img = t / ht;
            d_row = (img * grp + me) * ht + (t - img * ht);  // my rows of the feed output, image img
        }
        for (uint32_t u = 0; u < block_cols; ++u) {
            cb_wait_front(cb_y_out, 1);
            if (u < cols) {
                const uint32_t c = c_start + cb0 + u;
                uint64_t dst;
                if constexpr (feed) {
                    if (c / cb_tiles == me) {  // my own block: straight into my feed output
                        const auto d_acc = TensorAccessor(d_args, get_common_arg_val<uint32_t>(FEED_BASE), y_page);
                        dst = d_acc.get_noc_addr(d_row * cb_tiles + (c - me * cb_tiles));
                    } else {  // staged for the ports
                        dst = y_acc.get_noc_addr(t * c_tiles + c);
                    }
                } else {
                    dst = y_acc.get_noc_addr(t * c_tiles + c);
                }
                noc_async_write(get_read_ptr(cb_y_out), dst, y_page);
                noc_async_writes_flushed();
            }
            cb_pop_front(cb_y_out, 1);
        }
    };
    // feed: my row's staged y has landed in DRAM -> my published-row count into my slot of every port's cb_feed_poll
    // (the first only once every port armed me: zeroed its slots, then counted on my SEM_FEED_ROWS)
    [[maybe_unused]] uint32_t published = 0;
    [[maybe_unused]] auto publish_row = [&]() {
        if constexpr (feed) {
            noc_async_write_barrier();
            const uint32_t n_ports = get_common_arg_val<uint32_t>(FEED_PORTS);
            if (published++ == 0) {
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_feed_rows_id)), n_ports);
            }
            const uint32_t slot = get_write_ptr(cb_feed_poll) + (group * group_q + member_q) * 16;
            for (uint32_t p = 0; p < n_ports; ++p) {
                const uint32_t xy = get_common_arg_val<uint32_t>(FEED_PORTS + 1 + p);
                noc_inline_dw_write(get_noc_addr(xy & 0xFFFF, xy >> 16, slot), published);
            }
        }
    };

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

    // ---- root work of row t (t - t_start = member_q + k*Q): the chunk sums landed in region k of cb_root_slots ->
    // compute; stats (+ hc) -> DRAM; norm: rstd -> every member ----
    [[maybe_unused]] auto root_step = [&](uint32_t t) {
        const auto stats_acc = TensorAccessor(stats_args, get_common_arg_val<uint32_t>(1), f32_page);
        [[maybe_unused]] const auto hc_acc = TensorAccessor(hc_args, get_common_arg_val<uint32_t>(2), f32_page);
        cb_wait_front(cb_stats, 1);
        noc_async_write(get_read_ptr(cb_stats), stats_acc.get_noc_addr(t), f32_page);
        if constexpr (collapse) {
            // stats tile (token l = row, value v = column) -> SFPU lane layout (value v = vector v, token = lane)
            uint32_t* st = reinterpret_cast<uint32_t*>(get_read_ptr(cb_stats));
            cb_reserve_back(cb_hc_in, 1);
            uint32_t* lanes = reinterpret_cast<uint32_t*>(get_write_ptr(cb_hc_in));
            for (uint32_t v = 0; v <= ng + (norm ? 1 : 0); ++v) {  // [norm: + sum y_b^2 -> rstd]
                permute_vector<true>(st + tile_word(0, v), lanes + lane_word(v, 0));
            }
            cb_push_back(cb_hc_in, 1);
        }
        if constexpr (norm) {
            // rstd (lane vector ng+1 of the compute's rstd step, run before the hc step) -> token order -> slot
            // (t - t_start) of every member's landing; the member's SEM_RSTD_COUNT gets bit (t - t_start) % 32 (rows
            // land in any order; a bit is consumed long before its row + 32 can land: that row needs the member's
            // own partial, norm_lag < 32 rows on), deferred a count (every row lands before any is read)
            cb_wait_front(cb_rstd_lane, 1);
            const uint32_t* rl = reinterpret_cast<const uint32_t*>(get_read_ptr(cb_rstd_lane));
            uint32_t* send = reinterpret_cast<uint32_t*>(get_write_ptr(cb_rstd_send));
            for (uint32_t l = 0; l < 32; ++l) {
                send[l] = rl[lane_word(ng + 1, l)];
            }
            const uint32_t land = get_write_ptr(cb_rstd_land) + (t - t_start) * rstd_bytes;
            for (uint32_t q = 0; q < group_q; ++q) {
                noc_async_write(get_write_ptr(cb_rstd_send), member_noc(q, land), rstd_bytes);
            }
            noc_async_write_barrier();  // landed before the bits (and before the next row reuses send)
            for (uint32_t q = 0; q < group_q; ++q) {
                noc_semaphore_inc(
                    member_noc(q, get_semaphore(sem_rstd_count_id)), lagged ? 1u << ((t - t_start) % 32) : 1u);
            }
            cb_pop_front(cb_rstd_lane, 1);
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
    };
    // lagged norm: rstd of my local row r (wait for bit r % 32) -> column 0 of a cb_rstd_col page (the compute's COL
    // broadcast; deferred, the reader expands every row after its rows)
    [[maybe_unused]] auto expand_rstd = [&](uint32_t r) {
        if constexpr (lagged) {
            const uint32_t bit = 1u << (r % 32);
            const uint32_t sem = get_semaphore(sem_rstd_count_id);
            volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
            while ((*landed & bit) == 0) {
                invalidate_l1_cache();
            }
            noc_semaphore_inc(get_noc_addr(sem), 0u - bit);  // consume (atomic: other rows' bits may land now)
            const volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_write_ptr(cb_rstd_land) + r * rstd_bytes);  // private landing, same address everywhere
            cb_reserve_back(cb_rstd_col, 1);
            uint32_t* dst = reinterpret_cast<uint32_t*>(get_write_ptr(cb_rstd_col));
            for (uint32_t l = 0; l < 32; ++l) {  // token l -> (row l, column 0): face 0 rows 0..15, face 2 16..31
                dst[(l / FACE_DIM) * 2 * FACE_WORDS + (l % FACE_DIM) * FACE_DIM] = src[l];
            }
            cb_push_back(cb_rstd_col, 1);
        }
    };
    // norm: the row's y (normalized by the compute) -> DRAM / feed, then publish it
    [[maybe_unused]] auto write_norm_row = [&](uint32_t r) {
        for (uint32_t cb0 = 0; cb0 < c_count; cb0 += block_cols) {
            write_y_pass(t_start + r, cb0, (c_count - cb0) < block_cols ? (c_count - cb0) : block_cols);
        }
        publish_row();
    };

    // Virtual rows (the compute walks the same order): rows 0..t_count-1 store X' and send their partial; with norm
    // the loop runs norm_lag rows further and virtual row i also runs the root work of row i - root_lag (before
    // partial(i): its chunk sums need only rows <= i - root_lag + 1 of the leaders) and the y of row i - norm_lag.
    const uint32_t vrows = t_count + (lagged ? norm_lag : 0);
    for (uint32_t i = 0; i < vrows; ++i) {
        const uint32_t t = t_start + i;
        if (i < t_count) {
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
                if constexpr (collapse && !norm) {  // the pass's y tiles
                    write_y_pass(t, cb0, cols_in_pass);
                }
            }
            if constexpr (collapse && !norm) {
                publish_row();
            }
        }

        if constexpr (has_w) {
            if (i > 0 && i <= t_count && is_leader(t - 1)) {  // before partial(t): a 1-member chunk announces row t
                lead(t - 1);
            }
            if constexpr (lagged) {
                const uint32_t rr = i - root_lag;  // root work of row rr, if I root it
                if (i >= root_lag && rr < t_count && rr % group_q == member_q) {
                    // only row rr's chunk sums can have landed: the next row I root (rr + Q, Q >= root_lag) needs my
                    // own partial(rr + Q), which comes after this
                    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_root);
                    noc_semaphore_wait_min(cnt, num_chunks);
                    noc_semaphore_set(cnt, 0);
                    cb_reserve_back(cb_root_slots, num_chunks);  // region rr / Q, in order
                    cb_push_back(cb_root_slots, num_chunks);
                    root_step(t_start + rr);
                }
            }
            if (i < t_count) {
                // ---- partial(t): merge sum X'^2 into column ng, send to leader(t) ----
                cb_wait_front(cb_proj_out, 1);
                cb_wait_front(cb_sumsq_col, sumsq_pages);
                volatile tt_l1_ptr uint32_t* proj =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_proj_out));
                volatile tt_l1_ptr uint32_t* sq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_sumsq_col));
                for (uint32_t row = 0; row < 32; ++row) {
                    proj[tile_word(row, ng)] = sq[tile_word(row, 0)];
                }
                if constexpr (norm) {  // sum y_b^2 -> column ng+1 (the second page)
                    for (uint32_t row = 0; row < 32; ++row) {
                        proj[tile_word(row, ng + 1)] = sq[f32_page / sizeof(uint32_t) + tile_word(row, 0)];
                    }
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
                cb_pop_front(cb_sumsq_col, sumsq_pages);
            }
        }
        if constexpr (lagged) {
            if (i >= norm_lag) {
                expand_rstd(i - norm_lag);
                write_norm_row(i - norm_lag);
            }
        }
    }

    // ---- !norm: root work, deferred past my own rows (never on the grid's critical path): every row t I root has
    // its own region of cb_root_slots, all handed to the compute at once ----
    if constexpr (has_w && !lagged) {
        if (t_count > 0 && is_leader(t_start + t_count - 1)) {  // the lagged duty of my last row
            lead(t_start + t_count - 1);
        }
        const uint32_t rooted = member_q < t_count ? (t_count - member_q + group_q - 1) / group_q : 0;
        if (rooted > 0) {
            volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_root);
            noc_semaphore_wait_min(cnt, rooted * num_chunks);
            noc_semaphore_set(cnt, 0);
            cb_reserve_back(cb_root_slots, rooted * num_chunks);
            cb_push_back(cb_root_slots, rooted * num_chunks);
        }
        for (uint32_t k = 0; k < rooted; ++k) {
            root_step(t_start + member_q + k * group_q);
        }
    }
    if constexpr (norm && !lagged) {  // deferred: the y rows in order (the reader expands rstd, the compute normalizes)
        for (uint32_t r = 0; r < t_count; ++r) {
            write_norm_row(r);
        }
    }
    noc_async_atomic_barrier();
    noc_async_write_barrier();
}
