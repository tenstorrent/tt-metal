// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post feed port reader (RISCV_1, NoC0), regime R4 feed-DRAM-staged.
//
// One port core per (direction slot, link). The workers stage the y tiles bound for other chips in y_output and,
// behind their write barrier, write each row count they finished into their slot of every port's cb_poll (a
// per-worker count: a shared counter would let fast workers' later rows stand in for a slow worker's row). Wave by
// wave this reader waits (local scan) until every worker has published the wave, then stages the items of its
// destinations (the shared walk, this link's share) into the chunk ring for the sender:
//   D pages   one y_output page each when chip me's rows start on a tile boundary; otherwise (T % 32 != 0) the
//             (at most two) y tiles the D tile's rows span land in cb_yscr and their face rows are copied into
//             place in L1 (Refinement 2b);
//   pieces    my rows of a shared D tile row, run_pages columns, in the landing's compact layout.
// Transport structure: fabric_all_gather's port reader.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/ttnn/operations/mhc_post/kernels/mhc_post_feed_walk.hpp"

void kernel_main() {
    constexpr uint32_t cb_chunks = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t run_pages = get_compile_time_arg_val(2);
    constexpr uint32_t num_banks = get_compile_time_arg_val(3);
    constexpr uint32_t links = get_compile_time_arg_val(4);
    constexpr uint32_t wave_rows = get_compile_time_arg_val(5);
    constexpr uint32_t max_rows = get_compile_time_arg_val(6);
    constexpr uint32_t group_q = get_compile_time_arg_val(7);
    constexpr uint32_t sem_rows_id = get_compile_time_arg_val(8);   // workers' arm semaphore
    constexpr uint32_t cb_poll = get_compile_time_arg_val(9);       // the workers' published-row counts, 16 B each
    constexpr uint32_t cb_yscr = get_compile_time_arg_val(10);      // private scratch: 2 y tiles per staged tile
    constexpr uint32_t read_batch = get_compile_time_arg_val(11);   // chunks in flight per read barrier
    constexpr uint32_t ring_chunks = get_compile_time_arg_val(12);  // chunk slots of cb_chunks
    constexpr auto y_args = TensorAccessorArgs<13>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;
    static_assert(page_bytes == 2048, "the feed moves bf16 tiles");

    size_t a = 0;
    mhc_feed::Geom g;
    g.c_tiles = get_arg_val<uint32_t>(a++);
    g.blk_tiles = get_arg_val<uint32_t>(a++);
    g.ht = get_arg_val<uint32_t>(a++);
    g.g = get_arg_val<uint32_t>(a++);
    g.me = get_arg_val<uint32_t>(a++);
    g.r = get_arg_val<uint32_t>(a++);
    g.rows_base = get_arg_val<uint32_t>(a++);
    g.rows_extra = get_arg_val<uint32_t>(a++);
    g.t_rows = get_arg_val<uint32_t>(a++);
    g.htd = get_arg_val<uint32_t>(a++);
    g.nimg = get_arg_val<uint32_t>(a++);
    g.nsh = get_arg_val<uint32_t>(a++);
    const uint32_t link = get_arg_val<uint32_t>(a++);
    const uint32_t y_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n_dest = get_arg_val<uint32_t>(a++);
    const uint32_t dest_idx = a;  // destination positions d
    a += n_dest;
    // used workers in L order (x | y << 16): worker k is in row group k / group_q. Common RT args (the same table on
    // every port): as per-core args they set the program's unique-RTA region, which counts twice against the
    // kernel-config ring (trace replays overlap only if it fits: Refinement 5)

    const auto y_acc = TensorAccessor(y_args, y_addr, page_bytes);
    const uint32_t num_workers = g.r * group_q;
    const uint32_t poll = get_write_ptr(cb_poll);
    // Publication (push): every worker k writes its published-row count into slot k of this core's cb_poll (16 B a
    // slot, the same address on every core) after each row, behind its y write barrier. Arm: zero the slots, then
    // count one on every worker's SEM_FEED_ROWS; a worker writes its first count only once every port armed it, so no
    // slot holds a previous call's count when it is read. The wave wait is then a local scan (no NoC polling).
    const uint32_t rows_sem = get_semaphore(sem_rows_id);
    for (uint32_t k = 0; k < num_workers; ++k) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(poll + k * 16) = 0;
    }
    for (uint32_t k = 0; k < num_workers; ++k) {
        const uint32_t xy = get_common_arg_val<uint32_t>(k);
        noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, rows_sem), 1);
    }
    // wait until every worker k published min(i1, rows of its group) rows; `ready_rows` = rows every worker has
    // published (a worker with all its rows counts as max_rows)
    uint32_t ready_rows = 0;
    auto wait_wave = [&](uint32_t i1) {
        while (ready_rows < i1) {
            invalidate_l1_cache();
            ready_rows = max_rows;
            for (uint32_t k = 0; k < num_workers; ++k) {
                const uint32_t cnt = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(poll + k * 16);
                if (cnt < mhc_feed::group_rows(g, k / group_q) && cnt < ready_rows) {
                    ready_rows = cnt;
                }
            }
        }
    };

    const uint32_t scr = get_write_ptr(cb_yscr);
    auto y_page = [&](uint32_t d, uint32_t img, uint32_t t, uint32_t col) {
        return y_acc.get_noc_addr((img * g.ht + t) * g.c_tiles + d * g.blk_tiles + col);
    };
    // my y rows [y0, y0 + n) of column col (n >= 1, at most two tile rows) -> scratch pair k; returns the source
    // row address function over rows y0 % 32 .. (rows >= 32 in the second tile)
    auto fetch = [&](uint32_t d, uint32_t img, uint32_t col, uint32_t y0, uint32_t n, uint32_t k) {
        const uint32_t s0 = scr + 2 * k * page_bytes;
        noc_async_read(y_page(d, img, y0 / 32, col), s0, page_bytes);
        if ((y0 + n - 1) / 32 != y0 / 32) {
            noc_async_read(y_page(d, img, y0 / 32 + 1, col), s0 + page_bytes, page_bytes);
        }
    };
    auto src_row = [&](uint32_t k) {
        const uint32_t s0 = scr + 2 * k * page_bytes;
        return [s0](uint32_t r, uint32_t h) {
            return r < 32 ? mhc_feed::tile_row_addr(s0, r, h) : mhc_feed::tile_row_addr(s0 + page_bytes, r - 32, h);
        };
    };

    // Whole-page chunks (every page one whole y tile) are read `read_batch` at a time behind one barrier; a chunk
    // occupies one ring slot (the sender walks the slots in the same order) and is pushed alone (a push may not
    // cross the ring end).
    const uint32_t ring_base = get_write_ptr(cb_chunks);  // nothing reserved yet: the ring's base
    uint32_t slot = 0, pending = 0;  // ring slot of the first unpushed chunk; chunks read but not pushed
    auto flush = [&]() {
        if (pending > 0) {
            noc_async_read_barrier();
            for (uint32_t k = 0; k < pending; ++k) {
                cb_push_back(cb_chunks, run_pages);
            }
            slot = (slot + pending) % ring_chunks;
            pending = 0;
        }
    };

    for (uint32_t i0 = 0, w = 0; i0 < max_rows; i0 += wave_rows, ++w) {
        const uint32_t i1 = (i0 + wave_rows) < max_rows ? (i0 + wave_rows) : max_rows;
        wait_wave(i1);  // the wave is in y_output
        for (uint32_t k = 0; k < n_dest; ++k) {
            const uint32_t d = get_arg_val<uint32_t>(dest_idx + k);
            uint32_t ci = 0;  // item index within (wave, destination): the sender counts the same way
            mhc_feed::for_each_item<num_banks, run_pages, wave_rows>(
                g, w, ci, [&](uint32_t c, const mhc_feed::Item& it) {
                    if (c % links != link) {
                        return;
                    }
                    bool aligned = !it.piece;  // every page of the chunk is one whole y tile
                    for (uint32_t p = 0; aligned && p < it.n; ++p) {
                        const uint32_t drow = (it.j + p * num_banks) / g.blk_tiles;
                        aligned = (32 * (drow - (drow / g.htd) * g.htd) - g.me * g.t_rows) % 32 == 0;
                    }
                    if (!aligned) {
                        flush();
                    }
                    cb_reserve_back(cb_chunks, run_pages * (pending + 1));
                    const uint32_t l1 = ring_base + ((slot + pending) % ring_chunks) * chunk_bytes;
                    if (aligned) {
                        for (uint32_t p = 0; p < it.n; ++p) {
                            const uint32_t jp = it.j + p * num_banks;
                            const uint32_t drow = jp / g.blk_tiles, col = jp - drow * g.blk_tiles;
                            const uint32_t img = drow / g.htd, row = drow - img * g.htd;
                            noc_async_read(
                                y_page(d, img, (32 * row - g.me * g.t_rows) / 32, col),
                                l1 + p * page_bytes,
                                page_bytes);
                        }
                        if (++pending == read_batch) {
                            flush();
                        }
                        return;
                    }
                    slot = (slot + 1) % ring_chunks;  // pending == 0 here: this chunk is pushed below
                    if (!it.piece) {
                        bool shifted = false;
                        for (uint32_t p = 0; p < it.n; ++p) {
                            const uint32_t jp = it.j + p * num_banks;
                            const uint32_t drow = jp / g.blk_tiles, col = jp - drow * g.blk_tiles;
                            const uint32_t img = drow / g.htd, row = drow - img * g.htd;
                            const uint32_t y0 = 32 * row - g.me * g.t_rows;  // alone: my rows start at tile row 0
                            if (y0 % 32 == 0) {
                                noc_async_read(y_page(d, img, y0 / 32, col), l1 + p * page_bytes, page_bytes);
                            } else {
                                fetch(d, img, col, y0, mhc_feed::rows_b(g, g.me, row), p);
                                shifted = true;
                            }
                        }
                        noc_async_read_barrier();
                        if (shifted) {
                            for (uint32_t p = 0; p < it.n; ++p) {
                                const uint32_t jp = it.j + p * num_banks;
                                const uint32_t drow = jp / g.blk_tiles;
                                const uint32_t row = drow - (drow / g.htd) * g.htd;
                                const uint32_t y0 = 32 * row - g.me * g.t_rows;
                                if (y0 % 32 != 0) {
                                    const uint32_t dst = l1 + p * page_bytes;
                                    mhc_feed::copy_rows(
                                        0,
                                        y0 % 32,
                                        mhc_feed::rows_b(g, g.me, row),
                                        [dst](uint32_t r, uint32_t h) { return mhc_feed::tile_row_addr(dst, r, h); },
                                        src_row(p));
                                }
                            }
                            noc_async_read_barrier();
                        }
                    } else {
                        const uint32_t y0 = 32 * it.row + it.a - g.me * g.t_rows;
                        const uint32_t n = it.b - it.a;
                        for (uint32_t col = it.c0; col < it.c1; ++col) {
                            fetch(d, it.img, col, y0, n, col - it.c0);
                        }
                        noc_async_read_barrier();
                        for (uint32_t col = it.c0; col < it.c1; ++col) {
                            const uint32_t blk = l1 + (col - it.c0) * n * mhc_feed::TILE_ROW_BYTES;
                            const uint32_t ra = it.a, rb = it.b;
                            mhc_feed::copy_rows(
                                ra,
                                y0 % 32,
                                n,
                                [blk, ra, rb](uint32_t r, uint32_t h) {
                                    return mhc_feed::compact_row_addr(blk, ra, rb, r, h);
                                },
                                src_row(col - it.c0));
                        }
                        noc_async_read_barrier();
                    }
                    cb_push_back(cb_chunks, run_pages);
                });
        }
        flush();  // the wave's last chunks go out before the next wave's wait
    }
}
