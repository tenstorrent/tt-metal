// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// q_stem compute-core writer (RISCV_0 / NoC1). Block ops:
//   s1_handoff_block  (stage-1 cores) per q block, once compute has packed it into cb_q_slice:
//                       remote quarter -> one block-ready increment on every port of the destination's direction
//                                         (the ports pull the head-rows from my cb_q_slice themselves)
//                       own quarter    -> write my columns of each head-row into the destination core's landing
//                                         scratch (same address on every core), then increment its arrive[p] by
//                                         the tiles written (after the write barrier)
//   kpart_send_block  (K-split, partner groups) per q block: my partial (cb_q_kpart, m x cols) -> the owner core's
//                       cb_q_recv<g> slot (same CB address on every stage-1 core), once the owner has credited the
//                       slot; then one increment of the owner's ready count for my group (the owner's reader
//                       credits the slots and pushes the landed blocks: kpart_recv_service, q_stem_reader.cpp)
//   store_row_block   per (head, row): prefix N-blocks + rope tiles -> output pages, one flush per block
//                       TILE:      one page per tile
//                       ROW_MAJOR: the compute untilized each block into 32 sticks (cb_prefix_rm / cb_rope_rm);
//                                  each output row's stick is written as segments (prefix N-block(s) | rope) at
//                                  their byte offsets inside the (N + Dr) * 2 B output page; full_stick (one
//                                  N-block per row): one [prefix | rope] block, each stick = one whole-page write

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/ttnn/operations/q_stem/perf_experiments/qb_startup/q_stem_base/kernels/q_stem_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_q_slice = get_compile_time_arg_val(0);
    constexpr uint32_t cb_prefix_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rope_out = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t H = get_compile_time_arg_val(5);
    constexpr uint32_t N_t = get_compile_time_arg_val(6);
    constexpr uint32_t Dr_t = get_compile_time_arg_val(7);
    constexpr uint32_t s3n = get_compile_time_arg_val(8);
    constexpr uint32_t n_nb = get_compile_time_arg_val(9);
    constexpr uint32_t n_ports = get_compile_time_arg_val(10);
    constexpr uint32_t sem_block_ready_id = get_compile_time_arg_val(11);
    constexpr uint32_t B = get_compile_time_arg_val(12);
    constexpr uint32_t qrt = get_compile_time_arg_val(13);
    constexpr uint32_t m = get_compile_time_arg_val(14);
    constexpr uint32_t G = get_compile_time_arg_val(15);
    constexpr uint32_t hw = get_compile_time_arg_val(16);
    constexpr bool out_rm = get_compile_time_arg_val(17) != 0;
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(18);  // tile (TILE) or (N + Dr) * 2 B stick (RM)
    constexpr uint32_t cb_prefix_rm = get_compile_time_arg_val(19);
    constexpr uint32_t cb_rope_rm = get_compile_time_arg_val(20);
    constexpr bool full_stick = get_compile_time_arg_val(21) != 0;  // RM: one [prefix | rope] block per row
    constexpr uint32_t ts = get_compile_time_arg_val(22);           // stage-3 consumption order (q_stem::RowOrder)
    constexpr uint32_t Sq = get_compile_time_arg_val(23);  // output rows per (image, head); qrt = ceil(Sq / 32)
    constexpr auto out_args = TensorAccessorArgs<24>();
    constexpr uint32_t stick_tile_bytes = tile_bytes / 32;  // one tile-width of one row-major stick (bf16: 64 B)
    constexpr uint32_t out_w = N_t + Dr_t;
    constexpr uint32_t rows_per_head = B * qrt;
    constexpr uint32_t ksplit = get_named_compile_time_arg_val("ksplit");
    constexpr uint32_t cb_q_kpart = get_named_compile_time_arg_val("cb_q_kpart");
    constexpr uint32_t cb_q_recv1 = get_named_compile_time_arg_val("cb_q_recv1");
    constexpr uint32_t cb_q_recv2 = get_named_compile_time_arg_val("cb_q_recv2");
    constexpr uint32_t kpart_tiles = get_named_compile_time_arg_val("kpart_tiles");  // cb_q_kpart / _recv<g> pages
    constexpr uint32_t kpart_tile_bytes = get_named_compile_time_arg_val("kpart_tile_bytes");
    constexpr uint32_t sem_kpart_ready = get_named_compile_time_arg_val("sem_kpart_ready");  // + (g - 1)
    constexpr uint32_t sem_kpart_credit = get_named_compile_time_arg_val("sem_kpart_credit");

    size_t a = 0;
    const uint32_t p = get_arg_val<uint32_t>(a++);
    const uint32_t cols = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t landing_addr = get_arg_val<uint32_t>(a++);
    const uint32_t arrive_p_addr = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t kgroup = get_arg_val<uint32_t>(a++);  // K group (0: owner of q)
    // K-split peers (packed virtual xy): an owner's partners (groups 1 .. ksplit-1), a partner's owner
    [[maybe_unused]] uint32_t kpeer[ksplit > 1 ? ksplit - 1 : 1];
    for (uint32_t g = 0; g + 1 < ksplit; ++g) {
        kpeer[g] = get_arg_val<uint32_t>(a++);
    }
    uint32_t dirmap[G];
    for (uint32_t i = 0; i < G; ++i) {
        dirmap[i] = get_arg_val<uint32_t>(a++);
    }
    uint32_t rorder[G];
    for (uint32_t i = 0; i + 1 < G; ++i) {
        rorder[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t ports_idx = a;
    a += 2 * n_ports;
    const uint32_t n_pieces = get_arg_val<uint32_t>(a++);
    const uint32_t pieces_idx = a;
    a += 5 * n_pieces;
    const uint32_t hpc = get_arg_val<uint32_t>(a++);
    const uint32_t n_r = get_arg_val<uint32_t>(a++);  // remote heads (first, consumed row-major across heads)
    const uint32_t heads_idx = a;

    // ---- stage 1: K-split partner -----------------------------------------------------------------------------------
    if constexpr (ksplit > 1) {
        if (cols > 0 && kgroup > 0) {
            const uint32_t blk = m * cols;
            const uint32_t recv_cb = kgroup == 1 ? cb_q_recv1 : cb_q_recv2;
            const uint32_t recv_base = get_write_ptr(recv_cb);  // fifo start (unused here: the owner's address)
            volatile tt_l1_ptr uint32_t* credit =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_kpart_credit));
            const uint32_t ox = q_stem::unpack_x(kpeer[0]), oy = q_stem::unpack_y(kpeer[0]);
            const uint64_t owner_ready = get_noc_addr(ox, oy, get_semaphore(sem_kpart_ready + kgroup - 1));
            for (uint32_t j = 0; j < num_blocks; ++j) {
                cb_wait_front(cb_q_kpart, blk);
                noc_semaphore_wait_min(credit, j + 1);
                const uint32_t dst = recv_base + ((j * blk) % kpart_tiles) * kpart_tile_bytes;
                noc_async_write(get_read_ptr(cb_q_kpart), get_noc_addr(ox, oy, dst), blk * kpart_tile_bytes);
                noc_async_write_barrier();
                noc_semaphore_inc(owner_ready, 1);
                cb_pop_front(cb_q_kpart, blk);
            }
            noc_async_atomic_barrier();
        }
    }

    // ---- stage 1 hand-off -------------------------------------------------------------------------------------------
    if (cols > 0 && kgroup == 0) {
        MaybeDeviceZoneScope("WR_S1_HANDOFF");  // q blocks: port block-ready reports + own-quarter local delivery
        const uint32_t q_base = get_read_ptr(cb_q_slice);  // never popped before the end: block j at j*m*cols tiles
        // block-ready ring on the ports (host BLOCK_READY_RING): my idx-th report for a direction -> semaphore
        // sem_block_ready_id + idx % ring, so a port waiting for report idx counts no later block of a faster owner
        constexpr uint32_t ready_ring = get_named_compile_time_arg_val("ready_ring");
        uint32_t reported[2] = {0, 0};  // per direction (FWD / BWD)
        q_stem::for_each_block(B, qrt, m, G, p, rorder, [&](uint32_t j, uint32_t b, uint32_t d, uint32_t r0) {
            cb_wait_front(cb_q_slice, (j + 1) * m * cols);
            if (d != p) {
                const uint32_t block_ready = get_semaphore(sem_block_ready_id + reported[dirmap[d]]++ % ready_ring);
                for (uint32_t k = 0; k < n_ports; ++k) {
                    if (get_arg_val<uint32_t>(ports_idx + 2 * k + 1) == dirmap[d]) {
                        const uint32_t pc = get_arg_val<uint32_t>(ports_idx + 2 * k);
                        noc_semaphore_inc(get_noc_addr(q_stem::unpack_x(pc), q_stem::unpack_y(pc), block_ready), 1);
                    }
                }
                return;
            }
            for (uint32_t i = 0; i < m; ++i) {
                const uint32_t row = b * qrt + r0 + i;
                const uint32_t src_row = q_base + (j * m + i) * cols * tile_bytes;
                for (uint32_t e = 0; e < n_pieces; ++e) {
                    const uint32_t t = pieces_idx + 5 * e;
                    const uint32_t lc = get_arg_val<uint32_t>(t);
                    const uint32_t n = get_arg_val<uint32_t>(t + 1);
                    const uint32_t dc = get_arg_val<uint32_t>(t + 2);
                    const uint32_t slot = get_arg_val<uint32_t>(t + 3) + row * get_arg_val<uint32_t>(t + 4);
                    noc_async_write(
                        src_row + lc * tile_bytes,
                        get_noc_addr(q_stem::unpack_x(dc), q_stem::unpack_y(dc), landing_addr + slot * tile_bytes),
                        n * tile_bytes);
                }
            }
            noc_async_write_barrier();
            for (uint32_t e = 0; e < n_pieces; ++e) {
                const uint32_t t = pieces_idx + 5 * e;
                const uint32_t dc = get_arg_val<uint32_t>(t + 2);
                noc_semaphore_inc(
                    get_noc_addr(q_stem::unpack_x(dc), q_stem::unpack_y(dc), arrive_p_addr),
                    m * get_arg_val<uint32_t>(t + 1));
            }
        });
        noc_async_atomic_barrier();
        cb_pop_front(cb_q_slice, num_blocks * m * cols);
    }

    // ---- stage 3 output ---------------------------------------------------------------------------------------------
    const auto out = TensorAccessor(out_args, out_addr, out_page_bytes);
    // TILE: n_tiles tile pages from the front of cb, starting at output tile page `page`
    auto store_tiles = [&](uint32_t cb, uint32_t n_tiles, uint32_t page) {
        cb_wait_front(cb, n_tiles);
        uint32_t l1 = get_read_ptr(cb);
        for (uint32_t c = 0; c < n_tiles; ++c) {
#ifndef ABLATE_OUT_WRITE
            noc_async_write(l1, out.get_noc_addr(page + c), tile_bytes);
#endif
            l1 += tile_bytes;
        }
        noc_async_writes_flushed();  // the tiles have left L1 (acks are collected once, at the end)
        cb_pop_front(cb, n_tiles);
    };
    // ROW_MAJOR: one untilized block of n_tiles tile-widths = 32 sticks of n_tiles * 64 B, the first n_sticks of
    // them (32, or the valid rows of a quarter's last row tile when Sq % 32 != 0) written into sticks
    // stick0 .. stick0 + n_sticks - 1 at byte offset col_byte (one write per stick, one flush)
    auto store_sticks = [&](uint32_t cb, uint32_t n_tiles, uint32_t stick0, uint32_t n_sticks, uint32_t col_byte) {
        const uint32_t seg = n_tiles * stick_tile_bytes;
        cb_wait_front(cb, n_tiles);
        uint32_t l1 = get_read_ptr(cb);
        for (uint32_t i = 0; i < n_sticks; ++i) {
#ifndef ABLATE_OUT_WRITE
            noc_async_write(l1, out.get_noc_addr(stick0 + i, col_byte), seg);
#endif
            l1 += seg;
        }
        noc_async_writes_flushed();
        cb_pop_front(cb, n_tiles);
    };
    // store_row_block: one (head, row) = n_nb prefix blocks + the rope tiles, in the compute's consumption order
    auto store_row = [&](uint32_t u, uint32_t t) {
        const uint32_t h = get_arg_val<uint32_t>(heads_idx + u);
        const uint32_t b = t / qrt;
        const uint32_t r = t - b * qrt;
        const uint32_t row_tile = (b * H + h) * qrt + r;  // output tile row (32 output rows)
        if constexpr (out_rm) {
            const uint32_t stick0 = (b * H + h) * Sq + r * 32;  // output row = stick
            const uint32_t n_sticks = Sq - r * 32 < 32 ? Sq - r * 32 : 32;
            if constexpr (full_stick) {  // whole output pages, one write each
                store_sticks(cb_prefix_rm, out_w, stick0, n_sticks, 0);
                return;
            }
            for (uint32_t nb = 0; nb < n_nb; ++nb) {
                store_sticks(cb_prefix_rm, s3n, stick0, n_sticks, nb * s3n * stick_tile_bytes);
            }
            store_sticks(cb_rope_rm, Dr_t, stick0, n_sticks, N_t * stick_tile_bytes);
        } else {
            const uint32_t page0 = row_tile * out_w;
            for (uint32_t nb = 0; nb < n_nb; ++nb) {
                store_tiles(cb_prefix_out, s3n, page0 + nb * s3n);
            }
            store_tiles(cb_rope_out, Dr_t, page0 + N_t);
        }
    };
    const q_stem::RowOrder order(n_r, hpc, rows_per_head, ts);
    MaybeDeviceZoneScope("WR_S3_OUT");  // output rows (waits on compute inside store_*) + final barrier
    for (uint32_t c = 0; c < order.total(); ++c) {
        uint32_t u, t;
        order.at(c, u, t);
        store_row(u, t);
    }
    noc_async_write_barrier();
}
