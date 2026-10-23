// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm — worker writer (RISCV_0, NoC1).
//   publish_partials: the worker's fp32 partial row vectors (2 x 64 B half-rows per tile row) into its own slot of
//                     BOTH port boards of its link; one barrier, then one increment of each port's local counter.
//   fanout_inv_rms  : leader only — inv_rms half-rows into cb_inv_row's bytes on every core of the row-group
//                     (itself included): per rectangle of members, one linked multicast per tile row, then a
//                     multicast SET of sem_inv_ready to r + 1 (same NoC / VC / path as the data => it lands after
//                     it, no barrier); a 1-core rectangle gets the same as ordered unicasts. The leader is the
//                     row-group's last row-major core = the NoC1-start (high) corner of its rectangle.
//                     Raw NoC, not the mcast_pipe helper (Perf 2, perf_experiments/inv_fanout_mcast: fan-out latency
//                     1044 -> 420 ns at C = 8, h = 1, 10 concurrent leaders; mcast_pipe Counter 1318 ns): its Counter
//                     signal pays a write barrier per rectangle before a non-posted atomic inc, and its Flag signal
//                     resets the receiver's cell per receive (needs clear-before-next-set back-pressure the pipelined
//                     rounds do not have) and sources the SET from the sender's own ready cell (the leader's reader
//                     would see "ready" before its loopback data landed).
//   store_block     : output tiles -> DRAM, one batch (block row of w_act tiles) per barrier. A sharded output is
//                     packed by compute straight into the core's own shard (cb_out_tiles backs it): nothing to store.
//   pre-combine     : a worker's partials go to its row-group leader's cb_pregather (row = its chunk) instead of
//                     the ports; the leader publishes the row-group's combined vector (a second cb_partial_row push).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

constexpr uint32_t cb_partial_row = 6;
constexpr uint32_t cb_inv_row_out = 5;
constexpr uint32_t cb_inv_row = 8;
constexpr uint32_t cb_out_tiles = 11;
constexpr uint32_t sem_port_local_id = 0;
constexpr uint32_t sem_inv_ready_id = 2;
constexpr uint32_t sem_inv_sig_id = 9;         // leader scratch: the value (r + 1) multicast-set into sem_inv_ready
constexpr uint32_t sem_port_local_odd_id = 3;  // odd rounds (see the port kernel: per-parity local counters)
constexpr uint32_t cb_pregather = 12;
constexpr uint32_t sem_pre_ready_id = 5;
constexpr uint32_t sem_pre_ready_odd_id = 6;
constexpr uint32_t fp32_tile_bytes = 4096;
constexpr uint32_t face_bytes_fp32 = 1024;
constexpr uint32_t half_row_bytes = 64;
constexpr uint32_t inv_row_bytes = face_bytes_fp32 + half_row_bytes;  // row 0 of faces 0 and 1 (one fanout write)

void kernel_main() {
    constexpr uint32_t block_h = get_compile_time_arg_val(0);
    constexpr uint32_t block_w = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t x_page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(4);  // partial vectors per tile row on the board
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t num_rounds = get_compile_time_arg_val(6);
    constexpr uint32_t round_stride = get_compile_time_arg_val(7);
    constexpr uint32_t inv_rounds = get_compile_time_arg_val(8);   // cb_inv_row depth in rounds (1 or 2)
    constexpr uint32_t round_bytes = get_compile_time_arg_val(9);  // board region of one round (G slots)
    constexpr bool out_sharded = get_compile_time_arg_val(10) != 0;
    constexpr bool pre_combine = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t n_pre_tiles = get_compile_time_arg_val(12);
    // store batches release their CB pages once the data has left L1 (flushed) rather than once DRAM acked it
    // (barrier); one barrier at the end of the kernel covers every store
    constexpr bool store_flush = get_compile_time_arg_val(13) != 0;
    constexpr bool dram_stagger = get_compile_time_arg_val(14) != 0;  // see the reader
    constexpr auto out_args = TensorAccessorArgs<15>();

    uint32_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t row0 = get_arg_val<uint32_t>(a++);
    const uint32_t row_end = get_arg_val<uint32_t>(a++);
    const uint32_t col0 = get_arg_val<uint32_t>(a++);
    const uint32_t w_act = get_arg_val<uint32_t>(a++);
    const uint32_t my_chunk = get_arg_val<uint32_t>(a++);   // W-chunk index in the row-group
    const uint32_t leader_xy = get_arg_val<uint32_t>(a++);  // the row-group leader (pre-combine target)
    const uint32_t my_slot = get_arg_val<uint32_t>(a++);
    const uint32_t board_addr = get_arg_val<uint32_t>(a++);
    const uint32_t rt_base = get_arg_val<uint32_t>(a++);
    const uint32_t fwd_xy = get_arg_val<uint32_t>(a++);
    const uint32_t bwd_xy = get_arg_val<uint32_t>(a++);
    const bool is_leader = get_arg_val<uint32_t>(a++) != 0;
    const uint32_t n_rects = get_arg_val<uint32_t>(a++);  // fan-out rectangles: (sx, sy, ex, ey, count, has_self)
    const uint32_t rects_idx = a;
    // board position of my vector: my chunk, or (pre-combine) the row-group's single combined vector
    const uint32_t wc = pre_combine ? 0 : my_chunk;

    auto rows_of = [&](uint32_t r) -> uint32_t {
        const uint32_t first = row0 + r * round_stride;
        if (first >= row_end) {
            return 0;
        }
        return (row_end - first) < block_h ? (row_end - first) : block_h;
    };
    // ---- publish_partials(r): own slot of round r on BOTH ports of the link, then one increment of each port's
    // local counter (an empty round still increments: the ports count every worker every round)
    const uint32_t ports[2] = {fwd_xy, bwd_xy};
    // cb_partial_row pages per publish: h, or (pre-combine) always block_h. The pre-combine leader's stream mixes a
    // short last round with a later full combined vector, and the reduce packs its block linearly from the write
    // pointer, so compute pads every push to block_h to keep each block at the FIFO base.
    auto partial_pages = [&](uint32_t h) -> uint32_t { return pre_combine ? block_h : h; };
    auto publish = [&](uint32_t r) {
        const uint32_t h = rows_of(r);
        if (h > 0) {
            {
                MaybeDeviceZoneScope("W_pwait");
                cb_wait_front(cb_partial_row, partial_pages(h));
            }
            MaybeDeviceZoneScope("W_pub");
            const uint32_t src_base = get_read_ptr(cb_partial_row);
            const uint32_t slot_base = board_addr + r * round_bytes + my_slot * slot_bytes;
            for (uint32_t rt = 0; rt < h; ++rt) {
                for (uint32_t hh = 0; hh < 2; ++hh) {
                    const uint32_t src = src_base + rt * fp32_tile_bytes + hh * face_bytes_fp32;
                    const uint32_t dst = slot_base + (((rt_base + rt) * 2 + hh) * num_w_chunks + wc) * half_row_bytes;
                    for (uint32_t q = 0; q < 2; ++q) {
                        noc_async_write(src, get_noc_addr(ports[q] & 0xFFFF, ports[q] >> 16, dst), half_row_bytes);
                    }
                }
            }
            noc_async_write_barrier();  // the data lands before the counter says so
        }
        const uint32_t local_sem = get_semaphore((r & 1) ? sem_port_local_odd_id : sem_port_local_id);
        for (uint32_t q = 0; q < 2; ++q) {
            noc_semaphore_inc(get_noc_addr(ports[q] & 0xFFFF, ports[q] >> 16, local_sem), 1);
        }
        if (h > 0) {
            cb_pop_front(cb_partial_row, partial_pages(h));
        }
    };

    // ---- pre-combine: my partials of round r -> row `my_chunk` of half r % 2 of the leader's cb_pregather (rows of
    // tile t = s / 32: faces 0/1 for s % 32 < 16, faces 2/3 otherwise), then one increment of its parity counter
    const uint32_t pre_base = get_write_ptr(cb_pregather);  // FIFO base, identical on every worker
    auto publish_to_leader = [&](uint32_t r) {
        const uint32_t h = rows_of(r);
        if (h == 0) {
            return;  // every chunk of a row-group has the same rows: the leader skips this round too
        }
        cb_wait_front(cb_partial_row, block_h);
        const uint32_t src_base = get_read_ptr(cb_partial_row);
        const uint32_t lx = leader_xy & 0xFFFF, ly = leader_xy >> 16;
        const uint32_t s = my_chunk;
        const uint32_t half_base = pre_base + (r & 1) * n_pre_tiles * block_h * fp32_tile_bytes;
        for (uint32_t rt = 0; rt < h; ++rt) {
            for (uint32_t hh = 0; hh < 2; ++hh) {
                const uint32_t src = src_base + rt * fp32_tile_bytes + hh * face_bytes_fp32;
                const uint32_t face = ((s % 32) < 16 ? 0 : 2) + hh;
                const uint32_t dst = half_base + ((s / 32) * h + rt) * fp32_tile_bytes + face * face_bytes_fp32 +
                                     (s % 16) * half_row_bytes;
                noc_async_write(src, get_noc_addr(lx, ly, dst), half_row_bytes);
            }
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(lx, ly, get_semaphore((r & 1) ? sem_pre_ready_odd_id : sem_pre_ready_id)), 1);
        cb_pop_front(cb_partial_row, block_h);
    };
    // first publish of round r: to the ports, or (pre-combine) to the leader; second (pre-combine leader only): the
    // combined vector to the ports
    auto publish_local = [&](uint32_t r) {
        if constexpr (pre_combine) {
            publish_to_leader(r);
        } else {
            publish(r);
        }
    };
    auto publish_combined = [&](uint32_t r) {
        if constexpr (pre_combine) {
            if (is_leader) {
                publish(r);
            }
        }
    };

    // ---- fanout_inv_rms(r) (leader): into half (r % inv_rounds) of cb_inv_row's bytes on every member
    const uint32_t inv_base = get_write_ptr(cb_inv_row);  // FIFO base: this thread never moves cb_inv_row's pointer
    const uint32_t sig_addr = get_semaphore(sem_inv_sig_id);
    volatile tt_l1_ptr uint32_t* sig = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sig_addr);
    auto fanout = [&](uint32_t r) {
        const uint32_t h = rows_of(r);
        if (!is_leader || h == 0) {
            return;
        }
        {
            MaybeDeviceZoneScope("W_fwait");
            cb_wait_front(cb_inv_row_out, h);
        }
        MaybeDeviceZoneScope("W_fan");
        const uint32_t src_base = get_read_ptr(cb_inv_row_out);
        const uint32_t dst_base = inv_base + (r % inv_rounds) * block_h * fp32_tile_bytes;  // same on every worker
        const uint32_t ready = get_semaphore(sem_inv_ready_id);
        // the previous round's flush retired every read of the cell, so it can be rewritten
        *sig = r + 1;
        for (uint32_t k = 0; k < n_rects; ++k) {
            const uint32_t b = rects_idx + k * 6;
            const uint32_t sx = get_arg_val<uint32_t>(b), sy = get_arg_val<uint32_t>(b + 1);
            const uint32_t ex = get_arg_val<uint32_t>(b + 2), ey = get_arg_val<uint32_t>(b + 3);
            const uint32_t cnt = get_arg_val<uint32_t>(b + 4);
            const bool self = get_arg_val<uint32_t>(b + 5) != 0;
            // one transaction per tile row: face0 row 0 .. face1 row 0 (the bytes in between are rows 1-15 of face 0,
            // which neither the transpose's column 0 nor the Col broadcast reads), half the NoC issues of 2 x 64 B
            if (cnt == 1) {  // ordered unicast: data, then the 4 B flag (same src / dst / NoC / VC => in order)
                for (uint32_t rt = 0; rt < h; ++rt) {
                    const uint32_t off = rt * fp32_tile_bytes;
                    noc_async_write(src_base + off, get_noc_addr(sx, sy, dst_base + off), inv_row_bytes);
                }
                noc_async_write(sig_addr, get_noc_addr(sx, sy, ready), 4);
                continue;
            }
            for (uint32_t rt = 0; rt < h; ++rt) {
                const uint32_t off = rt * fp32_tile_bytes;
                const uint64_t m = get_noc_multicast_addr(sx, sy, ex, ey, dst_base + off);
                if (self) {
                    noc_async_write_multicast_loopback_src(src_base + off, m, inv_row_bytes, cnt, /*linked=*/true);
                } else {
                    noc_async_write_multicast(src_base + off, m, inv_row_bytes, cnt, /*linked=*/true);
                }
            }
            const uint64_t ms = get_noc_multicast_addr(sx, sy, ex, ey, ready);
            if (self) {
                noc_semaphore_set_multicast_loopback_src(sig_addr, ms, cnt, false);
            } else {
                noc_semaphore_set_multicast(sig_addr, ms, cnt, false);
            }
        }
        noc_async_writes_flushed();  // the inv pages and the sig cell are free once the data has left L1
        cb_pop_front(cb_inv_row_out, h);
    };

    // ---- store_block(r): one batch (block row of w_act tiles) per barrier. A ragged chunk (w_act < block_w) does not
    // divide the CB's capacity, so a batch may wrap past the FIFO end.
    const auto out_acc = TensorAccessor(out_args, out_addr, x_page_bytes);
    const uint32_t fifo_limit = get_local_cb_interface(cb_out_tiles).fifo_limit;
    const uint32_t fifo_size = get_local_cb_interface(cb_out_tiles).fifo_size;
    auto store = [&](uint32_t r) {
        if constexpr (out_sharded) {
            return;  // compute packed round r into the output shard itself
        }
        const uint32_t h = rows_of(r);
        const uint32_t first = row0 + r * round_stride;
        if (h > 0) {
            MaybeDeviceZoneScope("W_owait");  // starved on compute's first normalized tile row of the round
            cb_wait_front(cb_out_tiles, w_act);
        }
        // per-round zone (not per tile row: the marker budget is ~125 zones per RISC); includes later rows' waits
        MaybeDeviceZoneScope("W_store");
        for (uint32_t rt = 0; rt < h; ++rt) {
            cb_wait_front(cb_out_tiles, w_act);
            const uint32_t rbase = get_read_ptr(cb_out_tiles);
            const uint32_t page0 = (first + rt) * tensor_w_tiles + col0;
            const uint32_t rot = dram_stagger ? (first + rt) % w_act : 0;
            for (uint32_t i = 0, c = rot; i < w_act; ++i, c = (c + 1 == w_act) ? 0 : c + 1) {
                uint32_t rptr = rbase + c * x_page_bytes;
                if (rptr >= fifo_limit) {
                    rptr -= fifo_size;
                }
                noc_async_write_page(page0 + c, out_acc, rptr);
            }
            if constexpr (store_flush) {
                noc_async_writes_flushed();
            } else {
                noc_async_write_barrier();
            }
            // cb_pop_front only wraps when the read pointer lands exactly on the FIFO end, so a batch that straddles
            // it is popped in two parts (up to the end, then the rest from the base)
            const uint32_t to_end = (fifo_limit - get_read_ptr(cb_out_tiles)) / x_page_bytes;
            if (to_end < w_act) {
                cb_pop_front(cb_out_tiles, to_end);
                cb_pop_front(cb_out_tiles, w_act - to_end);
            } else {
                cb_pop_front(cb_out_tiles, w_act);
            }
        }
    };

    // Round order mirrors compute: with gamma (inv_rounds == 2 whenever num_rounds > 1) round r+1's partials are
    // produced before round r's combine/normalize, so they are published first and their exchange overlaps round r's
    // normalize/store.
    if constexpr (inv_rounds == 2) {
        publish_local(0);
        for (uint32_t r = 0; r < num_rounds; ++r) {
            if (r + 1 < num_rounds) {
                publish_local(r + 1);
            }
            publish_combined(r);
            fanout(r);
            store(r);
        }
    } else {
        for (uint32_t r = 0; r < num_rounds; ++r) {
            publish_local(r);
            publish_combined(r);
            fanout(r);
            store(r);
        }
    }
    noc_async_write_barrier();
}
