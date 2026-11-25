// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// q_stem port reader (row 0, RISCV_1 / NoC0): the pull half of `send_segment_block`.
// For every remote block whose destination's first hop is this port's direction (send order), once every stage-1
// writer has reported the block (block-ready count), pull each of this port's head-rows (one segment = hw tiles,
// columns hl*hw .. hl*hw+hw-1 of the block's rows) from the stage-1 cores' cb_q_slice (same address everywhere:
// the CB is declared on stage-1 rect + ports) into cb_port_stage, contiguous, for the sender.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/ttnn/operations/q_stem/perf_experiments/port_feed/q_stem_v/kernels/q_stem_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_port_stage = get_compile_time_arg_val(0);
    constexpr uint32_t cb_q_slice = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t C1 = get_compile_time_arg_val(3);
    constexpr uint32_t sem_block_ready_id = get_compile_time_arg_val(4);
    constexpr uint32_t B = get_compile_time_arg_val(5);
    constexpr uint32_t qrt = get_compile_time_arg_val(6);
    constexpr uint32_t m = get_compile_time_arg_val(7);
    constexpr uint32_t G = get_compile_time_arg_val(8);
    constexpr uint32_t hw = get_compile_time_arg_val(9);
    constexpr uint32_t batch = get_compile_time_arg_val(10);  // segments per read barrier
    constexpr uint32_t cap = 2 * batch;                       // cb_port_stage depth in segments (host: 2 batches)

    size_t a = 0;
    const uint32_t p = get_arg_val<uint32_t>(a++);
    const uint32_t my_dir = get_arg_val<uint32_t>(a++);
    uint32_t dirmap[G];
    for (uint32_t i = 0; i < G; ++i) {
        dirmap[i] = get_arg_val<uint32_t>(a++);
    }
    uint32_t rorder[G];
    for (uint32_t i = 0; i + 1 < G; ++i) {
        rorder[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t cols_base = get_arg_val<uint32_t>(a++);
    const uint32_t cols_extra = get_arg_val<uint32_t>(a++);
    const uint32_t n_heads = get_arg_val<uint32_t>(a++);
    const uint32_t heads_idx = a;
    a += n_heads;
    const uint32_t cores_idx = a;
    if (n_heads == 0) {
        return;
    }

    const uint32_t q_base = get_read_ptr(cb_q_slice);
    // block-ready ring (host BLOCK_READY_RING): the owners' idx-th report of my direction lands on semaphore
    // sem_block_ready_id + idx % ring; block idx is ready once that one holds every owner's report for idx
    constexpr uint32_t ready_ring = get_named_compile_time_arg_val("ready_ring");
    const uint32_t big = cols_extra * (cols_base + 1);  // columns owned by the cores with one extra column

#ifdef ABLATE_PF_DIAG
    uint32_t t_ready = 0, t_res = 0, t_bar = 0;
#define PF_T(acc, stmt)                           \
    {                                             \
        const uint32_t _t0 = get_timestamp_32b(); \
        stmt;                                     \
        acc += get_timestamp_32b() - _t0;         \
    }
#else
#define PF_T(acc, stmt) stmt
#endif
    {
        MaybeDeviceZoneScope("PORT_PULL");  // block-ready waits + q pulls from the stage-1 cores
        uint32_t mine = 0;
        uint32_t slot = 0;  // cb_port_stage write position in segments (mod cap): a batch never wraps past the CB end
        q_stem::for_each_block(B, qrt, m, G, p, rorder, [&](uint32_t j, uint32_t, uint32_t d, uint32_t) {
            if (d == p || dirmap[d] != my_dir) {
                return;
            }
            const uint32_t idx = mine++;
            PF_T(
                t_ready,
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        get_semaphore(sem_block_ready_id + idx % ready_ring)),
                    (idx / ready_ring + 1) * C1));
#ifdef ABLATE_PF_DIAG
            DeviceTimestampedData("PR_BLK", (uint64_t)t_res | ((uint64_t)t_bar << 21) | ((uint64_t)t_ready << 42));
#endif
            uint32_t pending = 0;  // segments read into the reserved batch, not yet pushed
            uint32_t limit = 0;    // segments in the reserved batch (<= batch, up to the CB end)
            uint32_t l1 = 0;
            auto flush = [&]() {
                PF_T(t_bar, noc_async_read_barrier());
                cb_push_back(cb_port_stage, pending * hw);
                slot = (slot + pending) % cap;
                pending = 0;
            };
            for (uint32_t i = 0; i < m; ++i) {
                for (uint32_t e = 0; e < n_heads; ++e) {
                    const uint32_t hl = get_arg_val<uint32_t>(heads_idx + e);
                    if (pending == 0) {
                        limit = cap - slot < batch ? cap - slot : batch;
                        PF_T(t_res, cb_reserve_back(cb_port_stage, limit * hw));
                        l1 = get_write_ptr(cb_port_stage);
                    }
                    uint32_t c = hl * hw;
                    const uint32_t c_end = c + hw;
                    while (c < c_end) {
                        uint32_t ci, c0, cols;
                        if (c < big) {
                            ci = c / (cols_base + 1);
                            cols = cols_base + 1;
                            c0 = ci * cols;
                        } else {
                            ci = cols_extra + (c - big) / cols_base;
                            cols = cols_base;
                            c0 = big + (ci - cols_extra) * cols_base;
                        }
                        const uint32_t run = (c0 + cols < c_end ? c0 + cols : c_end) - c;
                        const uint32_t pc = get_arg_val<uint32_t>(cores_idx + ci);
                        const uint32_t src = q_base + ((j * m + i) * cols + (c - c0)) * tile_bytes;
                        noc_async_read(
                            get_noc_addr(q_stem::unpack_x(pc), q_stem::unpack_y(pc), src), l1, run * tile_bytes);
                        l1 += run * tile_bytes;
                        c += run;
                    }
                    if (++pending == limit) {
                        flush();
                    }
                }
            }
            if (pending > 0) {
                flush();
            }
        });
    }
#ifdef ABLATE_PF_DIAG
    DeviceTimestampedData("PR_T_READY", t_ready);
    DeviceTimestampedData("PR_T_RESERVE", t_res);
    DeviceTimestampedData("PR_T_BARRIER", t_bar);
#endif
}
