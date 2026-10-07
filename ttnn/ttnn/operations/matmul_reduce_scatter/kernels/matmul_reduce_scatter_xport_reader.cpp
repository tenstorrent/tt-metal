// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — transport-core NCRISC (ports and finals): gather_partial_block + load_arrival_block.
//
// Per entry of this core's list -- one wave (compute unit) of a block (port: the blocks it sends, farthest first;
// final: the chip's own block), waves in order -- once every compute core has signalled the unit
// (sem_block_ready >= compute_cores * (k + 1)):
//   per segment of this core's set inside the wave's window (`count` of them from seg0, see
//   matmul_reduce_scatter_segments.hpp; a segment = seg_tiles consecutive tiles of one block row):
//     gather: the segment's pieces from the owning compute cores' hand-off slot (1-2 contiguous NoC reads, the
//             hand-off is TileRowMajor) -> cb_partial_target (cb_xport_partial, or cb_xport_sum on a line-end port);
//     arrival A / B (relay port: A = what upstream sent into relay_scratch slot j; final: A = forward slot p,
//             B = backward slot G), each gated by its arrival counter (one increment per inc_every segments of an
//             entry and one at the entry's last segment, so ceil(count / inc_every) per upstream entry). A ring port's
//             first entry has no upstream (has_upstream = 0: the chip's own partial starts that block's chain): only
//             the gather is pushed; the add kernel copies it through.
//   One read barrier + one CB push per xport_group segments (and at the CB wrap / entry end); after the entry's
//   last barrier, one multicast ack (sem_block_ack += 1) over the compute rectangle releases the hand-off slot.
// Before exit: re-arm sem_block_ready.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "matmul_reduce_scatter_segments.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

// Stage zones (permanent; opt-in via KERNEL_PERF_ZONES), per entry: xr_wait_ready = waiting for the compute grid to
// finish the block (pipeline fill / compute-bound), xr_entry = the entry's gather + arrival walk (includes the
// arrival-counter waits on upstream and back-pressure from the add / sender, so occupancy). MMRS_ABLATE_XREADS (perf
// ablation only, wrong results) drops the gather and arrival NoC reads and keeps every wait and CB credit.

void kernel_main() {
    constexpr uint32_t cb_partial_target = get_compile_time_arg_val(0);
    constexpr uint32_t cb_arrival_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_arrival_b = get_compile_time_arg_val(2);
    constexpr uint32_t has_a = get_compile_time_arg_val(3);
    constexpr uint32_t has_b = get_compile_time_arg_val(4);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t group = get_compile_time_arg_val(6);  // segments per barrier / push
    constexpr uint32_t inc_every = get_compile_time_arg_val(7);
    constexpr uint32_t core_m_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t core_n_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t m_on_y = get_compile_time_arg_val(10);    // m-lines are grid rows (orientation A)
    constexpr uint32_t cap_segs = get_compile_time_arg_val(11);  // CB capacity in segments
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(12);
    constexpr auto scr_args = TensorAccessorArgs<13>();
    constexpr uint32_t seg_bytes = seg_tiles * tile_bytes;
    constexpr uint32_t seg_pages = seg_tiles;

    size_t arg = 0;
    const uint32_t scr_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t handoff_base = get_arg_val<uint32_t>(arg++);
    const uint32_t slot_bytes = get_arg_val<uint32_t>(arg++);  // one hand-off slot (core_m_tiles * core_n_tiles tiles)
    const uint32_t segs_per_block = get_arg_val<uint32_t>(arg++);
    const uint32_t segs_per_row = get_arg_val<uint32_t>(arg++);
    const uint32_t blk_n_tiles = get_arg_val<uint32_t>(arg++);
    const uint32_t seg_stride = get_arg_val<uint32_t>(arg++);
    const uint32_t ready_sem_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t num_compute_cores = get_arg_val<uint32_t>(arg++);
    const uint32_t arr_a_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t arr_b_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t ack_sem_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t ack_x0 = get_arg_val<uint32_t>(arg++);
    const uint32_t ack_y0 = get_arg_val<uint32_t>(arg++);
    const uint32_t ack_x1 = get_arg_val<uint32_t>(arg++);
    const uint32_t ack_y1 = get_arg_val<uint32_t>(arg++);
    const uint32_t m_lines = get_arg_val<uint32_t>(arg++);
    const uint32_t n_lines = get_arg_val<uint32_t>(arg++);
    const uint32_t num_blocks = get_arg_val<uint32_t>(arg++);  // entries
    // num_blocks x [hand-off slot, scratch slot A, scratch slot B, has_upstream, first seg, seg count,
    //               wave tile origin r0, c0 (the hand-off slot holds the wave), wave segment columns s0, s1]
    constexpr uint32_t entry_stride = 10;
    const uint32_t entries_idx = arg;
    arg += entry_stride * num_blocks;
    const uint32_t mcoord_idx = arg;  // NoC coordinate of each m-line (y if m_on_y else x)
    arg += m_lines;
    const uint32_t ncoord_idx = arg;  // NoC coordinate of each n-line

    const auto scr = TensorAccessor(scr_args, scr_addr, seg_bytes);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_sem_addr);
    volatile tt_l1_ptr uint32_t* arr_a = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arr_a_addr);
    volatile tt_l1_ptr uint32_t* arr_b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arr_b_addr);
    const uint64_t ack_mcast = get_noc_multicast_addr(ack_x0, ack_y0, ack_x1, ack_y1, ack_sem_addr);

    uint32_t wpos = 0, apos = 0, batch = 0;  // write positions (segments) in the target / arrival CB rings
    uint32_t w_own = 0, w_a = 0, w_b = 0;
    uint32_t inc_base = 0;  // arrival increments of the upstream entries already consumed
    for (uint32_t k = 0; k < num_blocks; ++k) {
        const uint32_t e = entries_idx + entry_stride * k;
        const uint32_t hslot_off = get_arg_val<uint32_t>(e) * slot_bytes;
        const uint32_t base_a = get_arg_val<uint32_t>(e + 1) * segs_per_block;
        const uint32_t base_b = get_arg_val<uint32_t>(e + 2) * segs_per_block;
        const bool up = get_arg_val<uint32_t>(e + 3) != 0;
        const uint32_t seg0 = get_arg_val<uint32_t>(e + 4);
        const uint32_t count = get_arg_val<uint32_t>(e + 5);
        const uint32_t wave_r0 = get_arg_val<uint32_t>(e + 6);
        const uint32_t wave_c0 = get_arg_val<uint32_t>(e + 7);
        const uint32_t wave_s0 = get_arg_val<uint32_t>(e + 8);
        const uint32_t wave_s1 = get_arg_val<uint32_t>(e + 9);
        {
            MaybeDeviceZoneScope("xr_wait_ready");
            noc_semaphore_wait_min(ready, num_compute_cores * (k + 1));
        }
        MaybeDeviceZoneScope("xr_entry");
        uint32_t seg = seg0;
        for (uint32_t idx = 0; idx < count; ++idx) {
            const uint32_t row = seg / segs_per_row;
            const uint32_t c0 = (seg - row * segs_per_row) * seg_tiles;
            const uint32_t valid = blk_n_tiles - c0 < seg_tiles ? blk_n_tiles - c0 : seg_tiles;
            const uint32_t bytes = valid * tile_bytes;
            if (batch == 0) {
                cb_reserve_back(cb_partial_target, seg_pages);
                w_own = get_write_ptr(cb_partial_target);
                if constexpr (has_a) {
                    if (up) {
                        cb_reserve_back(cb_arrival_a, seg_pages);
                        w_a = get_write_ptr(cb_arrival_a);
                    }
                }
                if constexpr (has_b) {
                    if (up) {
                        cb_reserve_back(cb_arrival_b, seg_pages);
                        w_b = get_write_ptr(cb_arrival_b);
                    }
                }
            } else {
                cb_reserve_back(cb_partial_target, seg_pages * (batch + 1));
                if constexpr (has_a) {
                    if (up) {
                        cb_reserve_back(cb_arrival_a, seg_pages * (batch + 1));
                    }
                }
                if constexpr (has_b) {
                    if (up) {
                        cb_reserve_back(cb_arrival_b, seg_pages * (batch + 1));
                    }
                }
            }
            // gather: pieces of block row `row`, columns [c0, c0 + valid), from the owning compute cores (the
            // hand-off slot holds the wave: its tile (0, 0) is block tile (wave_r0, wave_c0))
            const uint32_t ml = (row - wave_r0) / core_m_tiles;
            const uint32_t lr = (row - wave_r0) - ml * core_m_tiles;
            const uint32_t mco = get_arg_val<uint32_t>(mcoord_idx + ml);
            uint32_t dst = w_own + batch * seg_bytes;
            for (uint32_t c = c0; c < c0 + valid;) {
                const uint32_t nl = (c - wave_c0) / core_n_tiles;
                const uint32_t lc = (c - wave_c0) - nl * core_n_tiles;
                const uint32_t run = (core_n_tiles - lc) < (c0 + valid - c) ? (core_n_tiles - lc) : (c0 + valid - c);
                const uint32_t nco = get_arg_val<uint32_t>(ncoord_idx + nl);
                const uint32_t x = m_on_y ? nco : mco;
                const uint32_t y = m_on_y ? mco : nco;
#ifndef MMRS_ABLATE_XREADS
                noc_async_read(
                    get_noc_addr(x, y, handoff_base + hslot_off + (lr * core_n_tiles + lc) * tile_bytes),
                    dst,
                    run * tile_bytes);
#endif
                dst += run * tile_bytes;
                c += run;
            }
            if (up) {
                if constexpr (has_a) {
                    {
                        MaybeDeviceZoneScope("xr_arr_wait_a");
                        noc_semaphore_wait_min(arr_a, inc_base + idx / inc_every + 1);
                    }
#if !defined(MMRS_ABLATE_XREADS) && !defined(MMRS_ABLATE_ARRREADS)
                    noc_async_read(scr.get_noc_addr(base_a + seg), w_a + batch * seg_bytes, bytes);
#endif
                }
                if constexpr (has_b) {
                    {
                        MaybeDeviceZoneScope("xr_arr_wait_b");
                        noc_semaphore_wait_min(arr_b, inc_base + idx / inc_every + 1);
                    }
#if !defined(MMRS_ABLATE_XREADS) && !defined(MMRS_ABLATE_ARRREADS)
                    noc_async_read(scr.get_noc_addr(base_b + seg), w_b + batch * seg_bytes, bytes);
#endif
                }
            }
            ++batch;
            const bool entry_end = idx + 1 == count;
            if (batch == group || wpos + batch == cap_segs || (up && apos + batch == cap_segs) || entry_end) {
                {
                    MaybeDeviceZoneScope("xr_barrier");
                    noc_async_read_barrier();
                }
                cb_push_back(cb_partial_target, seg_pages * batch);
                if constexpr (has_a) {
                    if (up) {
                        cb_push_back(cb_arrival_a, seg_pages * batch);
                    }
                }
                if constexpr (has_b) {
                    if (up) {
                        cb_push_back(cb_arrival_b, seg_pages * batch);
                    }
                }
                wpos += batch;
                if (wpos == cap_segs) {
                    wpos = 0;
                }
                if (up) {
                    apos += batch;
                    if (apos == cap_segs) {
                        apos = 0;
                    }
                }
                batch = 0;
            }
            if (!entry_end) {
                seg = mmrs::next_wave_seg(seg, seg_stride, segs_per_row, wave_s0, wave_s1);
            }
        }
        if (up) {
            inc_base += (count + inc_every - 1) / inc_every;
        }
        // every piece of this entry has landed: release the hand-off slot on the compute cores
        noc_semaphore_inc_multicast(ack_mcast, 1, num_compute_cores);
    }
    if (num_blocks > 0) {
        noc_semaphore_inc(get_noc_addr(ready_sem_addr), 0u - num_compute_cores * num_blocks);
    }
    noc_async_atomic_barrier();
}
