// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streamed-expert weight forwarder (NCRISC, NOC1): for every chunk (se_schedule.hpp) sends each of the R receivers its
// block into landing slot c % SLOTS once that receiver has granted the slot (credit), then bumps the receivers' data
// counters after the write barrier (an atomic is not ordered behind the write's data).
//
// CT: 0 CB, 1 R, 2 TILE_BYTES, 3 SLOT_TILES (reader CB slot), 4 LAND_SLOT_TILES, 5 SLOTS, 6 CREDIT_SEM0, 7 DATA_SEM,
//     8 KBLK, 9 NK_GU, 10 NK_D, 11 PASSES, 12 NUM_EXPERTS, 13 BATCH, 14 PIPELINED (chunk order), 15 DUP (consecutive
//     receivers sharing one block: the M-split variant sends each block to one core per M-group), 16 GU_BLK_TILES,
//     17 PAIR_MCAST (DUP = 2 and each pair is two adjacent cores: one 2-core multicast instead of two writes)
// RT: 0 landing base, 1..R receiver xy, then R x PASSES down block sizes (tiles), receiver-major
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "se_schedule.hpp"
#ifdef SE_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define SE_MARK(name)            \
    {                            \
        DeviceZoneScopedN(name); \
    }
#else
#define SE_MARK(name)
#endif

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t slot_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t land_slot_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t slots = get_compile_time_arg_val(5);
    constexpr uint32_t credit_sem0 = get_compile_time_arg_val(6);
    constexpr uint32_t data_sem = get_compile_time_arg_val(7);
    constexpr uint32_t kblk = get_compile_time_arg_val(8);
    constexpr uint32_t nk_gu = get_compile_time_arg_val(9);
    constexpr uint32_t nk_d = get_compile_time_arg_val(10);
    constexpr uint32_t passes = get_compile_time_arg_val(11);
    constexpr uint32_t num_experts = get_compile_time_arg_val(12);
    constexpr uint32_t batch = get_compile_time_arg_val(13);
    constexpr bool pipelined = get_compile_time_arg_val(14) != 0;
    constexpr uint32_t dup = get_compile_time_arg_val(15);
    constexpr uint32_t gu_blk_tiles = get_compile_time_arg_val(16);
    constexpr bool pair_mcast = get_compile_time_arg_val(17) != 0;
    static_assert(!pair_mcast || dup == 2);
    constexpr uint32_t land_slot_bytes = land_slot_tiles * tile_bytes;
    constexpr uint32_t gu_blk_bytes = gu_blk_tiles * tile_bytes;

    const uint32_t landing_base = get_arg_val<uint32_t>(0);
    uint64_t recv_noc[R];
    uint64_t recv_data_sem[R];
    uint32_t d_blk_bytes[R][passes];
    volatile tt_l1_ptr uint32_t* credit[R];
    for (uint32_t j = 0; j < R; ++j) {
        const uint32_t xy = get_arg_val<uint32_t>(1 + j);
        recv_noc[j] = get_noc_addr(xy >> 16, xy & 0xFFFF, 0);
        recv_data_sem[j] = get_noc_addr(xy >> 16, xy & 0xFFFF, get_semaphore(data_sem));
        credit[j] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_sem0 + j));
        for (uint32_t p = 0; p < passes; ++p) {
            d_blk_bytes[j][p] = get_arg_val<uint32_t>(1 + R + j * passes + p) * tile_bytes;
        }
    }

    uint64_t pair_rect[R / 2 > 0 ? R / 2 : 1];
    if constexpr (pair_mcast) {
        for (uint32_t j = 0; j + 1 < R; j += 2) {
            const uint32_t a = get_arg_val<uint32_t>(1 + j), b = get_arg_val<uint32_t>(2 + j);
            const uint32_t ax = a >> 16, ay = a & 0xFFFF, bx = b >> 16, by = b & 0xFFFF;
            const uint32_t lx = ax < bx ? ax : bx, hx = ax < bx ? bx : ax, ly = ay < by ? ay : by,
                           hy = ay < by ? by : ay;
            // NOC1 multicasts run from the high corner to the low one.
            pair_rect[j / 2] =
                noc_index == 1 ? get_noc_multicast_addr(hx, hy, lx, ly, 0) : get_noc_multicast_addr(lx, ly, hx, hy, 0);
        }
    }
    uint32_t c = 0, in_batch = 0, l1 = 0;
    for_each_chunk<nk_gu, nk_d, passes, num_experts, pipelined>([&](bool is_down, uint32_t p) {
        if (in_batch == 0) {
            cb_wait_front(cb, slot_tiles * batch);
            l1 = get_read_ptr(cb);
        }
        const uint32_t dst = landing_base + (c % slots) * land_slot_bytes;
        uint32_t src = l1 + in_batch * slot_tiles * tile_bytes;
        if constexpr (pair_mcast) {
            for (uint32_t j = 0; j < R; j += 2) {
                const uint32_t bytes = is_down ? d_blk_bytes[j][p] : gu_blk_bytes;
                noc_semaphore_wait_min(credit[j], c + 1);
                noc_semaphore_wait_min(credit[j + 1], c + 1);
                noc_async_write_multicast(src, pair_rect[j / 2] | dst, bytes, 2);
                src += bytes;
            }
        } else {
            for (uint32_t j = 0; j < R; ++j) {
                const uint32_t bytes = is_down ? d_blk_bytes[j][p] : gu_blk_bytes;
                noc_semaphore_wait_min(credit[j], c + 1);
                noc_async_write(src, recv_noc[j] | dst, bytes);
                if ((j + 1) % dup == 0) {
                    src += bytes;
                }
            }
        }
        noc_async_write_barrier();
        for (uint32_t j = 0; j < R; ++j) {
            noc_semaphore_inc(recv_data_sem[j], 1);
        }
        ++c;
        if (c % 8 == 0) {
            SE_MARK("W_FW");
        }
        if (++in_batch == batch) {
            cb_pop_front(cb, slot_tiles * batch);
            in_batch = 0;
        }
    });
    if (in_batch) {
        cb_pop_front(cb, slot_tiles * batch);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
