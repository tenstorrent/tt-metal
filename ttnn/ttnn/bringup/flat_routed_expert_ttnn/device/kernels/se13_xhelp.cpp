// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end flat expert: x relay helper forwarder (NCRISC). The helper core reads and tilizes every other
// super-block of its rectangle (se11_xrd.cpp / se11_tz.cpp, stride 2 offset 1) so the rectangle's one multicaster (its
// primary relay, se11_xmc.cpp with XMC_HELPER) does not have to read all of x from DRAM itself: each tilized
// super-block goes into slot b % LAND_SLOTS of the primary's landing ring once the primary freed it (CREDIT_SEM here
// counts freed slots), then the primary's DATA_SEM is bumped after the write is acknowledged.
// CT: 0 SB_CB, 1 MT, 2 TILE_BYTES, 3 LAND_SLOTS, 4 DATA_SEM (on the primary), 5 CREDIT_SEM (here), 6 NUM_E, 7 NSB
//     (SE_DYN)
// RT: 0 primary xy, 1 landing ring address, 2 super-blocks, 3 round, 4 place, 5.. se_dyn.hpp args (SE_DYN)
#include <stdint.h>
#ifndef SE_SBT
#define SE_SBT 32  // K tiles per super-block (the row-major chunk width / 32 columns)
#endif
#include "api/dataflow/dataflow_api.h"
#ifdef SE_DYN
#include "se_dyn.hpp"
#endif

void kernel_main() {
    constexpr uint32_t sb_cb = get_compile_time_arg_val(0);
    constexpr uint32_t mt = get_compile_time_arg_val(1);
    constexpr uint32_t tb = get_compile_time_arg_val(2);
    constexpr uint32_t land_slots = get_compile_time_arg_val(3);
    constexpr uint32_t data_sem = get_compile_time_arg_val(4);
    constexpr uint32_t credit_sem = get_compile_time_arg_val(5);
    constexpr uint32_t sb_tiles = mt * SE_SBT, sb_bytes = sb_tiles * tb;
    const uint32_t pxy = get_arg_val<uint32_t>(0), land = get_arg_val<uint32_t>(1);
#ifdef SE_DYN
    // Dynamic counts: the odd super-blocks of the active experts' stream (CT 6 NUM_E, 7 NSB; RT 3.. se_dyn.hpp args,
    // CB 7's upper half this RISC's scratch)
    SeDyn dyn;
    // RT 3 / 4: the primary's round (1 + helpers) and this helper's place in it (its super-blocks b % round == place)
    se_dyn_load<get_compile_time_arg_val(6)>(dyn, 5, get_write_ptr(tt::CBIndex::c_7) + 2 * SE_DYN_HALF, mt * 32);
    const uint32_t tot_sb = dyn.num_v * get_compile_time_arg_val(7);
    const uint32_t round_ = get_arg_val<uint32_t>(3), place = get_arg_val<uint32_t>(4);
#ifdef SE_SMALL_T
    const uint32_t num_sb = dyn.small || tot_sb <= place ? 0 : (tot_sb - place + round_ - 1) / round_;  // small: idle
#else
    const uint32_t num_sb = tot_sb <= place ? 0 : (tot_sb - place + round_ - 1) / round_;
#endif
#else
    const uint32_t num_sb = get_arg_val<uint32_t>(2);
#endif
    const uint64_t prim_land = get_noc_addr(pxy >> 16, pxy & 0xFFFF, land);
    const uint64_t prim_data = get_noc_addr(pxy >> 16, pxy & 0xFFFF, get_semaphore(data_sem));
    volatile tt_l1_ptr uint32_t* credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_sem));
    for (uint32_t b = 0; b < num_sb; ++b) {
        cb_wait_front(sb_cb, sb_tiles);
        if (b >= land_slots) {
            noc_semaphore_wait_min(credit, b + 1 - land_slots);  // the slot's previous super-block has been sent
        }
        noc_async_write(get_read_ptr(sb_cb), prim_land + (b % land_slots) * sb_bytes, sb_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(prim_data, 1);
        cb_pop_front(sb_cb, sb_tiles);
    }
    noc_async_atomic_barrier();
    // leave no NoC transaction in flight (reads, writes, atomics, posted writes): the next program starts clean
    noc_async_full_barrier();
}
