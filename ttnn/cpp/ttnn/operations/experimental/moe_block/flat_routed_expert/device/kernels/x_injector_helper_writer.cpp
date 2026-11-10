// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// x_injector_helper_writer: ships a helper relay's tilized x super-blocks into both primaries' landing rings
// Runs on: x helper relays (p.relays[NR..], one kernel per helper), RISCV_1 (NCRISC), NoC 0
// Responsibilities:
//  - reads tilized super-blocks (L1, CB SB_CB) from x_injector_tilize_compute; no DRAM reads
//  - unicasts each into slot b % LAND_SLOTS of its landing ring on both primary relays (x_injector_writer)
//  - waits for free slots on its CREDIT semaphores (5: primary 0, 6: primary 1, bumped by x_injector_writer),
//    then after the write barrier bumps DATA_SEM (4 or 7) on both primaries

// Flat expert x injector helper writer (an x helper relay's NCRISC). The helper core reads and tilizes every other
// super-block of its rectangle (x_injector_reader.cpp / x_injector_tilize_compute.cpp, stride 2 offset 1) so the
// rectangle's one multicaster (its primary relay, x_injector_writer.cpp) does not have to read all of x
// from DRAM itself: each tilized super-block goes into slot b % LAND_SLOTS of the primary's landing ring once the
// primary freed it (CREDIT_SEM here counts freed slots), then the primary's DATA_SEM is bumped after the write is
// acknowledged. CT: 0 SB_CB, 1 MT, 2 TILE_BYTES, 3 LAND_SLOTS, 4 DATA_SEM (on the primary), 5 CREDIT_SEM (here), 6
// NUM_E, 7 NSB
// RT: 0 primary xy, 1 landing ring address, 2 super-blocks, 3 round, 4 place, 5.. expert_schedule.hpp args
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "flat_options.hpp"
#include "expert_schedule.hpp"

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table, or the runtime args: moe_block_args.hpp)
    constexpr uint32_t sb_cb = get_compile_time_arg_val(0);
    constexpr uint32_t mt = get_compile_time_arg_val(1);
    constexpr uint32_t tb = get_compile_time_arg_val(2);
    constexpr uint32_t land_slots = get_compile_time_arg_val(3);
    constexpr uint32_t data_sem = get_compile_time_arg_val(4);
    constexpr uint32_t credit_sem = get_compile_time_arg_val(5);
    constexpr uint32_t sbt = flat_options::super_block_tiles;
    constexpr uint32_t sb_tiles = mt * sbt, sb_bytes = sb_tiles * tb;
    const uint32_t pxy = moe_block_args::arg(0), land = moe_block_args::arg(1);
    // The schedule: the odd super-blocks of the active experts' stream (CT 6 NUM_E, 7 NSB; RT 3.. expert_schedule.hpp
    // args, CB 7's upper half this RISC's scratch)
    ExpertSchedule schedule;
    // RT 3 / 4: the primary's round (1 + helpers) and this helper's place in it (its super-blocks b % round == place)
    load_schedule<get_compile_time_arg_val(6)>(schedule, 5, schedule_scratch(ScratchOwner::Ncrisc));
    const uint32_t tot_all = schedule.total_sub_blocks * get_compile_time_arg_val(7),
                   x2s = moe_block_args::arg(5 + kScheduleArgs), x2k = moe_block_args::arg(6 + kScheduleArgs);
    const uint32_t tot_sb = tot_all > x2k ? (tot_all - x2k + x2s - 1) / x2s : 0;
    const uint32_t round_ = moe_block_args::arg(3), place = moe_block_args::arg(4);
    const uint32_t num_sb = tot_sb <= place ? 0 : (tot_sb - place + round_ - 1) / round_;
    const uint64_t prim_land = get_noc_addr(pxy >> 16, pxy & 0xFFFF, land);
    const uint64_t prim_data = get_noc_addr(pxy >> 16, pxy & 0xFFFF, get_semaphore(data_sem));
    volatile tt_l1_ptr uint32_t* credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_sem));
    // x read once by the helpers: every super-block also goes into the second primary's landing ring (same ring, same
    // arrival semaphore there; RT after the X2 args: its xy), its freed slots counted on semaphore 6 here
    const uint32_t p2xy = moe_block_args::arg(7 + kScheduleArgs);
    const uint64_t prim2_land = get_noc_addr(p2xy >> 16, p2xy & 0xFFFF, land);
    const uint64_t prim2_data = get_noc_addr(p2xy >> 16, p2xy & 0xFFFF, get_semaphore(data_sem));
    volatile tt_l1_ptr uint32_t* credit2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(6));
    constexpr uint32_t nsb = get_compile_time_arg_val(7);
    for (uint32_t b = 0; b < num_sb; ++b) {
        cb_wait_front(sb_cb, sb_tiles);
        if (b >= land_slots) {
            noc_semaphore_wait_min(credit, b + 1 - land_slots);  // the slot's previous super-block has been sent
            noc_semaphore_wait_min(credit2, b + 1 - land_slots);
        }
        // (row tile major: a short last sub-block's rows are the super-block's first bytes - only those go)
        const uint32_t g = x2k + (place + b * round_) * x2s;
        const uint32_t bytes = sub_block_row_tiles(schedule, g / nsb) * sbt * tb;
        noc_async_write(get_read_ptr(sb_cb), prim_land + (b % land_slots) * sb_bytes, bytes);
        noc_async_write(get_read_ptr(sb_cb), prim2_land + (b % land_slots) * sb_bytes, bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(prim_data, 1);
        noc_semaphore_inc(prim2_data, 1);
        cb_pop_front(sb_cb, sb_tiles);
    }
    noc_async_atomic_barrier();
    // leave no NoC transaction in flight (reads, writes, atomics, posted writes): the next program starts clean
    noc_async_full_barrier();
}
