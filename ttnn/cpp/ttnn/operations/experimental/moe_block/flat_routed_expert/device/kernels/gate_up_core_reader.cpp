// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// gate_up_core_reader: a gate/up core's receiver loop: weight-slot grants, x / weight publishing, h out to down
// Runs on: gate/up cores (p.gu), RISCV_0 (BRISC), NoC 1
// Responsibilities:
//  - reads no DRAM: grants weight landing slots (GRANT sem) and hands the loads list to the NCRISC's weight reads
//    (gate_up_weight_read.hpp in gate_up_x_forwarder, its mailbox semaphore); publishes landed blocks (DATA) to in1_cb
//  - publishes x blocks to compute as XARR pieces arrive; posts the consumed / end counts (flat_options::x_consumed_sem
//    / x_end_sem) for the x forwarder
//  - writes its h slice ([MT x NP] tiles) to every down chain head's h_all (VC 0) once GO allows, then bumps GATH
// Note: NoC 1 because h writes, DRAM weight reads and x deliveries all on NoC 0 wedged the NoC (HW hang, ~1 in
//       1e4..2e5 launches; tests/ttnn/unit_tests/operations/noc_hang_probe).

// Gate/up core reader (BRISC, NOC1): a gate/up core's receiver loop. A non-blocking event loop that
//   * runs the in1 (weight) ring: grants each free landing slot (this core's grant counter) to the weight reads
//     (gate_up_weight_read.hpp: this core's NCRISC on NOC0) and publishes blocks as their DATA count moves; the ring
//     holds the weight slices of the schedule's loads, compute frees blocks after their last use,
//   * publishes x blocks to compute as their pieces arrive (XARR, incremented by the predecessor or the relay after
//     each piece's write is acknowledged) and posts the consumed count (x_consumed_sem) for the NCRISC forwarder
//     (gate_up_x_forwarder.cpp), which forwards x and reports freed slots upstream,
//   * sends its h slice of each virtual expert v ([MT x NP] tiles, h K-tiles CG * NP + p) into the h_all of every down
//     chain head with a GATH increment there, once go >= v (counted by the down
//     cores' coordinator).
//
// CT: 0 X_CB, 1 X_BLK_TILES, 2 IN1_CB, 3 SLOT_TILES, 4 NK_GU (weight blocks of a load, x blocks of a virtual
//     expert), 5 W_SLOTS, 6 H_LOCAL_CB, 7 MT, 8 H_TILE_BYTES, 9 NCC, 10 DATA_SEM, 11 GO_SEM, 12 HBUF, 13 XARR_SEM,
//     14 X_SLOTS, 15 NP, 16-17 GATH ids of h buffers 1, 2, 18 X_PIECES (XARR counts pieces), 19 NUM_E
// RT: 0 the grant counter's sem id here, 1 column group CG, 2 NH (chain heads of the group), 3 the GATH sem id, 4
//     h_all address, 5.. the expert_schedule.hpp args. Common args: the NH head xy
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "expert_schedule.hpp"
#include "flat_options.hpp"

// The h exchange's unicast VC (gate/up -> down heads, the down chains): its own, off the default write VC (1)
constexpr uint8_t h_vc = 0;

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table, or the runtime args: moe_block_args.hpp)
    constexpr uint32_t x_cb = get_compile_time_arg_val(0);
    constexpr uint32_t x_blk = get_compile_time_arg_val(1);
    constexpr uint32_t in1_cb = get_compile_time_arg_val(2);
    constexpr uint32_t slot = get_compile_time_arg_val(3);
    constexpr uint32_t nk_gu = get_compile_time_arg_val(4);
    constexpr uint32_t w_slots = get_compile_time_arg_val(5);
    constexpr uint32_t h_local_cb = get_compile_time_arg_val(6);
    constexpr uint32_t mt = get_compile_time_arg_val(7);
    constexpr uint32_t h_tile_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t ncc = get_compile_time_arg_val(9);
    constexpr uint32_t data_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(11);
    constexpr uint32_t hbuf = get_compile_time_arg_val(12);
    constexpr uint32_t xarr_sem_id = get_compile_time_arg_val(13);
    constexpr uint32_t x_slots = get_compile_time_arg_val(14);
    constexpr uint32_t np = get_compile_time_arg_val(15);
    constexpr uint32_t xp = get_compile_time_arg_val(18);
    constexpr uint32_t num_e = get_compile_time_arg_val(19);
    // with 16-column units (NCC cores x units = the K tiles x 2), else NP whole tile pairs a core
    constexpr bool units = flat_options::gate_up_units != 0;
    constexpr uint32_t units_per_core = flat_options::gate_up_units;
    constexpr uint32_t hpc = units ? flat_options::gate_up_units_h_tiles : np;  // h tiles a row
    // h: NCC cores x NP K-tiles (or x units / 2), MT rows
    constexpr uint32_t h_all_tiles = units ? ncc * units_per_core * mt / 2 : ncc * np * mt;
    static_assert(!units || (ncc * units_per_core) % 2 == 0);
    static_assert(!units || h_tile_bytes == 2048 || h_tile_bytes == 1088, "16-column units: bf16 or bfp8 h tiles");
    auto write_h = [](uint32_t src, uint64_t dst, uint32_t bytes) {
        noc_async_write(src, dst, bytes, noc_index, h_vc);
    };
    constexpr uint32_t half_bytes = h_all_tiles * h_tile_bytes;
    static_assert(hbuf >= 1 && hbuf <= 3);

    const uint32_t credit_sem_id = moe_block_args::arg(0);
    const uint32_t cg = moe_block_args::arg(1);
    const uint32_t nh = moe_block_args::arg(2);
    const uint32_t gath_sem_id = moe_block_args::arg(3);
    const uint32_t h_all_addr = moe_block_args::arg(4);
    constexpr uint32_t heads0 = 5;
    auto sem = [](uint32_t id) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)); };
    // the landing slots granted so far (RT 0: the semaphore counting them, zero at launch): the weight reads
    // (gate_up_weight_read.hpp) read into granted slots only
    volatile tt_l1_ptr uint32_t* granted_word = sem(credit_sem_id);
    auto grant = [&](uint32_t n) { *granted_word = *granted_word + n; };
    // The schedule: the active experts' weights and sub-blocks only; RT 5 + NH.. are the expert_schedule.hpp args, CB
    // 7 this RISC's scratch; the compute gets them in CB 6.
    ExpertSchedule schedule;
    load_schedule<num_e>(schedule, heads0, schedule_scratch(ScratchOwner::Brisc));
    publish_schedule(schedule, tt::CBIndex::c_6);
    const uint32_t total_w = schedule.loads * nk_gu;  // gate/up loads
    const uint32_t num_v = schedule.total_sub_blocks;
    volatile tt_l1_ptr uint32_t* data_sem = sem(data_sem_id);
    volatile tt_l1_ptr uint32_t* go_sem = sem(go_sem_id);
    volatile tt_l1_ptr uint32_t* xarr_sem = sem(xarr_sem_id);

    // the gate/up weights read by this core's NCRISC (gate_up_weight_read.hpp): the loads list handed to it in its half
    // of CB 7, then its mailbox word set
    {
        volatile tt_l1_ptr uint32_t* mb =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(schedule_scratch(ScratchOwner::Ncrisc));
        mb[0] = schedule.loads;
        for (uint32_t l = 0; l < schedule.loads; ++l) {
            mb[1 + l] = schedule.load_expert[l] | (uint32_t(schedule.load_region[l]) << 8);
        }
        // then the x forwarder's: the entries' sub-blocks and their last one's row tiles holding tokens
        volatile tt_l1_ptr uint32_t* xe = mb + 1 + schedule.loads;
        xe[0] = schedule.entries;
        for (uint32_t a = 0; a < schedule.entries; ++a) {
            xe[1 + a] = schedule.sub_blocks[a] | (last_sub_block_row_tiles(schedule, a) << 16);
        }
        *sem(flat_options::gate_up_mailbox_sem) = 1;
    }
    grant(w_slots);  // every landing slot starts free
    uint32_t granted = w_slots, pushed = 0;
    uint32_t x_pub = 0, x_cons = 0;
    uint32_t h_sent = 0, h_pending = 0;

    while (h_sent < num_v || pushed < total_w) {
        invalidate_l1_cache();
        while (granted < total_w && cb_pages_reservable_at_back(in1_cb, (granted - pushed + 1) * slot)) {
            grant(1);
            ++granted;
        }
        while (*data_sem > pushed) {
            cb_push_back(in1_cb, slot);
            ++pushed;
        }
        const uint32_t arrived_p = *xarr_sem;     // pieces
        const uint32_t arrived = arrived_p / xp;  // whole blocks
        while (arrived > x_pub) {
            cb_push_back(x_cb, x_blk);
            ++x_pub;
        }
        if (x_cons < x_pub && cb_pages_reservable_at_back(x_cb, (x_slots - (x_pub - x_cons) + 1) * x_blk)) {
            ++x_cons;  // compute has consumed one more block
        }
        *sem(flat_options::x_consumed_sem) = x_cons;
        if (!h_pending && h_sent < num_v && *go_sem + hbuf >= h_sent + 1 &&
            cb_pages_available_at_front(h_local_cb, mt * hpc)) {
            const uint32_t src = get_read_ptr(h_local_cb);
            const uint32_t dst = (h_sent % hbuf) * half_bytes;  // down h_all: [HBUF]
            for (uint32_t hd = 0; hd < nh; ++hd) {
                const uint32_t hxy = get_common_arg_val<uint32_t>(hd);
                const uint64_t bc_h = get_noc_addr(hxy >> 16, hxy & 0xFFFF, h_all_addr);
                if constexpr (units) {
                    // 16-column units cg NU..: h tiles from K-tile cg NU / 2, the first one's right half only when cg
                    // NU is odd (lead), the last one's left half only when (cg + 1) NU is odd (tail); the full ones are
                    // one run, the halves go face by face (bf16: faces 512 B; bfp8: 16 B exponents of each face, then
                    // 256 B data)
                    const uint32_t u0 = cg * units_per_core, k0 = u0 / 2, lead = u0 & 1,
                                   tail = (u0 + units_per_core) & 1;
                    const uint32_t j0 = lead, j1 = hpc - tail;
                    const uint64_t base = bc_h + dst + k0 * mt * h_tile_bytes;
                    if (j1 > j0) {
                        write_h(
                            src + j0 * mt * h_tile_bytes, base + j0 * mt * h_tile_bytes, (j1 - j0) * mt * h_tile_bytes);
                    }
                    auto half = [&](uint32_t j, uint32_t right) {  // tile j's left (faces 0 / 2) or right (1 / 3) half
                        for (uint32_t m = 0; m < mt; ++m) {
                            const uint32_t s = src + (j * mt + m) * h_tile_bytes;
                            const uint64_t d = base + (j * mt + m) * h_tile_bytes;
                            for (uint32_t f = right; f < 4; f += 2) {
                                if constexpr (h_tile_bytes == 2048) {
                                    write_h(s + f * 512, d + f * 512, 512);
                                } else {
                                    write_h(s + f * 16, d + f * 16, 16);
                                    write_h(s + 64 + f * 256, d + 64 + f * 256, 256);
                                }
                            }
                        }
                    };
                    if (lead) {
                        half(0, 1);
                    }
                    if (tail) {
                        half(hpc - 1, 0);
                    }
                } else {
                    // Down cores keep h K-tile-major ([K][rows]): this core's NP K-tiles are one run.
                    write_h(src, bc_h + dst + cg * np * mt * h_tile_bytes, np * mt * h_tile_bytes);
                }
            }
            h_pending = 1;
        }
        if (h_pending && ncrisc_noc_nonposted_writes_flushed(noc_index)) {
            // Slices of h(v) are counted per down-core buffer v % HBUF (CT 16.. give buffers 1, 2).
            const uint32_t gid = h_sent % hbuf == 0   ? gath_sem_id
                                 : h_sent % hbuf == 1 ? get_compile_time_arg_val(16)
                                                      : get_compile_time_arg_val(17);
            for (uint32_t hd = 0; hd < nh; ++hd) {
                const uint32_t hxy = get_common_arg_val<uint32_t>(hd);
                noc_semaphore_inc(get_noc_addr(hxy >> 16, hxy & 0xFFFF, get_semaphore(gid)), 1);
            }
            cb_pop_front(h_local_cb, mt * hpc);
            ++h_sent;
            h_pending = 0;
        }
    }
    // x_cons follows compute at most a block a loop pass, so the loop can end (all h sent) with it short of the call's
    // blocks; the end count must be the total (the forwarder ends at fwd == (end - 1) x pieces: a short count left it
    // spinning past it - a 1-in-13k whole-model hang - or would end it before the successor has every block)
    for (const uint32_t x_total = num_v * nk_gu; x_cons < x_total;) {
        invalidate_l1_cache();
        if (cb_pages_reservable_at_back(x_cb, (x_slots - (x_pub - x_cons) + 1) * x_blk)) {
            ++x_cons;
        }
    }
    *sem(flat_options::x_consumed_sem) = x_cons;
    *sem(flat_options::x_end_sem) = x_cons + 1;
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    // leave no NoC transaction in flight (reads, writes, atomics, posted writes): the next program starts clean
    noc_async_full_barrier();
}
