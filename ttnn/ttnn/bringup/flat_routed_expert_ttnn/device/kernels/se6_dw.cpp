// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Spatially pipelined expert: a down core's weight reader (NCRISC, NOC1). Streams this core's down weight slice of
// every expert ([KBLK_D x PCD] blocks, a contiguous region of one DRAM bank, one read per block) into its in1 ring,
// which holds two experts so the next expert loads while the current one is still in use. Up to BATCH blocks per read
// barrier.
//
// CT: 0 IN1_CB, 1 SLOT_TILES, 2 TILE_BYTES, 3 BLOCKS (all experts), 4 BATCH, 5 IS_BFP8, 6 NUM_E (SE_DYN), 7 SLOT_X,
//     8 PER_X, 9 X_CB (SE_SMALL_T: the small-M extra column slice)
// RT: 0 weight bank base address, 1 bank id, 2 byte offset of this core's region in the bank, 3.. se_dyn.hpp args
// (SE_DYN)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef SE_DYN
#include "se_dyn.hpp"
#endif

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t slot = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t blocks = get_compile_time_arg_val(3);
    constexpr uint32_t batch = get_compile_time_arg_val(4);
    constexpr bool is_bfp8 = get_compile_time_arg_val(5) != 0;
#ifdef SE_DW_DELAY
    // Experiment: leave DRAM to the gate/up readers while they load the first expert.
    {
        auto clk = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
        const uint32_t t0 = clk();
        while (clk() - t0 < static_cast<uint32_t>(SE_DW_DELAY)) {
        }
    }
#endif
    const uint64_t src =
        get_noc_addr_from_bank_id<true>(get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(0) + get_arg_val<uint32_t>(2));
#ifdef SE_DYN
    // only the active experts' blocks (CT 6 NUM_E: BLOCKS / NUM_E per expert; RT 3.. se_dyn.hpp args, CB 7 scratch)
    constexpr uint32_t num_e = get_compile_time_arg_val(6);
    constexpr uint32_t per_e = blocks / num_e;
    SeDyn dyn;
    se_dyn_load<num_e>(dyn, 3, get_write_ptr(tt::CBIndex::c_7) + 2 * SE_DYN_HALF, 1);  // NCRISC: upper half of CB 7
#ifdef SE_DN_REG
    // Pinned schedule (se_dyn.hpp): one read per gate/up load, into that load's ring region (the ring holds
    // SE_GU_NREG experts); the compute pops by count after each block's last use.
    const uint32_t total = dyn.n_load * per_e;
    const uint32_t ring_base = get_write_ptr(cb);  // nothing pushed yet
    auto dst = [&](uint32_t b, uint32_t) {
        return ring_base + (dyn.region[b / per_e] * per_e + b % per_e) * slot * tile_bytes;
    };
#else
    const uint32_t total = dyn.n_act * per_e;  // every schedule entry (a pinned expert's chunks re-read)
    auto dst = [&](uint32_t, uint32_t i) { return get_write_ptr(cb) + i * slot * tile_bytes; };
#endif
    auto blk = [&](uint32_t b) {
#ifdef SE_DN_REG
        return dyn.load_eid[b / per_e] * per_e + b % per_e;
#else
        return dyn.eid[b / per_e] * per_e + b % per_e;
#endif
    };
#else
    constexpr uint32_t total = blocks;
    auto blk = [](uint32_t b) { return b; };
    auto dst = [&](uint32_t, uint32_t i) { return get_write_ptr(cb) + i * slot * tile_bytes; };
#endif
    for (uint32_t b = 0; b < total;) {
        uint32_t n = total - b < batch ? total - b : batch;
#if defined(SE_DYN) && defined(SE_DN_REG)
        // a batch must not straddle a load boundary: its reservation could need slots of the load after it, which a
        // pinned schedule only frees once a later load retires (codex review: DW_BATCH 3, counts 1024,32,32 hung)
        n = n < per_e - b % per_e ? n : per_e - b % per_e;
#endif
        uint32_t l1[batch];
        for (uint32_t i = 0; i < n; ++i) {
            cb_reserve_back(cb, (i + 1) * slot);
            l1[i] = dst(b + i, i);
        }
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read(src + blk(b + i) * slot * tile_bytes, l1[i], slot * tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb, n * slot);
#if defined(SE_DYN) && defined(SE_SMALL_T)
        // Small-M role split: this core also computes an extra column slice (the reader tails' columns); its weights
        // ([KBLK_X x PCX] blocks, CT 7 SLOT_X, 8 PER_X per expert) go into CB 9 after each expert's main blocks.
        // RT after the se_dyn.hpp args: extra region bank base, bank id, byte offset, has an extra slice.
        {
            constexpr uint32_t x_cb = get_compile_time_arg_val(9), slot_x = get_compile_time_arg_val(7);
            constexpr uint32_t per_x = get_compile_time_arg_val(8);
            const uint32_t xa = 3 + SE_DYN_NARGS;
            if (dyn.small && get_arg_val<uint32_t>(xa + 3) && (b + n) % per_e == 0) {
                const uint64_t xsrc = get_noc_addr_from_bank_id<true>(
                    get_arg_val<uint32_t>(xa + 1), get_arg_val<uint32_t>(xa) + get_arg_val<uint32_t>(xa + 2));
                const uint32_t e = dyn.eid[(b + n) / per_e - 1];
                for (uint32_t j = 0; j < per_x; ++j) {
                    cb_reserve_back(x_cb, slot_x);
                    noc_async_read(
                        xsrc + (e * per_x + j) * slot_x * tile_bytes, get_write_ptr(x_cb), slot_x * tile_bytes);
                    noc_async_read_barrier();
                    cb_push_back(x_cb, slot_x);
                }
            }
        }
#endif
        b += n;
    }
    // leave no NoC transaction in flight (reads, writes, atomics, posted writes): the next program starts clean
    noc_async_full_barrier();
}
