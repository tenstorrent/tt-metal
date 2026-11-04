// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Spatially pipelined expert: a down core's weight reader (NCRISC, NOC1). Streams this core's down weight slice of
// every expert ([KBLK_D x PCD] blocks, a contiguous region of one DRAM bank, one read per block) into its in1 ring,
// which holds two experts so the next expert loads while the current one is still in use. Up to BATCH blocks per read
// barrier.
//
// CT: 0 IN1_CB, 1 SLOT_TILES, 2 TILE_BYTES, 3 BLOCKS (all experts), 4 BATCH, 5 NUM_E
// RT: 0 weight bank base address, 1 bank id, 2 byte offset of this core's region in the bank, 3.. se_dyn.hpp args
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "se_dyn.hpp"
#ifdef SE_Y_NC
// Row-major y written from here (NOC0) instead of the down core's BRISC (NOC1, the h chain's network): the weight
// reads' waits poll the y writer (CT 6 Y_MT: row tiles per virtual expert, 7 PCD, 8 HT; RT after the se_dyn.hpp
// args: y address, first y tile column)
#include "se_yrm.hpp"
#include "se_cmb_done.hpp"  // (combine overlap: this core's y writer reports to combine's collector)
#endif

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t slot = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t blocks = get_compile_time_arg_val(3);
    constexpr uint32_t batch = get_compile_time_arg_val(4);
    const uint64_t src =
        get_noc_addr_from_bank_id<true>(get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(0) + get_arg_val<uint32_t>(2));
    // block k of this core's sequence sits in bank (bank + k) % SE_DW_ROT at offset + (k / SE_DW_ROT) blocks (the host
    // packs it so: flat_weight_regions with the plan's dw_rot), so every core's reads rotate over the banks
    const uint32_t rot_bank = get_arg_val<uint32_t>(1), rot_addr = get_arg_val<uint32_t>(0) + get_arg_val<uint32_t>(2);
    auto wsrc = [&](uint32_t k) {
        return get_noc_addr_from_bank_id<true>(
            (rot_bank + k) % SE_DW_ROT, rot_addr + (k / SE_DW_ROT) * slot * tile_bytes);
    };
    // only the active experts' blocks (CT 5 NUM_E: BLOCKS / NUM_E per expert; RT 3.. se_dyn.hpp args, CB 7 scratch)
    constexpr uint32_t num_e = get_compile_time_arg_val(5);
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
#ifdef SE_Y_NC
    static_assert(get_compile_time_arg_val(6) >= 1);
    const uint32_t ya = 3 + SE_DYN_NARGS;
    SeYRmWriter<
        tt::CBIndex::c_16,
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(8)>
        yw(get_arg_val<uint32_t>(ya), get_arg_val<uint32_t>(ya + 1));
    uint32_t y_done = 0;
#ifdef SE_CMB_DONE_RT
    SeCmbDone cmb;
    cmb.init(dyn, num_e, true);
#endif
    auto y_poll = [&]() {
        yw.issue(dyn);
        if (yw.retire(dyn)) {
            ++y_done;
#ifdef SE_CMB_DONE_RT
            cmb.wrote(dyn.eid[yw.done_a]);
            cmb.report(false);
#endif
        }
    };
#else
    auto y_poll = []() {};
#endif
    for (uint32_t b = 0; b < total;) {
        uint32_t n = total - b < batch ? total - b : batch;
#ifdef SE_DN_REG
        // a batch must not straddle a load boundary: its reservation could need slots of the load after it, which a
        // pinned schedule only frees once a later load retires (codex review: DW_BATCH 3, counts 1024,32,32 hung)
        n = n < per_e - b % per_e ? n : per_e - b % per_e;
#else
        // a batch must not straddle the ring's end either: block b sits in ring slot b % SE_DW_RING (as the compute
        // reads it), a batch's writes are linear from the write pointer (a 1.5-expert ring of 27 blocks with batch 2
        // put every other lap's second block past the ring, onto the h buffers, and hung the chip: V4.1 units, bf16 h)
        n = n < SE_DW_RING - b % SE_DW_RING ? n : SE_DW_RING - b % SE_DW_RING;
#endif
        uint32_t l1[batch];
        for (uint32_t i = 0; i < n; ++i) {
            while (!cb_pages_reservable_at_back(cb, (i + 1) * slot)) {
                y_poll();
            }
            cb_reserve_back(cb, (i + 1) * slot);
            l1[i] = dst(b + i, i);
        }
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read(wsrc(blk(b + i)), l1[i], slot * tile_bytes);
        }
        while (!ncrisc_noc_reads_flushed(noc_index)) {
            y_poll();
        }
        noc_async_read_barrier();
        cb_push_back(cb, n * slot);
        b += n;
    }
#ifdef SE_Y_NC
    while (y_done < dyn.num_v) {
        invalidate_l1_cache();
        y_poll();
    }
#ifdef SE_Y_L1
    yw.finish();
#endif
    noc_async_write_barrier();
#ifdef SE_CMB_DONE_RT
    cmb.report();  // the experts this core writes nothing of
#endif
    noc_async_write_set_trid(0);  // (the next program's plain writes carry ID 0)
#endif
    // leave no NoC transaction in flight (reads, writes, atomics, posted writes): the next program starts clean
    noc_async_full_barrier();
}
