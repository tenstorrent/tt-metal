// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Regime-A in1 reader == consumer (runs on the core's in1 NoC/RISC).
//
// in1 is DRAM width-sharded across 8 banks; this core owns one bank's N-sub-band for its k-slice. It reads
// that sub-band's [kb, N_sub] blocks in ROTATED shard order (ring_pos, ring_pos-1, ... wrapping mod G) so
// each in1[k] block pairs with the in0[k] block arriving via the in0 ring in the writer. The matmul K-sum
// is commutative, so any consistent pairing is correct.
//
// M-split (m_slices > 1): the m==0 reader of each (bank, k-slice, n-slice) group reads in1 from DRAM ONCE
// and forwards each block to the Sm-1 M-slaves (unicast, or one multicast when IN1_MCAST). Slaves receive
// only. For m_slices == 1 every core is a "solo" reader (mrole == 2): read from DRAM, no forward.
//
// This is the cleaned unified-only port of the prototype reader_ring.cpp (strided sub-band branch + the
// M-split forward). No in0 read (the writer owns in0), no contiguous/deep-K/ablation paths.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t K_block = get_compile_time_arg_val(0);       // kb : K tiles per block
    constexpr uint32_t N_block = get_compile_time_arg_val(1);       // N_sub : N tiles per block
    constexpr uint32_t W = get_compile_time_arg_val(2);             // in1 blocks per ring shard
    constexpr uint32_t G = get_compile_time_arg_val(3);             // ring size (8 banks)
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);    // bf16 tile bytes
    constexpr uint32_t N_bpc = get_compile_time_arg_val(5);         // N-sub-blocks per core
    constexpr uint32_t N_band = get_compile_time_arg_val(6);        // padded per-bank N width (stride, tiles)
    constexpr uint32_t in1valid_sem = get_compile_time_arg_val(7);  // M-split: reader -> slaves "block delivered"
    constexpr uint32_t in1ready_sem = get_compile_time_arg_val(8);  // M-split: slaves -> reader "cb1 slot free"
    constexpr uint32_t in1_mcast = get_compile_time_arg_val(9);     // M-split forward: 1 = mcast, 0 = unicasts

    const uint32_t in1_addr = get_arg_val<uint32_t>(0);  // in1 base DRAM address
    const uint32_t bank_id = get_arg_val<uint32_t>(1);   // this core's DRAM bank
    const uint32_t ring_pos = get_arg_val<uint32_t>(2);  // ring position (rotates shard read order)
    const uint32_t k_start = get_arg_val<uint32_t>(3);   // first K tile of this k-slice (padded coords)
    const uint32_t n_base = get_arg_val<uint32_t>(4);    // this core's N-sub-band offset within the bank (tiles)
    const uint32_t mrole = get_arg_val<uint32_t>(5);     // 0 = slave, 1 = reader(read+fwd), 2 = solo
    const uint32_t mpeers = get_arg_val<uint32_t>(6);    // forward peer count (reader: Sm-1; slave: 1; solo: 0)
    // M-split peer coords (only present when Sm > 1). reader: mpeers*(x,y); slave: reader (x,y) at [7],[8].

    constexpr uint32_t in1_cb = 1;
    constexpr uint32_t in1_blk = K_block * N_block;
    constexpr uint32_t in1_blk_bytes = in1_blk * tile_bytes;

    // ---- M-split SLAVE: receive in1 from the reader, do not touch DRAM. ----
    if (mrole == 0) {
        volatile tt_l1_ptr uint32_t* valid =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(in1valid_sem));
        uint64_t reader_ready =
            get_noc_addr(get_arg_val<uint32_t>(7), get_arg_val<uint32_t>(8), get_semaphore(in1ready_sem));
        const uint32_t nblk = N_bpc * G * W;
        for (uint32_t b = 0; b < nblk; ++b) {
            cb_reserve_back(in1_cb, in1_blk);
            noc_semaphore_inc(reader_ready, 1);    // our cb1 slot is free -> reader may forward block b
            noc_semaphore_wait_min(valid, b + 1);  // reader forwarded block b into it
            cb_push_back(in1_cb, in1_blk);
        }
        return;
    }

    // ---- M-split READER forward helper (no-op for solo). ----
    const uint32_t in1valid_addr = get_semaphore(in1valid_sem);
    const uint32_t in1ready_addr = get_semaphore(in1ready_sem);
    volatile tt_l1_ptr uint32_t* in1ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1ready_addr);
    uint32_t mbc = 0;  // reader's forwarded-block counter
    auto mfwd = [&](uint32_t w1) {
        if (mrole != 1) {
            return;
        }
        noc_semaphore_wait_min(in1ready, (mbc + 1) * mpeers);  // every slave freed slot mbc
        if constexpr (in1_mcast) {
            uint32_t x0 = get_arg_val<uint32_t>(7), y0 = get_arg_val<uint32_t>(8), x1 = get_arg_val<uint32_t>(9),
                     y1 = get_arg_val<uint32_t>(10);
            noc_async_write_multicast(w1, get_noc_multicast_addr(x0, y0, x1, y1, w1), in1_blk_bytes, mpeers);
            noc_async_writes_flushed();
            noc_semaphore_inc_multicast(get_noc_multicast_addr(x0, y0, x1, y1, in1valid_addr), 1, mpeers);
        } else {
            for (uint32_t s = 0; s < mpeers; ++s) {
                uint32_t sx = get_arg_val<uint32_t>(7 + s * 2), sy = get_arg_val<uint32_t>(8 + s * 2);
                noc_async_write(w1, get_noc_addr(sx, sy, w1), in1_blk_bytes);
            }
            noc_async_writes_flushed();
            for (uint32_t s = 0; s < mpeers; ++s) {
                uint32_t sx = get_arg_val<uint32_t>(7 + s * 2), sy = get_arg_val<uint32_t>(8 + s * 2);
                noc_semaphore_inc(get_noc_addr(sx, sy, in1valid_addr), 1);
            }
        }
        ++mbc;
    };

    // ---- Strided sub-band read in rotated shard order. ----
    // For each N-sub-band nb, walk the ring shards s = (ring_pos - step) mod G, each W blocks of [kb, N_sub].
    // Within a block, read N_sub tiles per K-row at stride N_band (real DRAM-sharded layout).
    constexpr uint32_t seg_bytes = N_block * tile_bytes;  // N_sub tiles per K-row
    for (uint32_t nb = 0; nb < N_bpc; ++nb) {
        for (uint32_t step = 0; step < G; ++step) {
            uint32_t s = (ring_pos + G - step) % G;
            for (uint32_t wb = 0; wb < W; ++wb) {
                uint32_t kblk = s * W + wb;
                cb_reserve_back(in1_cb, in1_blk);
                uint32_t w1 = get_write_ptr(in1_cb);
                for (uint32_t kr = 0; kr < K_block; ++kr) {
                    uint32_t ktile = k_start + kblk * K_block + kr;
                    uint32_t off = (ktile * N_band + n_base + nb * N_block) * tile_bytes;
                    noc_async_read(get_noc_addr_from_bank_id<true>(bank_id, in1_addr + off), w1, seg_bytes);
                    w1 += seg_bytes;
                }
                noc_async_read_barrier();
                mfwd(get_write_ptr(in1_cb));  // M-split reader forwards this block to the Sm-1 slaves
                cb_push_back(in1_cb, in1_blk);
            }
        }
    }
}
