// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ---------------------------------------------------------------------------
// dual_noc_write -- the DM ROOFLINE INSTRUMENT.
// ---------------------------------------------------------------------------
// ONE kernel source, instantiated on BOTH data-movement RISC-Vs.  It is a pure
// DRAM->L1->DRAM copy with NO Tensix compute at all, reproducing the focus
// shape's per-core transfer pattern exactly:
//
//   * `num_rows` tile-rows per core (the op's row split over 110 cores)
//   * WT tiles per row, one 2048 B interleaved DRAM page per tile
//   * one `noc_async_read_tile` per tile, one barrier per row-block  (reader)
//   * one `noc_async_write_tile` per tile, one barrier per row-block (writer)
//
// The instrument's variable is WHICH RISC-V (and therefore which NoC) issues
// which half of the traffic.  Each instance owns up to two "lanes":
//
//   R_LANES : lanes this RISC READS   -- it is the CB's sole PRODUCER
//   W_LANES : lanes this RISC WRITES  -- it is the CB's sole CONSUMER
//
// A lane is (tile_start, tile_count, cb_id).  The lane's CB has exactly one
// producer RISC and one consumer RISC, so the CB single-producer/single-consumer
// invariant holds for every configuration -- including the ones where the SAME
// RISC is both (a private scratch buffer; still one producer, one consumer).
//
// RAW-API NOTE: this kernel deliberately uses the raw `noc_async_read_tile` /
// `noc_async_write_tile` + explicit barrier form rather than a dataflow helper.
// The whole point of the instrument is to control transaction count, barrier
// placement and NoC assignment individually, which is exactly what a helper
// abstracts away.
//
// Compile-time args:
//   0  WT              tiles per tile-row (72 on the focus shape)
//   1  TILE_BYTES
//   2  N_R_LANES       0..2
//   3  N_W_LANES       0..2
//   4  ABLATE_READ     stub the read payload (keep the CB scaffolding)
//   5  ABLATE_WRITE    stub the write payload
//   6  WRITE_FLUSH     1 = noc_async_writes_flushed() per block + ONE final
//                      barrier; 0 = full noc_async_write_barrier() per block
//                      (what the shipped writer does)
//   7  R_LANE0_START  8 R_LANE0_COUNT  9 R_LANE0_CB
//  10  R_LANE1_START 11 R_LANE1_COUNT 12 R_LANE1_CB
//  13  W_LANE0_START 14 W_LANE0_COUNT 15 W_LANE0_CB
//  16  W_LANE1_START 17 W_LANE1_COUNT 18 W_LANE1_CB
//  19  in TensorAccessorArgs ...  then out TensorAccessorArgs
//
// Runtime args: 0 = in_addr, 1 = out_addr, 2 = row_start, 3 = num_rows

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t TILE_BYTES = get_compile_time_arg_val(1);
    constexpr uint32_t N_R_LANES = get_compile_time_arg_val(2);
    constexpr uint32_t N_W_LANES = get_compile_time_arg_val(3);
    constexpr bool ABLATE_READ = get_compile_time_arg_val(4) != 0;
    constexpr bool ABLATE_WRITE = get_compile_time_arg_val(5) != 0;
    constexpr bool WRITE_FLUSH = get_compile_time_arg_val(6) != 0;

    constexpr uint32_t R0_START = get_compile_time_arg_val(7);
    constexpr uint32_t R0_COUNT = get_compile_time_arg_val(8);
    constexpr uint32_t R0_CB = get_compile_time_arg_val(9);
    constexpr uint32_t R1_START = get_compile_time_arg_val(10);
    constexpr uint32_t R1_COUNT = get_compile_time_arg_val(11);
    constexpr uint32_t R1_CB = get_compile_time_arg_val(12);
    constexpr uint32_t W0_START = get_compile_time_arg_val(13);
    constexpr uint32_t W0_COUNT = get_compile_time_arg_val(14);
    constexpr uint32_t W0_CB = get_compile_time_arg_val(15);
    constexpr uint32_t W1_START = get_compile_time_arg_val(16);
    constexpr uint32_t W1_COUNT = get_compile_time_arg_val(17);
    constexpr uint32_t W1_CB = get_compile_time_arg_val(18);

    // COALESCE_BANKS != 0 -> the BANK-RUN transform (see the head comment below).
    constexpr uint32_t NBANKS = get_compile_time_arg_val(19);

    // ALT_TAIL != 0 -> the SINGLE-RISC DUAL-NOC write.  The LAST ALT_TAIL tiles
    // of write lane 0 are issued on ALT_NOC instead of this kernel's own NoC, by
    // THIS SAME RISC-V.  Requires noc_mode == DM_DYNAMIC_NOC on BOTH data-movement
    // kernels: under DM_DEDICATED_NOC each RISC tracks its issued-transaction
    // count locally against a NoC-global hardware counter, so two RISCs sharing a
    // NoC hang in the barrier (measured: `both_noc0`/`both_noc1` deadlock).
    constexpr uint32_t ALT_TAIL = get_compile_time_arg_val(20);
    constexpr uint8_t ALT_NOC = (uint8_t)get_compile_time_arg_val(21);

    constexpr auto in_args = TensorAccessorArgs<22>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t row_start = get_arg_val<uint32_t>(2);
    const uint32_t num_rows = get_arg_val<uint32_t>(3);

    const auto in_acc = TensorAccessor(in_args, in_addr, TILE_BYTES);
    const auto out_acc = TensorAccessor(out_args, out_addr, TILE_BYTES);

    // -----------------------------------------------------------------------
    // BANK-RUN COALESCING (NBANKS != 0).  A DRAM-interleaved tensor maps page p
    // to bank p % NUM_DRAM_BANKS at bank-local offset (p / NUM_DRAM_BANKS) *
    // aligned_page_size.  WT is a multiple of NUM_DRAM_BANKS on every shape this
    // op sees, so a core's tile-row [base, base+WT) lands as NBANKS runs of
    // WT/NBANKS pages that are CONTIGUOUS inside their bank.  One NoC read per
    // bank therefore replaces WT/NBANKS reads.  The tiles arrive in L1 in
    // bank-major order; the write half applies the identical transform, so the
    // DRAM->DRAM copy is bit-exact (this is a ROOFLINE PROBE for "how fast can
    // 75.5 MB move at all", not a drop-in for the op, whose compute needs tile
    // order).
    //
    // RAW-API: this bypasses `noc_async_read_tile`/`noc_async_write_tile` (and
    // every dataflow helper built on them) because those are page-at-a-time by
    // construction -- expressing a multi-page contiguous bank run is exactly the
    // capability they do not have.
    constexpr uint32_t RUN_TILES = (NBANKS != 0) ? (WT / NBANKS) : 0;
    constexpr uint32_t RUN_BYTES = RUN_TILES * TILE_BYTES;

    for (uint32_t r = 0; r < num_rows; ++r) {
        const uint32_t tile_base = (row_start + r) * WT;

        if constexpr (NBANKS != 0) {
            static_assert(NBANKS == 0 || WT % NBANKS == 0, "bank-run coalescing needs NBANKS | WT");
            if constexpr (N_R_LANES >= 1) {
                cb_reserve_back(R0_CB, WT);
                uint32_t l1 = get_write_ptr(R0_CB);
                if constexpr (!ABLATE_READ) {
                    for (uint32_t b = 0; b < NBANKS; ++b) {
                        noc_async_read(in_acc.get_noc_addr(tile_base + b), l1 + b * RUN_BYTES, RUN_BYTES);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(R0_CB, WT);
            }
            if constexpr (N_W_LANES >= 1) {
                cb_wait_front(W0_CB, WT);
                uint32_t wl1 = get_read_ptr(W0_CB);
                if constexpr (!ABLATE_WRITE) {
                    for (uint32_t b = 0; b < NBANKS; ++b) {
                        noc_async_write(wl1 + b * RUN_BYTES, out_acc.get_noc_addr(tile_base + b), RUN_BYTES);
                    }
                }
                if constexpr (WRITE_FLUSH) {
                    noc_async_writes_flushed();
                } else {
                    noc_async_write_barrier();
                }
                cb_pop_front(W0_CB, WT);
            }
            continue;
        }

        // ---- READ half: this RISC is the sole producer of each read lane ----
        if constexpr (N_R_LANES >= 1) {
            cb_reserve_back(R0_CB, R0_COUNT);
            uint32_t l1 = get_write_ptr(R0_CB);
            if constexpr (!ABLATE_READ) {
                for (uint32_t w = 0; w < R0_COUNT; ++w) {
                    noc_async_read_tile(tile_base + R0_START + w, in_acc, l1);
                    l1 += TILE_BYTES;
                }
            }
        }
        if constexpr (N_R_LANES >= 2) {
            cb_reserve_back(R1_CB, R1_COUNT);
            uint32_t l1 = get_write_ptr(R1_CB);
            if constexpr (!ABLATE_READ) {
                for (uint32_t w = 0; w < R1_COUNT; ++w) {
                    noc_async_read_tile(tile_base + R1_START + w, in_acc, l1);
                    l1 += TILE_BYTES;
                }
            }
        }
        if constexpr (N_R_LANES >= 1) {
            noc_async_read_barrier();
            cb_push_back(R0_CB, R0_COUNT);
            if constexpr (N_R_LANES >= 2) {
                cb_push_back(R1_CB, R1_COUNT);
            }
        }

        // ---- WRITE half: this RISC is the sole consumer of each write lane ----
        if constexpr (N_W_LANES >= 1) {
            cb_wait_front(W0_CB, W0_COUNT);
            uint32_t l1 = get_read_ptr(W0_CB);
            if constexpr (!ABLATE_WRITE) {
                constexpr uint32_t OWN = (ALT_TAIL < W0_COUNT) ? (W0_COUNT - ALT_TAIL) : 0;
                for (uint32_t w = 0; w < OWN; ++w) {
                    noc_async_write_tile(tile_base + W0_START + w, out_acc, l1);
                    l1 += TILE_BYTES;
                }
                for (uint32_t w = OWN; w < W0_COUNT; ++w) {
                    noc_async_write_tile(tile_base + W0_START + w, out_acc, l1, ALT_NOC);
                    l1 += TILE_BYTES;
                }
            }
        }
        if constexpr (N_W_LANES >= 2) {
            cb_wait_front(W1_CB, W1_COUNT);
            uint32_t l1 = get_read_ptr(W1_CB);
            if constexpr (!ABLATE_WRITE) {
                for (uint32_t w = 0; w < W1_COUNT; ++w) {
                    noc_async_write_tile(tile_base + W1_START + w, out_acc, l1);
                    l1 += TILE_BYTES;
                }
            }
        }
        if constexpr (N_W_LANES >= 1) {
            if constexpr (WRITE_FLUSH) {
                noc_async_writes_flushed();  // data has LEFT L1 -> the CB slot is reusable
                if constexpr (ALT_TAIL != 0) {
                    noc_async_writes_flushed(ALT_NOC);
                }
            } else {
                noc_async_write_barrier();  // the shipped writer's per-block full ACK wait
                if constexpr (ALT_TAIL != 0) {
                    noc_async_write_barrier(ALT_NOC);
                }
            }
            cb_pop_front(W0_CB, W0_COUNT);
            if constexpr (N_W_LANES >= 2) {
                cb_pop_front(W1_CB, W1_COUNT);
            }
        }
    }
    noc_async_write_barrier();
    if constexpr (ALT_TAIL != 0) {
        noc_async_write_barrier(ALT_NOC);
    }
}
