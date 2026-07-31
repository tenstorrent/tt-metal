// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BAKE-OFF — moe_fused_swiglu's x-activation staging path (reader half).
//
// Reconstructs ONE injector's per-tile-row prologue exactly as it exists today in
// moe_fused_swiglu_reader.cpp's `reader_xstage` block (~line 311): read 32 row-major bf16
// stick-slices, barrier, push, wait for compute's fused tilize, self-copy the tilized
// tile-row into the resident slot. VARIANT selects the candidate read strategy; the
// downstream tilize + self-copy + writeback contract is IDENTICAL across variants so any
// measured delta is attributable to the read strategy alone.
//
// VARIANT 0 baseline           — verbatim reconstruction of the op's current approach.
// VARIANT 1 wide_read_individual — read the WHOLE 14336B DRAM page per stick (32 individual
//           whole-page reads) into a wide scratch CB, then extract this core's KR_PAD-tile
//           slice via 32 L1->L1 copies. Isolates "does a bigger transaction (same COUNT)
//           cost more" from "does fewer transactions help".
// VARIANT 2 bank_run_grouped   — the op's own WRUN bank-run trick (moe_fused_swiglu_bank_runs.hpp),
//           applied to WHOLE pages instead of tiles: pages p, p+NUM_BANKS, p+2*NUM_BANKS, ...
//           are physically contiguous inside one DRAM bank (op_design.md §1.5), so ONE
//           `noc_async_read` per bank grabs PAGES_PER_BANK=32/NUM_BANKS whole sticks at once
//           (NUM_BANKS transactions total instead of 32), then 32 L1->L1 copies reassemble
//           row order and extract this core's slice.
// VARIANT 3 dual_noc_split     — split the read across BOTH data-movement RISC-Vs
//           (reader/NoC0 = sticks [0,16), writer/NoC1 = sticks [16,32)), split_reader-style.
// VARIANT 4 bfp8_tile_direct   — the op's INPUT_FORMAT==1 twin: `kr` whole bfp8 tiles read
//           straight into the resident slot, no row-major stick, no tilize. Bounds how much
//           of the bf16_rm cost is the sticks/tilize vs the irreducible DRAM read.
// VARIANT 5 self_copy_ablation — baseline minus the final self-copy (diagnostic only, NOT
//           correctness-gated): isolates the self-copy's share of the stage.
//
// RAW LLK / raw dataflow justification: every variant here uses the SAME raw `noc_async_read`
// + `TensorAccessor` primitives the real op already uses for this stage (no in-tree helper
// expresses "read a bank-contiguous run of WHOLE pages then extract a sub-slice" — the op's own
// BankRuns helper coalesces whole TILES, never row-major sub-page slices). Variants 1/2 add
// manual L1->L1 "gather" reads with `get_noc_addr(local_addr)` (1-arg loopback overload), the
// same primitive the real op's self-copy already uses (`moe_fused_swiglu_reader.cpp`'s
// `cb_wait_front(cb_x_stage, KR_PAD); noc_async_read(get_noc_addr(get_read_ptr(cb_x_stage)), ...)`).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
constexpr uint32_t KR_PAD = get_compile_time_arg_val(1);
constexpr uint32_t X_SLICE = get_compile_time_arg_val(2);  // cb_x_in page: KR_PAD*32*2 bytes
constexpr uint32_t X_PAGE = get_compile_time_arg_val(3);   // full DRAM stick: emb*2 bytes
constexpr uint32_t BFP8_TILE = get_compile_time_arg_val(4);
constexpr uint32_t NUM_BANKS = get_compile_time_arg_val(5);
constexpr uint32_t SEM_SPLIT = get_compile_time_arg_val(6);
constexpr uint32_t TA_BASE = 7;
constexpr auto x_args = TensorAccessorArgs<TA_BASE>();

constexpr uint32_t CB_X_IN = 0;
constexpr uint32_t CB_X_STAGE = 1;
constexpr uint32_t CB_X_RESIDENT = 2;
constexpr uint32_t CB_X_WIDE = 3;
constexpr uint32_t CB_X_BANKGRP = 4;

constexpr uint32_t TILE_H = 32;
constexpr uint32_t X_ROW_BYTES = KR_PAD * BFP8_TILE;
constexpr uint32_t PAGES_PER_BANK = (VARIANT == 2) ? (TILE_H / (NUM_BANKS ? NUM_BANKS : 1)) : 0;
constexpr uint32_t BANK_GROUP_BYTES = PAGES_PER_BANK * X_PAGE;

// The accessor's own page size: X_PAGE (a full bf16 stick) for every row-major variant, or a
// single bfp8 tile (VARIANT 4, the op's INPUT_FORMAT==1 path) — chosen at compile time so ONE
// accessor construction serves every variant.
constexpr uint32_t X_ACC_PAGE = (VARIANT == 4) ? BFP8_TILE : X_PAGE;

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t kstart_tiles = get_arg_val<uint32_t>(1);  // this core's tile-column offset in the stick
    const uint32_t kr = get_arg_val<uint32_t>(2);            // real K tiles (== KR_PAD, no ragged tail here)

    const auto x_acc = TensorAccessor(x_args, x_addr, X_ACC_PAGE);
    const uint32_t kstart_bytes = kstart_tiles * TILE_H * 2;  // bf16 element size

    if constexpr (VARIANT == 4) {
        // ---- bfp8_tile_direct: kr whole tiles straight into the resident slot, no tilize ----
        cb_reserve_back(CB_X_RESIDENT, KR_PAD);
        const uint32_t dst = get_write_ptr(CB_X_RESIDENT);
        for (uint32_t i = 0; i < kr; ++i) {
            noc_async_read(x_acc.get_noc_addr(kstart_tiles + i), dst + i * BFP8_TILE, BFP8_TILE);
        }
        noc_async_read_barrier();
        cb_push_back(CB_X_RESIDENT, KR_PAD);
        return;
    }

    // ---- bf16 ROW_MAJOR variants: land 32 stick-slices in cb_x_in ----
    cb_reserve_back(CB_X_IN, TILE_H);
    const uint32_t wp = get_write_ptr(CB_X_IN);

    if constexpr (VARIANT == 0 || VARIANT == 5) {
        // BASELINE (verbatim moe_fused_swiglu_reader.cpp reader_xstage): 32 separate
        // sub-page transactions, one per stick, ONE barrier for all 32.
        for (uint32_t s = 0; s < TILE_H; ++s) {
            noc_async_read(x_acc.get_noc_addr(s, kstart_bytes), wp + s * X_SLICE, kr * TILE_H * 2);
        }
        noc_async_read_barrier();
    } else if constexpr (VARIANT == 1) {
        // WIDE_READ_INDIVIDUAL: 32 whole-page (14336 B) reads, ONE barrier, then 32 L1->L1
        // extraction copies (ONE barrier) pulling this core's kr-tile slice out of each
        // landed page into cb_x_in at the correct row offset.
        cb_reserve_back(CB_X_WIDE, TILE_H);
        const uint32_t wpw = get_write_ptr(CB_X_WIDE);
        for (uint32_t s = 0; s < TILE_H; ++s) {
            noc_async_read(x_acc.get_noc_addr(s, 0), wpw + s * X_PAGE, X_PAGE);
        }
        noc_async_read_barrier();
        cb_push_back(CB_X_WIDE, TILE_H);

        cb_wait_front(CB_X_WIDE, TILE_H);
        const uint32_t rpw = get_read_ptr(CB_X_WIDE);
        for (uint32_t s = 0; s < TILE_H; ++s) {
            noc_async_read(get_noc_addr(rpw + s * X_PAGE + kstart_bytes), wp + s * X_SLICE, kr * TILE_H * 2);
        }
        noc_async_read_barrier();
        cb_pop_front(CB_X_WIDE, TILE_H);
    } else if constexpr (VARIANT == 2) {
        // BANK_RUN_GROUPED: the op's own WRUN trick (moe_fused_swiglu_bank_runs.hpp), applied to
        // WHOLE pages. Pages p, p+NUM_BANKS, p+2*NUM_BANKS, ... are physically contiguous inside
        // ONE DRAM bank (op_design.md §1.5), so bank g's PAGES_PER_BANK sticks land as ONE
        // `noc_async_read` of PAGES_PER_BANK*X_PAGE bytes. NUM_BANKS transactions total instead
        // of 32 for the DRAM side; then 32 L1->L1 copies reassemble row order + extract the slice.
        // (NUM_BANKS dividing TILE_H is asserted host-side in xstage_bench.py's build_program --
        // NOT as a static_assert here: `if constexpr (VARIANT == 2)` in this non-template function
        // does not exempt a non-dependent static_assert from being evaluated for OTHER VARIANTs,
        // since the "discarded statement" carve-out only applies inside templates.)
        cb_reserve_back(CB_X_BANKGRP, NUM_BANKS);
        const uint32_t wpg = get_write_ptr(CB_X_BANKGRP);
        for (uint32_t g = 0; g < NUM_BANKS; ++g) {
            noc_async_read(x_acc.get_noc_addr(g, 0), wpg + g * BANK_GROUP_BYTES, BANK_GROUP_BYTES);
        }
        noc_async_read_barrier();
        cb_push_back(CB_X_BANKGRP, NUM_BANKS);

        cb_wait_front(CB_X_BANKGRP, NUM_BANKS);
        const uint32_t rpg = get_read_ptr(CB_X_BANKGRP);
        for (uint32_t s = 0; s < TILE_H; ++s) {
            const uint32_t g = s % NUM_BANKS;
            const uint32_t slot = s / NUM_BANKS;
            const uint32_t src = rpg + g * BANK_GROUP_BYTES + slot * X_PAGE + kstart_bytes;
            noc_async_read(get_noc_addr(src), wp + s * X_SLICE, kr * TILE_H * 2);
        }
        noc_async_read_barrier();
        cb_pop_front(CB_X_BANKGRP, NUM_BANKS);
    } else if constexpr (VARIANT == 3) {
        // DUAL_NOC_SPLIT: this (reader/NoC0) half does sticks [0,16); the writer kernel
        // (NoC1) independently computes the SAME cb_x_in write pointer (get_write_ptr is a
        // shared-L1, read-only query — safe before any push_back this launch) and does
        // sticks [16,32) in parallel. We wait on SEM_SPLIT before pushing so compute never
        // sees a partially-landed cb_x_in.
        constexpr uint32_t HALF = TILE_H / 2;
        for (uint32_t s = 0; s < HALF; ++s) {
            noc_async_read(x_acc.get_noc_addr(s, kstart_bytes), wp + s * X_SLICE, kr * TILE_H * 2);
        }
        noc_async_read_barrier();
        const uint32_t sem_addr = static_cast<uint32_t>(get_semaphore(SEM_SPLIT));
        volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
        noc_semaphore_wait(sem_ptr, 1);
        noc_semaphore_set(sem_ptr, 0);
    }

    cb_push_back(CB_X_IN, TILE_H);

    // ---- wait for compute's fused tilize, then self-copy into the resident slot ----
    // Identical to the real op's reader_xstage tail for every variant (incl. the ablation,
    // MINUS the self-copy itself for VARIANT 5).
    cb_wait_front(CB_X_STAGE, KR_PAD);
    cb_reserve_back(CB_X_RESIDENT, KR_PAD);
    const uint32_t dst = get_write_ptr(CB_X_RESIDENT);
    if constexpr (VARIANT != 5) {
        noc_async_read(get_noc_addr(get_read_ptr(CB_X_STAGE)), dst, X_ROW_BYTES);
        noc_async_read_barrier();
    }
    cb_pop_front(CB_X_STAGE, KR_PAD);
    cb_push_back(CB_X_RESIDENT, KR_PAD);
}
