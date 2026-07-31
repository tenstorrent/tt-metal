// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BAKE-OFF — moe_fused_swiglu x-activation staging, writer half.
//
// Two jobs, both trivial by design (so the reader/compute pair above is the only thing under
// study):
//   1. VARIANT 3 (dual_noc_split) ONLY — read sticks [16,32) on NoC1 (BRISC's default NoC),
//      in parallel with the reader's [0,16) on NoC0 (split_reader's dual-issue pattern), then
//      signal SEM_SPLIT so the reader knows it is safe to push cb_x_in.
//   2. Every variant — wait for the resident tile-row (cb_x_resident, KR_PAD bfp8 tiles) and
//      write it back to a DRAM output tensor so the host can check it against a torch/ttnn
//      reference. Not part of the measured stage; kept off the reader/compute critical path
//      (waits only on the CB, no NoC contention with the read side).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
constexpr uint32_t KR_PAD = get_compile_time_arg_val(1);
constexpr uint32_t X_SLICE = get_compile_time_arg_val(2);
constexpr uint32_t X_PAGE = get_compile_time_arg_val(3);
constexpr uint32_t BFP8_TILE = get_compile_time_arg_val(4);
constexpr uint32_t SEM_SPLIT = get_compile_time_arg_val(5);
constexpr uint32_t TA_BASE = 6;
constexpr auto x_args = TensorAccessorArgs<TA_BASE>();
constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

constexpr uint32_t CB_X_IN = 0;
constexpr uint32_t CB_X_RESIDENT = 2;

constexpr uint32_t TILE_H = 32;

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t kstart_tiles = get_arg_val<uint32_t>(2);
    const uint32_t kr = get_arg_val<uint32_t>(3);

    if constexpr (VARIANT == 3) {
        // The reader's NoC0 half handles sticks [0,16); we take [16,32) on NoC1 in parallel.
        const auto x_acc = TensorAccessor(x_args, x_addr, X_PAGE);
        const uint32_t kstart_bytes = kstart_tiles * TILE_H * 2;
        const uint32_t wp = get_write_ptr(CB_X_IN);  // same address the reader computed; no push yet
        constexpr uint32_t HALF = TILE_H / 2;
        for (uint32_t s = HALF; s < TILE_H; ++s) {
            noc_async_read(x_acc.get_noc_addr(s, kstart_bytes), wp + s * X_SLICE, kr * TILE_H * 2);
        }
        noc_async_read_barrier();
        noc_semaphore_inc(get_noc_addr(static_cast<uint32_t>(get_semaphore(SEM_SPLIT))), 1);
    }

    // ---- writeback for correctness checking only; not on the measured critical path ----
    const auto out_acc = TensorAccessor(out_args, out_addr, BFP8_TILE);
    cb_wait_front(CB_X_RESIDENT, KR_PAD);
    const uint32_t rp = get_read_ptr(CB_X_RESIDENT);
    for (uint32_t i = 0; i < kr; ++i) {
        noc_async_write(rp + i * BFP8_TILE, out_acc.get_noc_addr(i), BFP8_TILE);
    }
    noc_async_write_barrier();
    cb_pop_front(CB_X_RESIDENT, KR_PAD);
}
