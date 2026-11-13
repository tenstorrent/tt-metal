// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/tensor/tensor_accessor.h"
#include "../persistent.h"

// The persistent op's sharded router, a partial core (RISCV_0). The op's chips split the router by expert: chip
// m scores experts [32m, 32m + 32), one tile column of the gate, whose [K, 32] slice for each layer is its own
// tensor (the layer table's router gate address), K-split over the partial cores; this core's k_slice tiles,
// rows [k_start, +k_slice), sit in its L1 (or wherever the layer's tensor puts them). Per iteration it
//   1. publishes the row (cb_rin is aliased over this core's replica shard) and reads its gate tiles of the
//      layer into cb_rw;
//   2. takes router_partial.cpp's 1x32 partial from cb_part and writes it into slot `part` of the reducer's
//      cb_parts (the same address there), then bumps the reducer's parts count (a GlobalSemaphore).
//
// Compile-time args:
//   0: cb_rin   1: k_tiles   2: cb_rw   3: k_slice   4: gate tile bytes   5: cb_part   6: cb_parts
//   7: part tile bytes   8+: TensorAccessorArgs(gate)
//
// Runtime args: 0: k_start   1: part (this core's slot)   2: reducer NoC x   3: reducer NoC y
//   4: the reducer's parts count address

void kernel_main() {
    constexpr uint32_t cb_rin_id = get_compile_time_arg_val(0);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rw_id = get_compile_time_arg_val(2);
    constexpr uint32_t k_slice = get_compile_time_arg_val(3);
    constexpr uint32_t gate_tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t cb_part_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_parts_id = get_compile_time_arg_val(6);
    constexpr uint32_t part_tile_bytes = get_compile_time_arg_val(7);
    constexpr auto gate_args = TensorAccessorArgs<8>();

    const uint32_t k_start = get_arg_val<uint32_t>(0);
    const uint32_t part = get_arg_val<uint32_t>(1);
    const uint32_t reducer_x = get_arg_val<uint32_t>(2);
    const uint32_t reducer_y = get_arg_val<uint32_t>(3);
    const uint32_t parts_count_addr = get_arg_val<uint32_t>(4);

    for (uint32_t iter = 0; fe_begin_iteration(iter, cb_rin_id, k_tiles, /*has_writer=*/false); ++iter) {
        cb_reserve_back(cb_rin_id, k_tiles);
        cb_push_back(cb_rin_id, k_tiles);
        {
            DeviceZoneScopedN("FE_R_PART_READ");
            const auto gate = TensorAccessor(gate_args, fe_header(kFeHdrRouterGate));
            cb_reserve_back(cb_rw_id, k_slice);
            const uint32_t dst = get_write_ptr(cb_rw_id);
            for (uint32_t t = 0; t < k_slice; ++t) {
                noc_async_read(gate.get_noc_addr(k_start + t), dst + t * gate_tile_bytes, gate_tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_rw_id, k_slice);
        }
        cb_wait_front(cb_part_id, 1);
        noc_async_write(
            get_read_ptr(cb_part_id),
            get_noc_addr(reducer_x, reducer_y, get_write_ptr(cb_parts_id) + part * part_tile_bytes),
            part_tile_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(reducer_x, reducer_y, parts_count_addr), 1);
        noc_async_atomic_barrier();
        cb_pop_front(cb_part_id, 1);
    }
}
