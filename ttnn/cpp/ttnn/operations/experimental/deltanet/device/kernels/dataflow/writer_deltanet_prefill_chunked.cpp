// SPDX-License-Identifier: Apache-2.0
//
// Writer for chunked DeltaNet prefill. One core per v-head h. Writes each chunk's
// output [C,Dv] to the [Hv*Sp,Dv] output tensor at tile-row (h*nC + c), then writes
// the final recurrent state [Dk,Dv] to new_state at tile (h*state_tiles + t).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cbOutput  = get_compile_time_arg_val(0);
    constexpr uint32_t cbStateA  = get_compile_time_arg_val(1);
    constexpr uint32_t cbStateB  = get_compile_time_arg_val(2);
    constexpr uint32_t Dv_tiles  = get_compile_time_arg_val(3);
    constexpr uint32_t state_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t nC = get_compile_time_arg_val(5);
    constexpr auto acc_args = TensorAccessorArgs<6>();
    // final state lives in A if last chunk index (nC-1) is even, else B
    constexpr uint32_t cbStateFinal = (nC % 2 == 1) ? cbStateA : cbStateB;

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t state_out_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_row_base = get_arg_val<uint32_t>(2);   // h*nC
    const uint32_t state_out_base = get_arg_val<uint32_t>(3); // h*state_tiles

    const uint32_t tbo = get_tile_size(cbOutput);
    const uint32_t tbs = get_tile_size(cbStateFinal);
    const auto out_acc = TensorAccessor(acc_args, out_addr, tbo);
    const auto state_acc = TensorAccessor(acc_args, state_out_addr, tbs);

    for (uint32_t c = 0; c < nC; c++) {
        cb_wait_front(cbOutput, Dv_tiles);
        uint32_t l1 = get_read_ptr(cbOutput);
        uint32_t base = (out_row_base + c) * Dv_tiles;
        for (uint32_t t = 0; t < Dv_tiles; t++) { noc_async_write_tile(base + t, out_acc, l1); l1 += tbo; }
        noc_async_write_barrier();
        cb_pop_front(cbOutput, Dv_tiles);
    }
    // final state (compute leaves it in cbStateFinal, not consumed by compute)
    cb_wait_front(cbStateFinal, state_tiles);
    uint32_t l1 = get_read_ptr(cbStateFinal);
    for (uint32_t t = 0; t < state_tiles; t++) { noc_async_write_tile(state_out_base + t, state_acc, l1); l1 += tbs; }
    noc_async_write_barrier();
    cb_pop_front(cbStateFinal, state_tiles);
}
