// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// WRITE_{INPUT,OTHER}_GRAD defines gate per-slot NoC write; wait/pop stay unconditional
// so compute's push always drains (partial-mask callers skip only the DRAM store).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

#ifndef WRITE_INPUT_GRAD
#define WRITE_INPUT_GRAD 1
#endif
#ifndef WRITE_OTHER_GRAD
#define WRITE_OTHER_GRAD 1
#endif

void kernel_main() {
    const uint32_t dst0_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst1_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out1 = get_compile_time_arg_val(1);

    constexpr auto dst0_args = TensorAccessorArgs<2, 0>();
    constexpr auto dst1_args =
        TensorAccessorArgs<dst0_args.next_compile_time_args_offset(), dst0_args.next_common_runtime_args_offset()>();

    Noc noc;
    DataflowBuffer dfb0(cb_id_out0);
    DataflowBuffer dfb1(cb_id_out1);

    const uint32_t tile_bytes_0 = dfb0.get_entry_size();
    const uint32_t tile_bytes_1 = dfb1.get_entry_size();
    const auto s0 = TensorAccessor(dst0_args, dst0_addr);
    const auto s1 = TensorAccessor(dst1_args, dst1_addr);
    constexpr uint32_t onetile = 1;

    const uint32_t end_id = start_id + num_tiles;
    for (uint32_t tile_id = start_id; tile_id < end_id; ++tile_id) {
        dfb0.wait_front(onetile);
#if WRITE_INPUT_GRAD
        noc.async_write(dfb0, s0, tile_bytes_0, {}, {.page_id = tile_id});
#endif

        dfb1.wait_front(onetile);
#if WRITE_OTHER_GRAD
        noc.async_write(dfb1, s1, tile_bytes_1, {}, {.page_id = tile_id});
#endif

#if WRITE_INPUT_GRAD || WRITE_OTHER_GRAD
        noc.async_writes_flushed();
#endif
        dfb0.pop_front(onetile);
        dfb1.pop_front(onetile);
    }
#if WRITE_INPUT_GRAD || WRITE_OTHER_GRAD
    noc.async_write_barrier();
#endif
}
