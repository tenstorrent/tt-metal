// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    // single-tile ublocks
    constexpr uint32_t onetile = 1;

    CircularBuffer cb_out(cb_id_out);

#ifndef OUT_SHARDED
    const auto s = TensorAccessor(dst_args, dst_addr);
#endif

#ifdef DECODE_MODE
    uint32_t cos_sin_offset = get_arg_val<uint32_t>(3);
    uint32_t Wt = get_arg_val<uint32_t>(4);
    uint32_t Wbytes = get_arg_val<uint32_t>(5);

    constexpr uint32_t decode_cta_offset = dst_args.next_compile_time_args_offset();
    constexpr uint32_t untilized_cos_cb_id = get_compile_time_arg_val(decode_cta_offset + 0);
    constexpr uint32_t untilized_cos_sync_cb_id = get_compile_time_arg_val(decode_cta_offset + 1);
    constexpr uint32_t untilized_sin_cb_id = get_compile_time_arg_val(decode_cta_offset + 2);
    constexpr uint32_t untilized_sin_sync_cb_id = get_compile_time_arg_val(decode_cta_offset + 3);

    CircularBuffer cb_untilized_cos(untilized_cos_cb_id);
    CircularBuffer cb_untilized_cos_sync(untilized_cos_sync_cb_id);
    CircularBuffer cb_untilized_sin(untilized_sin_cb_id);
    CircularBuffer cb_untilized_sin_sync(untilized_sin_sync_cb_id);

    cb_untilized_sin.wait_front(Wt);
    cb_untilized_sin_sync.reserve_back(Wt);
    uint32_t sin_l1_read_addr = cb_untilized_sin.get_read_ptr() + cos_sin_offset;
    uint32_t sin_l1_write_addr = cb_untilized_sin.get_read_ptr();
    noc.async_read(
        UnicastEndpoint{},
        CoreLocalMem<uint32_t>(sin_l1_write_addr),
        Wbytes,
        {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
         .noc_y = (uint32_t)my_y[noc.get_noc_id()],
         .addr = sin_l1_read_addr},
        {});
    noc.async_read_barrier();
    cb_untilized_sin_sync.push_back(Wt);

    cb_untilized_cos.wait_front(Wt);
    cb_untilized_cos_sync.reserve_back(Wt);
    uint32_t cos_l1_read_addr = cb_untilized_cos.get_read_ptr() + cos_sin_offset;
    uint32_t cos_l1_write_addr = cb_untilized_cos.get_read_ptr();
    noc.async_read(
        UnicastEndpoint{},
        CoreLocalMem<uint32_t>(cos_l1_write_addr),
        Wbytes,
        {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
         .noc_y = (uint32_t)my_y[noc.get_noc_id()],
         .addr = cos_l1_read_addr},
        {});
    noc.async_read_barrier();
    cb_untilized_cos_sync.push_back(Wt);
#endif

#ifdef OUT_SHARDED
    cb_out.wait_front(num_tiles);
#else
    constexpr uint32_t out_tile_size = get_tile_size(cb_id_out);
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_out.wait_front(onetile);
        uint32_t l1_read_addr = cb_out.get_read_ptr();

        noc.async_write(CoreLocalMem<uint32_t>(l1_read_addr), s, out_tile_size, {}, {.page_id = i});

        noc.async_write_barrier();

        cb_out.pop_front(onetile);
    }
#endif
}
