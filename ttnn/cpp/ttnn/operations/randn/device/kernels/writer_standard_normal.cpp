// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

using namespace tt;

void kernel_main() {
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t num_pairs = num_tiles >> 1;

    const uint32_t is_odd = num_tiles & 1;

    const auto output_addrg = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    CircularBuffer cb_dst(dst_cb_id);
    const uint32_t dst_tile_bytes = cb_dst.get_tile_size();

    for (uint32_t p = 0; p < num_pairs; p++) {
        uint32_t i = start_id + (p << 1);
        cb_dst.wait_front(2);

        uint32_t dst_cb_read_base = cb_dst.get_read_ptr();
        uint32_t dst_cb_read0_ptr = dst_cb_read_base;
        uint32_t dst_cb_read1_ptr = dst_cb_read_base + dst_tile_bytes;

        noc.async_write(
            CoreLocalMem<uint32_t>(dst_cb_read0_ptr),
            output_addrg,
            dst_tile_bytes,
            {},
            {.page_id = i});
        noc.async_write(
            CoreLocalMem<uint32_t>(dst_cb_read1_ptr),
            output_addrg,
            dst_tile_bytes,
            {},
            {.page_id = i + 1});
        noc.async_write_barrier();
        cb_dst.pop_front(2);
    }

    if (is_odd) {
        uint32_t i = start_id + (num_pairs << 1);
        cb_dst.wait_front(1);

        uint32_t dst_cb_read0_ptr = cb_dst.get_read_ptr();

        noc.async_write(
            CoreLocalMem<uint32_t>(dst_cb_read0_ptr),
            output_addrg,
            dst_tile_bytes,
            {},
            {.page_id = i});
        noc.async_write_barrier();
        cb_dst.pop_front(1);
    }
}
