// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    const uint32_t gathered_addr = get_arg_val<uint32_t>(0);
    const uint32_t initial_addr = get_arg_val<uint32_t>(1);
    const uint32_t output_addr = get_arg_val<uint32_t>(2);
    const uint32_t output_tiles = get_arg_val<uint32_t>(3);
    const uint32_t rank = get_arg_val<uint32_t>(4);
    const uint32_t source_offset = get_arg_val<uint32_t>(5);

    constexpr auto gathered_args = TensorAccessorArgs<0>();
    constexpr auto initial_args = TensorAccessorArgs<gathered_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<initial_args.next_compile_time_args_offset()>();
    const auto gathered = TensorAccessor(gathered_args, gathered_addr);
    const auto initial = TensorAccessor(initial_args, initial_addr);
    const auto output = TensorAccessor(output_args, output_addr);

    constexpr uint32_t tile_bytes = 2048;
    CircularBuffer scratch(tt::CBIndex::c_0);
    Noc noc;
    scratch.reserve_back(1);

    for (uint32_t tile = 0; tile < output_tiles; ++tile) {
        if (rank == 0) {
            noc.async_read(initial, scratch, tile_bytes, {.page_id = tile}, {.offset_bytes = 0});
        } else {
            noc.async_read(gathered, scratch, tile_bytes, {.page_id = source_offset + tile}, {.offset_bytes = 0});
        }
        noc.async_read_barrier();
        noc.async_write(
            use<CircularBuffer::AddrSelector::WRITE_PTR>(scratch),
            output,
            tile_bytes,
            {.offset_bytes = 0},
            {.page_id = tile});
        noc.async_write_barrier();
    }
}
