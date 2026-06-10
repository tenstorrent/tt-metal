// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t num_rows = get_compile_time_arg_val(1);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(2);
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t input_page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(6);
    constexpr auto input_args = TensorAccessorArgs<7>();

    const auto input = TensorAccessor(input_args, src_addr, input_page_bytes);
    CircularBuffer input_cb(cb_in);
    Noc noc;

    for (uint32_t row = 0; row < num_rows; ++row) {
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            for (uint32_t tile = 0; tile < tiles_per_chunk; ++tile) {
                input_cb.reserve_back(1);
                const uint32_t tile_offset = tile * tile_bytes;
                const uint32_t read_bytes =
                    tile_offset < chunk_bytes
                        ? (chunk_bytes - tile_offset < tile_bytes ? chunk_bytes - tile_offset : tile_bytes)
                        : 0;
                if (read_bytes != 0) {
                    noc.async_read(
                        input,
                        input_cb,
                        read_bytes,
                        {.page_id = row, .offset_bytes = chunk * chunk_bytes + tile_offset},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                }
                input_cb.push_back(1);
            }
        }
    }
}
