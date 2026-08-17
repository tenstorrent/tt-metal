// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
// Trace-safe metadata: shared 1-element-tensor read (async_read -> barrier -> invalidate_l1_cache ->
// volatile load). The invalidate is load-bearing: the DRAM address is reused every chunk, so a cached L1
// line would silently serve the PREVIOUS chunk's value.
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);
    uint32_t num_chunks = get_arg_val<uint32_t>(3);
    uint32_t tail_chunk_bytes = get_arg_val<uint32_t>(4);
    const uint32_t input_page_bytes = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(3);
    constexpr auto input_args = TensorAccessorArgs<4>();
    // Metadata block appended after the input accessor. Guard the INDEX, not the value: a ternary does not
    // stop get_compile_time_arg_val from being evaluated, so an absent arg is a hard compile error.
    constexpr uint32_t meta_flag_arg = input_args.next_compile_time_args_offset();
    constexpr bool valid_length_from_metadata = get_compile_time_arg_val(meta_flag_arg) != 0;
    constexpr uint32_t meta_base = valid_length_from_metadata ? meta_flag_arg + 1 : 0;
    constexpr uint32_t meta_cb = get_compile_time_arg_val(meta_base + 0);
    constexpr uint32_t meta_llk_k = get_compile_time_arg_val(meta_base + 1);
    constexpr uint32_t meta_offset = get_compile_time_arg_val(meta_base + 2);
    constexpr uint32_t meta_elem_bytes = get_compile_time_arg_val(meta_base + 3);
    constexpr auto meta_args = TensorAccessorArgs<valid_length_from_metadata ? meta_flag_arg + 5 : 0>();

    const auto input = TensorAccessor(input_args, src_addr, input_page_bytes);
    CircularBuffer input_cb(cb_in);
    Noc noc;

    if constexpr (valid_length_from_metadata) {
        // Derive the search width on-device, then PUBLISH num_chunks/tail_elements to compute. Compute must
        // not re-derive: it pops exactly num_chunks pages while this loop pushes num_chunks * tiles_per_chunk,
        // so any disagreement between the two is a hang, not a wrong answer.
        CircularBuffer meta_cb_obj(meta_cb);
        meta_cb_obj.reserve_back(1);
        const uint32_t scratch = meta_cb_obj.get_write_ptr();
        const uint32_t search_len =
            trace_metadata::read_metadata_scalar_u32(noc, meta_args, get_arg_val<uint32_t>(6), scratch) + meta_offset;
        num_chunks = (search_len + meta_llk_k - 1) / meta_llk_k;
        const uint32_t tail_elements = search_len - ((num_chunks - 1) * meta_llk_k);
        tail_chunk_bytes = tail_elements * meta_elem_bytes;
        CoreLocalMem<volatile uint32_t> mailbox(scratch);
        mailbox[0] = num_chunks;
        mailbox[1] = tail_elements;
        meta_cb_obj.push_back(1);
    }

    for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
        const uint32_t row = start_row + local_row;
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            const uint32_t active_chunk_bytes = (chunk + 1 == num_chunks) ? tail_chunk_bytes : chunk_bytes;
            input_cb.reserve_back(tiles_per_chunk);
            for (uint32_t tile = 0; tile < tiles_per_chunk; ++tile) {
                const uint32_t tile_offset = tile * tile_bytes;
                const uint32_t read_bytes =
                    tile_offset < active_chunk_bytes
                        ? (active_chunk_bytes - tile_offset < tile_bytes ? active_chunk_bytes - tile_offset
                                                                         : tile_bytes)
                        : 0;
                if (read_bytes != 0) {
                    noc.async_read(
                        input,
                        input_cb,
                        read_bytes,
                        {.page_id = row, .offset_bytes = chunk * chunk_bytes + tile_offset},
                        {.offset_bytes = tile_offset});
                }
            }
            noc.async_read_barrier();
            input_cb.push_back(tiles_per_chunk);
        }
    }
}
