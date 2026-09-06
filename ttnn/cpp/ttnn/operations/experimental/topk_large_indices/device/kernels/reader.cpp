// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
// Shared metadata read invalidates the reused L1 address before loading the next value.
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"
#include "topk_large_indices_metadata.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);
    uint32_t search_len = get_arg_val<uint32_t>(3);
    const uint32_t input_page_bytes = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t element_bytes = sizeof(uint16_t);  // input is validated as BFLOAT16
    constexpr uint32_t llk_k = chunk_bytes / element_bytes;
    constexpr auto input_args = TensorAccessorArgs<4>();
    constexpr uint32_t metadata_args_base = input_args.next_compile_time_args_offset();
    constexpr bool valid_length_from_metadata = get_compile_time_arg_val(metadata_args_base) != 0;
    constexpr uint32_t meta_cb = get_compile_time_arg_val(metadata_args_base + 1);
    constexpr uint32_t meta_offset = get_compile_time_arg_val(metadata_args_base + 2);
    constexpr auto meta_args = TensorAccessorArgs<metadata_args_base + 3>();

    const auto input = TensorAccessor(input_args, src_addr, input_page_bytes);
    const uint32_t input_width = input_page_bytes / element_bytes;
    CircularBuffer input_cb(cb_in);
    Noc noc;

    TopkMetadataBounds bounds;
    if constexpr (valid_length_from_metadata) {
        CircularBuffer meta_cb_obj(meta_cb);
        meta_cb_obj.reserve_back(1);
        const uint32_t scratch = meta_cb_obj.get_write_ptr();
        const uint32_t metadata_length =
            trace_metadata::read_metadata_scalar_u32(noc, meta_args, get_arg_val<uint32_t>(5), scratch);
        // Validate before addition so the offset cannot wrap and no malformed metadata can produce a NoC read
        // outside the input row. Device ASSERT surfaces as a runtime failure, matching the scalar API's rejection.
        ASSERT(
            meta_offset <= input_width && metadata_length <= input_width - meta_offset &&
            (metadata_length != 0 || meta_offset != 0));
        search_len = metadata_length + meta_offset;
        bounds = calculate_topk_bounds(search_len, llk_k);
        CoreLocalMem<TopkMetadataBounds> mailbox(scratch);
        mailbox->num_chunks = bounds.num_chunks;
        mailbox->tail_elements = bounds.tail_elements;
        clobber_all_memory();
        meta_cb_obj.push_back(1);
    } else {
        bounds = calculate_topk_bounds(search_len, llk_k);
    }
    const uint32_t tail_chunk_bytes = bounds.tail_elements * element_bytes;

    for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
        const uint32_t row = start_row + local_row;
        for (uint32_t chunk = 0; chunk < bounds.num_chunks; ++chunk) {
            const uint32_t active_chunk_bytes = (chunk + 1 == bounds.num_chunks) ? tail_chunk_bytes : chunk_bytes;
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
