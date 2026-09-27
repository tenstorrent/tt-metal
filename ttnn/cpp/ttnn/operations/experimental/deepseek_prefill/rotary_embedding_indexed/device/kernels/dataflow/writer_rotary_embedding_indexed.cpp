// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Compute produces only the rotary region. Copy the other complete tiles directly
// from input to output, without additional graph operations or dtype conversion.
void kernel_main() {
    Noc noc;
    const auto batch_start = get_arg(args::batch_start);
    const auto batch_end = get_arg(args::batch_end);
    const auto seq_t_start = get_arg(args::seq_t_start);
    const auto seq_t_end = get_arg(args::seq_t_end);
    const auto head_start = get_arg(args::head_start);
    const auto head_end = get_arg(args::head_end);
    constexpr auto n_heads = get_arg(args::n_heads);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto input_Wt = get_arg(args::input_Wt);
    constexpr auto rotary_offset_t = get_arg(args::rotary_offset_t);
    constexpr auto prefix_Wt = get_arg(args::prefix_Wt);
    constexpr auto output_Wt = get_arg(args::output_Wt);
    constexpr auto Ht = get_arg(args::Ht);

    const auto input = TensorAccessor(tensor::input);
    const auto output = TensorAccessor(tensor::output);
    const auto concat_prefix = TensorAccessor(tensor::concat_prefix);
    DataflowBuffer rotated(dfb::out);
    DataflowBuffer copy(dfb::copy);
    const uint32_t tile_bytes = rotated.get_entry_size();

    for (uint32_t batch = batch_start; batch < batch_end; ++batch) {
        for (uint32_t head = head_start; head < head_end; ++head) {
            for (uint32_t row = seq_t_start; row < seq_t_end; ++row) {
                const uint32_t row_index = (batch * n_heads + head) * Ht + row;
                const uint32_t input_row_page = row_index * input_Wt;
                const uint32_t output_row_page = row_index * output_Wt;
                constexpr uint32_t copy_tiles = prefix_Wt + input_Wt - Wt;
                if constexpr (copy_tiles > 0) {
                    // Fetch all passthrough tiles together while compute produces the rotary tiles.
                    copy.reserve_back(copy_tiles);
                    uint32_t destination = copy.get_write_ptr();
                    for (uint32_t col = 0; col < prefix_Wt; ++col) {
                        noc.async_read(
                            concat_prefix,
                            CoreLocalMem<uint32_t>(destination),
                            tile_bytes,
                            {.page_id = row_index * prefix_Wt + col},
                            {});
                        destination += tile_bytes;
                    }
                    for (uint32_t col = 0; col < input_Wt; ++col) {
                        if (col >= rotary_offset_t && col < rotary_offset_t + Wt) {
                            continue;
                        }
                        noc.async_read(
                            input,
                            CoreLocalMem<uint32_t>(destination),
                            tile_bytes,
                            {.page_id = input_row_page + col},
                            {});
                        destination += tile_bytes;
                    }
                }
                uint32_t source;
                if constexpr (prefix_Wt == 0) {
                    rotated.wait_front(Wt);
                    source = rotated.get_read_ptr();
                    for (uint32_t col = 0; col < Wt; ++col) {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(source),
                            output,
                            tile_bytes,
                            {},
                            {.page_id = output_row_page + rotary_offset_t + col});
                        source += tile_bytes;
                    }
                }
                if constexpr (copy_tiles > 0) {
                    noc.async_read_barrier();
                    copy.push_back(copy_tiles);
                    copy.wait_front(copy_tiles);
                    source = copy.get_read_ptr();
                    for (uint32_t col = 0; col < prefix_Wt; ++col) {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(source), output, tile_bytes, {}, {.page_id = output_row_page + col});
                        source += tile_bytes;
                    }
                    for (uint32_t col = 0; col < input_Wt; ++col) {
                        if (col >= rotary_offset_t && col < rotary_offset_t + Wt) {
                            continue;
                        }
                        noc.async_write(
                            CoreLocalMem<uint32_t>(source),
                            output,
                            tile_bytes,
                            {},
                            {.page_id = output_row_page + prefix_Wt + col});
                        source += tile_bytes;
                    }
                }
                if constexpr (prefix_Wt > 0) {
                    // Start writing prefix tiles while the compute kernel finishes rotation.
                    rotated.wait_front(Wt);
                    source = rotated.get_read_ptr();
                    for (uint32_t col = 0; col < Wt; ++col) {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(source),
                            output,
                            tile_bytes,
                            {},
                            {.page_id = output_row_page + prefix_Wt + rotary_offset_t + col});
                        source += tile_bytes;
                    }
                }
                noc.async_write_barrier();
                rotated.pop_front(Wt);
                if constexpr (copy_tiles > 0) {
                    copy.pop_front(copy_tiles);
                }
            }
        }
    }
}
