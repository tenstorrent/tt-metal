// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "named_args.hpp"

#include "reduce_scaler.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t tile_bytes = CT_ARG("tile_bytes_bf16");
    constexpr uint32_t row_tiles = CT_ARG("h_tiles");
    constexpr uint32_t chunk_tiles = CT_ARG("norm_chunk_tiles");
    constexpr uint32_t chunks_per_row = row_tiles / chunk_tiles;
    constexpr uint32_t ring_chunks = CT_ARG("norm_in_chunks");
    constexpr uint32_t in_flight = CT_ARG("norm_reads_in_flight");
    const auto h = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_h")>(), RT_ARG("h_addr"), tile_bytes);
    const auto w_norm = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_w_norm")>(), RT_ARG("w_norm_addr"), tile_bytes);
    CircularBuffer in(CT_ARG("cb_norm_in"));
    const uint32_t ring = in.get_write_ptr();

    fill_reduce_scaler(noc, CT_ARG("cb_norm_scaler"), tile_bytes);
    fill_ones(noc, CT_ARG("cb_norm_ones"), tile_bytes);

    uint32_t streamed = 0;
    auto stream = [&](const auto& source, uint32_t chunks, auto first_page) {
        for (uint32_t chunk = 0; chunk < chunks + in_flight - 1; chunk++) {
            if (chunk < chunks) {
                in.reserve_back((chunk < in_flight ? chunk + 1 : in_flight) * chunk_tiles);
                const uint32_t destination = ring + ((streamed + chunk) % ring_chunks) * chunk_tiles * tile_bytes;
                for (uint32_t t = 0; t < chunk_tiles; t++) {
                    noc.async_read<NocOptions::TXN_ID>(
                        source,
                        CoreLocalMem<uint32_t>(destination + t * tile_bytes),
                        tile_bytes,
                        {.page_id = first_page(chunk) + t},
                        {},
                        {.trid = 1 + chunk % in_flight});
                }
            }
            if (chunk + 1 >= in_flight) {
                noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 1 + (chunk + 1 - in_flight) % in_flight});
                in.push_back(chunk_tiles);
            }
        }
        streamed += chunks;
    };

    stream(w_norm, chunks_per_row, [](uint32_t chunk) { return chunk * chunk_tiles; });
    stream(h, CT_ARG("rows") * chunks_per_row, [](uint32_t chunk) {
        return (CT_ARG("group") + chunk / chunks_per_row * CT_ARG("groups")) * row_tiles +
               chunk % chunks_per_row * chunk_tiles;
    });
}
