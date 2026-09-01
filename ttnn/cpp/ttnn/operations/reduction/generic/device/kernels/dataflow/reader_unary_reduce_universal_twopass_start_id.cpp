// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    std::uint32_t num_tiles = get_arg(args::num_tiles);
    std::uint32_t start_id = get_arg(args::start_id);

    constexpr std::uint32_t Wt = get_arg(args::Wt);

    constexpr std::uint32_t max_read_batch = WELFORD_TWO_PASS_STREAMING_CB_TILES;
    DataflowBuffer dfb_in0(dfb::in0);
    const std::uint32_t tile_bytes = dfb_in0.get_tile_size();

    auto tensor_accessor = TensorAccessor(tensor::src);
    Noc noc;

#ifndef WELFORD_TWO_PASS_L1_REPLAY
    std::uint32_t stream_write_page = 0;
#endif

    const std::uint32_t num_rows = num_tiles / Wt;
    for (std::uint32_t row = 0; row < num_rows; ++row) {
        const std::uint32_t row_start_id = start_id + row * Wt;
#ifdef WELFORD_TWO_PASS_L1_REPLAY
        // The compute kernel keeps this complete row in the enlarged input CB and
        // indexes it twice, so DRAM is traversed only once.
        for (std::uint32_t wt_base = 0; wt_base < Wt; wt_base += max_read_batch) {
            const std::uint32_t read_batch = std::min(max_read_batch, Wt - wt_base);
            dfb_in0.reserve_back(read_batch);
            for (std::uint32_t wt = 0; wt < read_batch; ++wt) {
                noc.async_read(
                    tensor_accessor,
                    dfb_in0,
                    tile_bytes,
                    {.page_id = row_start_id + wt_base + wt},
                    {.offset_bytes = wt * tile_bytes});
            }
            noc.async_read_barrier();
            dfb_in0.push_back(read_batch);
        }
#else
#ifdef WELFORD_TWO_PASS_BFP8_INPUT
        constexpr std::uint32_t num_front_retained = 2;
#else
        constexpr std::uint32_t num_front_retained = 3;
#endif
        constexpr std::uint32_t num_passes = Wt <= num_front_retained + 1 ? 1 : 2;
        for (std::uint32_t pass = 0; pass < num_passes; ++pass) {
            // Compute retains the first two or three transposed tiles and the final
            // tile in DEST across passes, so only stream the middle tiles on
            // pass two. Tile order remains unchanged.
            const std::uint32_t pass_start = pass == 0 ? 0 : std::min(Wt, num_front_retained);
            const std::uint32_t pass_end = pass == 0 ? Wt : Wt - 1;
            for (std::uint32_t wt_base = pass_start; wt_base < pass_end;) {
                const std::uint32_t contiguous_pages = max_read_batch - stream_write_page;
                const std::uint32_t read_batch =
                    std::min(std::min(max_read_batch, pass_end - wt_base), contiguous_pages);
                dfb_in0.reserve_back(read_batch);
                for (std::uint32_t wt = 0; wt < read_batch; ++wt) {
                    noc.async_read(
                        tensor_accessor,
                        dfb_in0,
                        tile_bytes,
                        {.page_id = row_start_id + wt_base + wt},
                        {.offset_bytes = wt * tile_bytes});
                }
                noc.async_read_barrier();
                dfb_in0.push_back(read_batch);
                wt_base += read_batch;
                stream_write_page = (stream_write_page + read_batch) % max_read_batch;
            }
        }
#endif
    }
}
