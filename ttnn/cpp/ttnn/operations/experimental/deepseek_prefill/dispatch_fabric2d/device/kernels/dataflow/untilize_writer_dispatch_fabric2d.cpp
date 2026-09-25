// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Untilizer pool, writer RISC. Puts each untilized tile row into the staging buffer the stream cores
// read tokens from, then tells every stream core that one more tile row has landed.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "dispatch_fabric2d_untilize_args.hpp"

namespace {
using args = dspf2d::UntilizeCtArgs;

constexpr uint32_t cb_out = get_compile_time_arg_val(args::kRowCb);
constexpr uint32_t num_tile_rows = get_compile_time_arg_val(args::kNumTileRows);
constexpr uint32_t pool_size = get_compile_time_arg_val(args::kPoolSize);
constexpr uint32_t token_bytes = get_compile_time_arg_val(args::kTokenBytes);
constexpr uint32_t rows_per_tile_row = get_compile_time_arg_val(args::kRowsPerTileRow);
constexpr uint32_t stream_count = get_compile_time_arg_val(args::kStreamCount);
constexpr uint32_t sem_addr = get_compile_time_arg_val(args::kUntilizeSemAddr);

constexpr uint32_t stream_coords_base = get_compile_time_arg_val(args::kStreamCoordsBase);
constexpr uint32_t accessor_base = stream_coords_base + 2u * stream_count;
constexpr auto staging_args = TensorAccessorArgs<accessor_base>();

}  // namespace

void kernel_main() {
    const uint32_t first_tile_row = get_arg_val<uint32_t>(dspf2d::UntilizeRtArg::kFirstTileRow);
    const auto staging_acc =
        TensorAccessor(staging_args, get_arg_val<uint32_t>(dspf2d::UntilizeRtArg::kBufferAddr), token_bytes);

    for (uint32_t s = first_tile_row; s < num_tile_rows; s += pool_size) {
        cb_wait_front(cb_out, rows_per_tile_row);
        const uint32_t read_ptr = get_read_ptr(cb_out);
        const uint32_t first_page = s * rows_per_tile_row;
        for (uint32_t r = 0; r < rows_per_tile_row; r++) {
            noc_async_write(read_ptr + r * token_bytes, staging_acc.get_noc_addr(first_page + r), token_bytes);
        }
        // Stream cores read these pages as soon as the counter below is signalled, so the writes must be
        // acknowledged first; a flush would only show the CB can be reused.
        noc_async_write_barrier();
        cb_pop_front(cb_out, rows_per_tile_row);
        for (uint32_t i = 0; i < stream_count; i++) {
            const uint32_t x = kernel_compile_time_args[stream_coords_base + 2u * i];
            const uint32_t y = kernel_compile_time_args[stream_coords_base + 2u * i + 1u];
            noc_semaphore_inc(get_noc_addr(x, y, sem_addr), 1);
        }
    }
    // Wait for the increments to complete before the kernel exits, or the stream cores may never see them.
    noc_async_atomic_barrier();
}
