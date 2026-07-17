// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize reader (NCRISC / NoC0).
//
// Streams ROW_MAJOR sticks out of the source tensor into cb_rm_in at TILE
// granularity: for each width-chunk it batches 32 sticks (= one tile-row) and
// pushes Wt_chunk tile-sized pages per tile-row (read_sticks_for_tilize TILE
// mode). Wide W is chunked to keep the CB footprint bounded by a constant:
// each chunk reads `chunk_bytes` of every stick starting at
// `chunk * chunk_bytes`.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_rm_in = 0;
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(0);  // Wt_chunk * 32 * elem_size
    constexpr uint32_t num_chunks = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_stick = get_arg_val<uint32_t>(1);     // per-core stick offset
    const uint32_t total_num_rows = get_arg_val<uint32_t>(2);  // per-core stick count

    const auto accessor = TensorAccessor(src_args, src_addr);

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        dataflow_kernel_lib::read_sticks_for_tilize<cb_rm_in, dataflow_kernel_lib::TilizeGranularity::TILE>(
            accessor, total_num_rows, chunk_bytes, start_stick, chunk * chunk_bytes);
    }
}
