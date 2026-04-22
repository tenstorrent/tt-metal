// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for register-based argmax over a non-HW dim (NC-style).
// Writes UINT32 output tiles produced by the compute kernel from cb_out0 to
// the interleaved output buffer (DRAM or L1).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

template <bool is_dram>
inline void write_output_loop(const uint32_t output_addr, const uint32_t num_output_tiles, const uint32_t start_id) {
    constexpr uint32_t cb_out0 = 16;
    constexpr uint32_t onetile = 1;

    const uint32_t output_tile_bytes = get_tile_size(cb_out0);
    const auto output_dataformat = get_dataformat(cb_out0);

    const InterleavedAddrGenFast<is_dram> s0 = {
        .bank_base_address = output_addr,
        .page_size = output_tile_bytes,
        .data_format = output_dataformat,
    };

    for (uint32_t out_i = 0; out_i < num_output_tiles; ++out_i) {
        const uint32_t write_tile_id = start_id + out_i;
        cb_wait_front(cb_out0, onetile);
        const uint32_t read_ptr = get_read_ptr(cb_out0);
        noc_async_write_tile(write_tile_id, s0, read_ptr);
        noc_async_write_barrier();
        cb_pop_front(cb_out0, onetile);
    }
}

void kernel_main() {
    // Runtime args
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    const bool output_in_dram = get_arg_val<uint32_t>(3) != 0;

    if (output_in_dram) {
        write_output_loop<true>(output_addr, num_output_tiles, start_id);
    } else {
        write_output_loop<false>(output_addr, num_output_tiles, start_id);
    }
}
