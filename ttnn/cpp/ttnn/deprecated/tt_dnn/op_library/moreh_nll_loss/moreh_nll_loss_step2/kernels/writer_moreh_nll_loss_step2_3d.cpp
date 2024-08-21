// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    uint32_t i = 0;
    auto output_addr = get_arg_val<uint32_t>(i++);
    auto num_tiles_per_core = get_arg_val<uint32_t>(i++);
    auto start_id = get_arg_val<uint32_t>(i++);
    auto W = get_arg_val<uint32_t>(i++);
    auto element_size = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_output = tt::CB::c_out0;

    constexpr bool output_is_dram = get_compile_time_arg_val(0) == 1;

    const uint32_t output_tile_bytes = get_tile_size(cb_output);

    const InterleavedAddrGen<output_is_dram> output_addrg = {
        .bank_base_address = output_addr,
        .page_size = output_tile_bytes,
    };

    uint32_t Wf = (W + FACE_WIDTH - 1) / FACE_WIDTH;
    uint32_t Wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    constexpr uint32_t onetile = 1;
    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        // output: (N, d1)
        // noc_id = nt * Wt + wt
        cb_wait_front(cb_output, onetile);
        uint32_t n = i / Wf;
        uint32_t w = (i % Wf) * FACE_WIDTH;
        uint32_t nt = n / TILE_HEIGHT;
        uint32_t wt = w / TILE_WIDTH;

        uint32_t output_l1_write_addr = get_read_ptr(cb_output);

        uint32_t noc_id = nt * Wt + wt;
        uint32_t noc_offset;
        get_noc_offset(n, w, element_size, noc_offset);

        uint64_t dst_noc_addr = get_noc_addr(noc_id, output_addrg, noc_offset);
        noc_async_write(output_l1_write_addr, dst_noc_addr, NOC_MINIMUM_READ_SIZE);
        noc_async_write_barrier();

        cb_pop_front(cb_output, onetile);
    }
}
