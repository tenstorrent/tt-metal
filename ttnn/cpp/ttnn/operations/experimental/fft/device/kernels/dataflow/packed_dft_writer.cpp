// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// packed_dft_writer.cpp — BRISC1 / writer for the PACKED DIRECT-DFT kernel.
//
// For each output tile `t` in this core's slice, wait for the compute engine
// to push (CB_OUT_R, CB_OUT_I), flush both to their DRAM pages, and pop the
// CB slots so the pipeline can advance.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "packed_dft_common.h"

void kernel_main() {
    const uint32_t out_r_addr      = get_arg_val<uint32_t>(0);
    const uint32_t out_i_addr      = get_arg_val<uint32_t>(1);
    const uint32_t base_tile_idx   = get_arg_val<uint32_t>(2);
    const uint32_t tiles_per_core  = get_arg_val<uint32_t>(3);

    const DataFormat df = get_dataformat(CB_OUT_R);
    const uint32_t   ts = get_tile_size(CB_OUT_R);

    InterleavedAddrGenFast<true> out_r_gen = {
        .bank_base_address = out_r_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> out_i_gen = {
        .bank_base_address = out_i_addr, .page_size = ts, .data_format = df};

    for (uint32_t k = 0; k < tiles_per_core; ++k) {
        const uint32_t t = base_tile_idx + k;

        cb_wait_front(CB_OUT_R, 1);
        cb_wait_front(CB_OUT_I, 1);

        noc_async_write_tile(t, out_r_gen, get_read_ptr(CB_OUT_R));
        noc_async_write_tile(t, out_i_gen, get_read_ptr(CB_OUT_I));
        noc_async_write_barrier();

        cb_pop_front(CB_OUT_R, 1);
        cb_pop_front(CB_OUT_I, 1);
    }
}
