// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// batch_fft_writer.cpp — BRISC1 / writer for device-side BATCH FFT.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "batch_fft_common.h"

void kernel_main() {
    const uint32_t out_r_addr      = get_arg_val<uint32_t>(0);
    const uint32_t out_i_addr      = get_arg_val<uint32_t>(1);
    const uint32_t base_tile_idx   = get_arg_val<uint32_t>(2);
    const uint32_t batch_per_core  = get_arg_val<uint32_t>(3);

    const DataFormat df = get_dataformat(CB_STATE_R);
    const uint32_t   ts = get_tile_size(CB_STATE_R);

    InterleavedAddrGenFast<true> out_r_gen = {
        .bank_base_address = out_r_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> out_i_gen = {
        .bank_base_address = out_i_addr, .page_size = ts, .data_format = df};

    for (uint32_t k = 0; k < batch_per_core; ++k) {
        const uint32_t tile_idx = base_tile_idx + k;

        cb_wait_front(CB_SYNC, 1);
        cb_wait_front(CB_STATE_R, 1);
        cb_wait_front(CB_STATE_I, 1);

        noc_async_write_tile(tile_idx, out_r_gen, get_read_ptr(CB_STATE_R));
        noc_async_write_tile(tile_idx, out_i_gen, get_read_ptr(CB_STATE_I));
        noc_async_write_barrier();

        cb_pop_front(CB_SYNC, 1);
        cb_pop_front(CB_STATE_R, 1);
        cb_pop_front(CB_STATE_I, 1);
    }
}
