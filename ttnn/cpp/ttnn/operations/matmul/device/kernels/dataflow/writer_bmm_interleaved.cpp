// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// #include "debug/dprint.h"

void kernel_main() {
    // same arg indices as in reader_bmm_8bank for reuse
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t batch = get_arg_val<uint32_t>(7);

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr int onetile = 1;
    constexpr uint32_t cb_id_out0 = 16;
    const uint32_t tile_bytes = get_tile_size(cb_id_out0);
    uint32_t itileC = 0;
    const DataFormat data_format = get_dataformat(cb_id_out0);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    // C is MN so we iterate in tile RM order
    for (uint32_t nb = 0; nb < batch; nb++)
        for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C)        // output tile of C
            for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C) {  // output tile index of C
                // bmm will generate C's tiles C=A*B, MN=MK*KN, in row major order, we just read them from CB and write
                // out to DRAM
                cb_wait_front(cb_id_out0, onetile);
                uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
                noc_async_write_tile(itileC, s, l1_read_addr);
                noc_async_write_barrier();
                cb_pop_front(cb_id_out0, onetile);
                // DPRINT << 'W' << 'C' << itileC << ' ' << 'a' << dst_addr << ENDL();
                // DPRINT << itileC << ' ' << uint32_t(dst_noc_addr) << ENDL();
                itileC++;
            }
}
