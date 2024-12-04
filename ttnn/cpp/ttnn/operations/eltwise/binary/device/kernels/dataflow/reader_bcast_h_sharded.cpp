// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This code is temporarily copied from ttnn/cpp/ttnn/operations/datamovement/binary/device/ to demonstrate
// the new ability to keep the CircularBufferConfigs continuous during dispatching.  See the use of CBIndex::c_2 below.
// When broadcating is properly supported we expect this code to be deleted or refactored substantially.

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src1_addr = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    uint32_t offset = get_arg_val<uint32_t>(3);
    uint32_t NC = get_arg_val<uint32_t>(4);
    uint32_t batch_offset = get_arg_val<uint32_t>(5);  // if weight has multiple batches

    // constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);

    // constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t onetile = 1;

    // single-tile ublocks
    const uint32_t tile_bytes = get_tile_size(cb_id_in1);
    const DataFormat data_format = get_dataformat(cb_id_in1);

    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    uint32_t i = 0;
    cb_push_back(cb_id_in0, Ht * Wt);
    for (uint32_t ht = 0; ht < Ht; ht++) {
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // for each W-tile of the first tensor we push one tile from the second arg tile list
            // but we loop the second list around
            cb_reserve_back(cb_id_in1, onetile);
            l1_write_addr_in1 = get_write_ptr(cb_id_in1);
            noc_async_read_tile(offset, s1, l1_write_addr_in1);
            noc_async_read_barrier();
            cb_push_back(cb_id_in1, onetile);
            offset++;
        }

        // bcast tensor should be NC1W (actually NC32W padded with 0s in H)
        // wrap W around for each h (broadcast)
        offset -= Wt;
        if (ht % NC == (NC - 1)) {
            offset += batch_offset;  // switching to next batch
        }
    }
}
