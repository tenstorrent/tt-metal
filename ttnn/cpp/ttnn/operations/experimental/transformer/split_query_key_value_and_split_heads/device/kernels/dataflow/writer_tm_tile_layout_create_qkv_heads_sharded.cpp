// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    constexpr uint32_t num_heads_per_tensor = get_arg(args::num_heads_per_tensor);  // 2
    constexpr uint32_t block_ht = get_arg(args::block_ht);                          // 12
    constexpr uint32_t block_wt = get_arg(args::block_wt);                          // 12
    constexpr uint32_t out_block_wt = get_arg(args::out_block_wt);                  // 2
    constexpr uint32_t block_wt_size_bytes = get_arg(args::block_wt_size_bytes);
    constexpr uint32_t out_block_wt_size_bytes = get_arg(args::out_block_wt_size_bytes);
    constexpr uint32_t num_tiles_per_tensor = get_arg(args::num_tiles_per_tensor);
    constexpr uint32_t tensor_stride_size_bytes = get_arg(args::tensor_stride_size_bytes);

    DataflowBuffer dfb_in0_obj(dfb::in);
    DataflowBuffer dfb_im0_obj(dfb::k_pre_transpose);

    const uint32_t single_tile_size_bytes = dfb_in0_obj.get_tile_size();

    const uint8_t noc_id = noc.get_noc_id();
    const uint32_t my_noc_x = my_x[noc_id];
    const uint32_t my_noc_y = my_y[noc_id];
    UnicastEndpoint src_ep;

    // re-order k
    uint32_t l1_read_addr = dfb_in0_obj.get_read_ptr();
    l1_read_addr += tensor_stride_size_bytes;
    uint32_t src_noc_addr_offset_outer_most = 0;
    for (uint32_t j = 0; j < num_heads_per_tensor; j++) {  // 2
        uint32_t src_noc_addr_offset_outer = 0;
        for (uint32_t k = 0; k < out_block_wt; k++) {  // 2
            uint32_t l1_read_addr_offset = 0;
            uint32_t l1_write_addr_out1 = dfb_im0_obj.get_write_ptr();
            dfb_im0_obj.reserve_back(block_ht);
            for (uint32_t i = 0; i < block_ht; i++) {  // 12
                const uint32_t src_l1_addr =
                    l1_read_addr + l1_read_addr_offset + src_noc_addr_offset_outer + src_noc_addr_offset_outer_most;
                noc.async_read(
                    src_ep,
                    CoreLocalMem<uint32_t>(l1_write_addr_out1),
                    single_tile_size_bytes,
                    {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = src_l1_addr},
                    {});
                l1_read_addr_offset += block_wt_size_bytes;
                l1_write_addr_out1 += single_tile_size_bytes;
            }
            noc.async_read_barrier();
            dfb_im0_obj.push_back(block_ht);
            src_noc_addr_offset_outer += single_tile_size_bytes;
        }
        src_noc_addr_offset_outer_most += out_block_wt_size_bytes;
    }
}
