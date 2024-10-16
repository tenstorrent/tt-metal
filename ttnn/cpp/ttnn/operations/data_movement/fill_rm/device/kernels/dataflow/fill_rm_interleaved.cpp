// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include "dataflow_api.h"

void kernel_main() {
    // Kernel args
    // This kernel accepts a RM row-interleaved tensor laid out as NC,H,(Wt*32)-RM
    // H should be < 32 at the moment
    // It will write out a tensor NC,32,Wt*32

    // Note: this kernel is written with maximum simplicity in mind and (deliberately) doesn't pursue performance

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t NC = get_arg_val<uint32_t>(1);
    uint32_t H = get_arg_val<uint32_t>(2);
    uint32_t W = get_arg_val<uint32_t>(3);
    uint32_t fillH = get_arg_val<uint32_t>(4);
    uint32_t fillW = get_arg_val<uint32_t>(5);
    uint32_t val_hi = get_arg_val<uint32_t>(6);
    uint32_t val_lo = get_arg_val<uint32_t>(7);

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    const InterleavedAddrGen<dst_is_dram> s0 = {.bank_base_address = dst_addr, .page_size = W << 1};

    // DPRINT << "fill_rm_8bank: NC=" << NC << " H=" << H << " W=" << W << " fillH=" << fillH << " fillW=" << fillW <<
    // ENDL();
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    // How many bytes along a row in the original tensor
    uint32_t num_bytes_per_tile = get_tile_size(cb_id_in0);
    uint32_t num_bytes_per_tile_row = 64;
    uint32_t Wt = (W >> 5);

    // Variables
    uint64_t replicate_dest_addr;
    uint32_t start_dram_addr_offset_for_tensor_row = 0;

    cb_reserve_back(cb_id_in0, 16);  // in this kernel we are not pushing anything into CBs, just using the space
    cb_reserve_back(cb_id_in1, 16);
    uint32_t l1_w_addr = get_write_ptr(cb_id_in0);
    uint32_t l1_zeros_addr = get_write_ptr(cb_id_in1);
    uint32_t w;
    for (w = 0; w < fillW; w++) reinterpret_cast<uint16_t *>(l1_w_addr)[w] = val_hi;
    for (w = fillW; w < W; w++) reinterpret_cast<uint16_t *>(l1_w_addr)[w] = val_lo;
    for (w = 0; w < W; w++) reinterpret_cast<uint16_t *>(l1_zeros_addr)[w] = val_lo;

    uint32_t nch_dst = 0;
    // input is NCH(Wt*32) unpadded RM
    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t h = 0; h < H; h++) {
            uint64_t dst_noc_addr = get_noc_addr(nch_dst, s0);
            if (h < fillH) {
                noc_async_write(l1_w_addr, dst_noc_addr, (W << 1));  // TODO(AP): segment this write
            } else {
                noc_async_write(l1_zeros_addr, dst_noc_addr, (W << 1));  // TODO(AP): segment this write
            }
            noc_async_write_barrier();
            nch_dst++;
        }  // h<paddedH
    }  // nc
}
