// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t blk = get_arg_val<uint32_t>(1);
    const uint32_t NCht =
        get_arg_val<uint32_t>(3);  // same arg index as in reader_unary and in reader_unary_transpose_wh_8bank
    const uint32_t tile_offset = get_arg_val<uint32_t>(4);
    const uint32_t Wt = get_arg_val<uint32_t>(5);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0, cb_id_in1 = tt::CBIndex::c_1;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat src0_data_format = get_dataformat(cb_id_in0);

#if FUSED_SCALE_MASK
    uint32_t Ht = get_arg_val<uint32_t>(6);
    uint32_t mask_addr = get_arg_val<uint32_t>(7);
    uint32_t start_ht = get_arg_val<uint32_t>(8);
    uint32_t start_mask_id = get_arg_val<uint32_t>(9);
    constexpr bool mask_is_dram = get_compile_time_arg_val(1) == 1;

    constexpr uint32_t cb_id_attn = 4;
    uint32_t mask_tile_bytes = get_tile_size(cb_id_attn);
    const DataFormat mask_data_format = get_dataformat(cb_id_attn);

    const InterleavedAddrGenFast<mask_is_dram> addr_mask = {
        .bank_base_address = mask_addr, .page_size = mask_tile_bytes, .data_format = mask_data_format};

#if CAUSAL_MASK
    constexpr uint32_t num_tiles_causal_mask = get_compile_time_arg_val(2);
    uint32_t mask_start_ht = get_arg_val<uint32_t>(11);
    uint32_t mask_offset = get_arg_val<uint32_t>(12);

    uint32_t mask_id_offset = mask_offset;
    uint32_t mask_ht = mask_start_ht;
#endif

    uint32_t ht = start_ht;
    uint32_t mask_id = start_mask_id;
    bool read_mask = true;
    constexpr auto cb_fused_scale = tt::CBIndex::c_3;
    const uint32_t pre_scale = get_arg_val<uint32_t>(2);
    generate_bcast_unary_scalar(cb_fused_scale, pre_scale);
#endif

    const InterleavedAddrGenFast<src0_is_dram> src_a = {
        .bank_base_address = src_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};

    {
        constexpr uint32_t cb_in_2 = tt::CBIndex::c_2;
        const uint32_t reduce_scaler = get_arg_val<uint32_t>(10);
        generate_reduce_scaler(cb_in_2, reduce_scaler);
    }

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t tile_index = tile_offset;
    constexpr uint33_t total_passes = 2;
    for (uint32_t ncht = 0; ncht < NCht; ncht++;) {
        // We need to pass once in order to calcualte the sum and then to calculate the final value.
        for (uint32_t cur_pass = 0; cur_pass < total_passes; cur_pass++) {
            // We want to fill up the CB for input, and do so in chunks of blk
            for (uint32_t wt = 0; wt < Wt; wt += cb_length) {
                // We read in the cb_length amount by the number of destination registers
                for (uint32_t blk_i = 0; i < cb_length; blk_i += blk) {
                    cb_reserve_back(cb_id_in0, blk);
                    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
                    for (uint32_t regs = 0; regs < blk; blk++) {
                        noc_async_read_tile(tile_index, src_a, l1_write_addr);  // TODO(AP): data type size
                        tile_index++;
                        l1_write_addr += src0_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_id_in0, blk);
                }
            }
        }
    }
}
