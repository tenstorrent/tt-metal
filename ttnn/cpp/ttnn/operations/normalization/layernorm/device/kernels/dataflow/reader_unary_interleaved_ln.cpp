// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t NCHt      = get_arg_val<uint32_t>(1);
    uint32_t Wt        = get_arg_val<uint32_t>(2);
    uint32_t tile_offset = get_arg_val<uint32_t>(3);

    uint32_t gamma_addr = get_arg_val<uint32_t>(6);
    uint32_t beta_addr = get_arg_val<uint32_t>(7);
    uint32_t b_addr = get_arg_val<uint32_t>(8);


    constexpr uint32_t cb_id_in0 = 0, cb_id_in1 = 1;
    constexpr uint32_t cb_id_gamma = 5;
    constexpr uint32_t cb_id_beta = 6;

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat src0_data_format = get_dataformat(cb_id_in0);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool gamma_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool beta_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t blk = get_compile_time_arg_val(4); // needed for correctness of softmax/LN kernels

    const InterleavedAddrGenFast<src0_is_dram> src_a = {
        .bank_base_address = src_addr,
        .page_size = src0_tile_bytes,
        .data_format = src0_data_format
    };

    #ifdef FUSE_GAMMA
    const uint32_t gamma_tile_bytes = get_tile_size(cb_id_gamma);
    const DataFormat gamma_data_format = get_dataformat(cb_id_gamma);
    const InterleavedAddrGenFast<gamma_is_dram> addrg = {
        .bank_base_address = gamma_addr,
        .page_size = gamma_tile_bytes,
        .data_format = gamma_data_format
    };
    #endif
    #ifdef FUSE_BETA
    const uint32_t beta_tile_bytes = get_tile_size(cb_id_beta);
    const DataFormat beta_data_format = get_dataformat(cb_id_beta);
    const InterleavedAddrGenFast<beta_is_dram> addrb = {
        .bank_base_address = beta_addr,
        .page_size = beta_tile_bytes,
        .data_format = beta_data_format
    };
    #endif
    #ifdef FUSE_PRE_ADD
    const uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    const DataFormat src1_data_format = get_dataformat(cb_id_in1);
    const InterleavedAddrGenFast<src1_is_dram> src_b = {
        .bank_base_address = b_addr,
        .page_size = src1_tile_bytes,
        .data_format = src1_data_format
    };
    #endif


    // Generate constant tiles for layernorm compute
    {
        constexpr uint32_t cb_in_2 = 2;
        uint32_t scaler = get_arg_val<uint32_t>(4);
        generate_reduce_scaler(cb_in_2, scaler);
    }
    constexpr uint32_t eps_cb_id = 3;
    const uint32_t eps = get_arg_val<uint32_t>(5);
    generate_bcast_col_scalar(eps_cb_id, eps);

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t offs = 0;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        for (uint32_t wt = 0; wt<Wt; wt += blk) {
            cb_reserve_back(cb_id_in0, blk);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

            for (uint32_t r = 0; r<blk; r++) {
                noc_async_read_tile(offs+wt+r+tile_offset, src_a, l1_write_addr);
                l1_write_addr += src0_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, blk);

            #ifdef FUSE_PRE_ADD
            // TODO(AP): refactor the ifdefs
            cb_reserve_back(cb_id_in1, blk);
            l1_write_addr = get_write_ptr(cb_id_in1);
            for (uint32_t r = 0; r<blk; r++) {
                noc_async_read_tile(offs+wt+r+tile_offset, src_b, l1_write_addr);
                l1_write_addr += src1_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_in1, blk);
            #endif
        } // wt loop

        #if defined FUSE_GAMMA || defined FUSE_BETA
        if (ncht == 0) {
            for (uint32_t wt = 0; wt<Wt; wt += blk) {
                #ifdef FUSE_GAMMA
                {
                    cb_reserve_back(cb_id_gamma, blk);
                    uint32_t l1_write_addr = get_write_ptr(cb_id_gamma);
                    for (uint32_t r = 0; r<blk; r++) {
                        noc_async_read_tile(wt + r, addrg, l1_write_addr);
                        l1_write_addr += gamma_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_id_gamma, blk);
                }
                #endif

                #ifdef FUSE_BETA
                {
                    cb_reserve_back(cb_id_beta, blk);
                    uint32_t l1_write_addr = get_write_ptr(cb_id_beta);
                    for (uint32_t r = 0; r<blk; r++) {
                        noc_async_read_tile(wt + r, addrb, l1_write_addr);
                        l1_write_addr += beta_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_id_beta, blk);
                }
                #endif
            } // wt loop
        }
        #endif
        offs += Wt;
    } // ncht loop
}
