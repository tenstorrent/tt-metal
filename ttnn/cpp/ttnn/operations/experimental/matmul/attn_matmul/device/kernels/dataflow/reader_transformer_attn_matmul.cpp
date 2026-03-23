// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include "experimental/endpoints.h"
#include "experimental/core_local_mem.h"

void kernel_main() {
    // same arg indices as in reader_binary_diff_lengths for compat
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t MtKt = get_arg_val<uint32_t>(5);           // if 0
    uint32_t in1_KtNt_skip = get_arg_val<uint32_t>(6);  // 0 if in0 and in1 Kt are the same
    uint32_t in1_KtNt_mul_32 = get_arg_val<uint32_t>(7);
    uint32_t blocks = get_arg_val<uint32_t>(8);
    uint32_t itileA_start = get_arg_val<uint32_t>(9);
    uint32_t itileB_start = get_arg_val<uint32_t>(10);

    constexpr bool transpose_hw_bool = get_compile_time_arg_val(0) == 1;
    constexpr bool fp32_acc_en = get_compile_time_arg_val(1) == 1;

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_intermed0 = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_intermed1 = tt::CBIndex::c_3;
    constexpr uint32_t cb_id_intermed2 = tt::CBIndex::c_4;

    experimental::Noc noc;
    experimental::CircularBuffer cb_in0_obj(cb_id_in0);
    experimental::CircularBuffer cb_in1_obj(cb_id_in1);
    experimental::CircularBuffer cb_intermed1_obj(cb_id_intermed1);
    experimental::CircularBuffer cb_intermed2_obj(cb_id_intermed2);

    constexpr uint32_t onetile = 1;
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);

    constexpr auto in0_args = TensorAccessorArgs<2>();
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto s0 = TensorAccessor(in0_args, src0_addr, in0_tile_bytes);
    const auto s1 = TensorAccessor(in1_args, src1_addr, in1_tile_bytes);

    uint32_t itileA_batch = itileA_start;
    uint32_t itileB_batch;
    uint32_t itileA_Mt;
    uint32_t itileB_Nt;
    uint32_t itileA;
    uint32_t itileB;

    uint32_t cb_intermed1_addr_initial = cb_intermed1_obj.get_read_ptr();
    uint32_t cb_intermed2_addr_initial = cb_intermed2_obj.get_write_ptr();
    uint32_t cb_intermed1_addr;
    uint32_t cb_intermed2_addr;
    constexpr uint32_t bfloat16_row_bytes = fp32_acc_en ? 128 : 64;
    constexpr uint32_t num_rows_in_one_tile = 32;

    uint32_t local_noc_x = my_x[noc.get_noc_id()];
    uint32_t local_noc_y = my_y[noc.get_noc_id()];
    experimental::UnicastEndpoint local_src;

    for (uint32_t b = 0; b < blocks; b++) {
        itileA_Mt = itileA_batch;
        itileB_batch = itileB_start;

        for (uint32_t m = 0; m < Mt; m++) {
            itileB_Nt = itileB_batch;

            for (uint32_t n = 0; n < Nt; n++) {
                cb_intermed1_addr = cb_intermed1_addr_initial;
                cb_intermed2_addr = cb_intermed2_addr_initial;
                itileA = itileA_Mt;
                itileB = itileB_Nt;

                cb_in0_obj.reserve_back(Kt);
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    // Read A's tile at (mt, kt)
                    noc.async_read(s0, cb_in0_obj, in0_tile_bytes, {.page_id = itileA}, {.offset_bytes = 0});
                    noc.async_read_barrier();
                    cb_in0_obj.push_back(onetile);

                    itileA++;  // A is MK
                }

                cb_intermed2_obj.reserve_back(1);
                for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; tile_row_id++) {
                    for (uint32_t kt = 0; kt < Kt; kt++) {
                        // Read B's tile at (kt, nt)
                        cb_in1_obj.reserve_back(onetile);
                        noc.async_read(s1, cb_in1_obj, in1_tile_bytes, {.page_id = itileB}, {.offset_bytes = 0});
                        noc.async_read_barrier();
                        cb_in1_obj.push_back(onetile);

                        if constexpr (transpose_hw_bool) {
                            itileB++;  // Kt is in B[3], so it is contiguous in memory
                        } else {
                            itileB += Nt;  // Kt is in B[2], so stride is Nt
                        }
                    }  // Kt loop

                    // Read 32 untilized tiles and select correct rows to reconstruct single correct tile
                    cb_intermed1_obj.wait_front(1);
                    experimental::CoreLocalMem<uint32_t> local_dst(cb_intermed2_addr);
                    noc.async_read(
                        local_src,
                        local_dst,
                        bfloat16_row_bytes,
                        {.noc_x = local_noc_x, .noc_y = local_noc_y, .addr = cb_intermed1_addr},
                        {});
                    noc.async_read_barrier();
                    cb_intermed1_obj.pop_front(1);
                    cb_intermed1_addr += bfloat16_row_bytes;
                    cb_intermed2_addr += bfloat16_row_bytes;

                    itileB += in1_KtNt_skip;  // different depending on transpose_hw
                }  // 32 tiles loop
                cb_intermed2_obj.push_back(1);

                // Next tile in Nt
                if constexpr (transpose_hw_bool) {
                    itileB_Nt += Kt;  // next tile in Nt is in B[2], so stride is Kt
                } else {
                    itileB_Nt++;
                }
            }  // Nt loop

            itileA_Mt += Kt;
            // here, KtNt is the stride of the full B tensor (ie. max cache length is incorporated in one of Kt or Nt
            // depending on transpose_hw)
            itileB_batch += in1_KtNt_mul_32;  // different depending on transpose_hw
        }  // Mt loop

        itileA_batch += MtKt;
    }  // B loop
}
