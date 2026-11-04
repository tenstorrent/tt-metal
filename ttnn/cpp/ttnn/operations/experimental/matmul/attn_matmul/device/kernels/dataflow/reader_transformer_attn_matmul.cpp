// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // same arg set (and order) as in reader_binary_diff_lengths for compat
    uint32_t Mt = get_arg(args::Mt);
    uint32_t Kt = get_arg(args::Kt);
    uint32_t Nt = get_arg(args::Nt);
    uint32_t MtKt = get_arg(args::MtKt);                    // if 0
    uint32_t in1_KtNt_skip = get_arg(args::in1_KtNt_skip);  // 0 if in0 and in1 Kt are the same
    uint32_t in1_KtNt_mul_32 = get_arg(args::in1_KtNt_mul_32);
    uint32_t blocks = get_arg(args::blocks);
    uint32_t itileA_start = get_arg(args::itileA_start);
    uint32_t itileB_start = get_arg(args::itileB_start);

    constexpr bool transpose_hw_bool = get_arg(args::transpose_hw) == 1;
    constexpr bool fp32_acc_en = get_arg(args::fp32_acc_en) == 1;

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_intermed1(dfb::intermed1);
    DataflowBuffer dfb_intermed2(dfb::intermed2);

    constexpr uint32_t onetile = 1;
    const uint32_t in0_tile_bytes = dfb_in0.get_tile_size();
    const uint32_t in1_tile_bytes = dfb_in1.get_tile_size();

    const auto s0 = TensorAccessor(tensor::src0);
    const auto s1 = TensorAccessor(tensor::src1);

    uint32_t itileA_batch = itileA_start;
    uint32_t itileB_batch;
    uint32_t itileA_Mt;
    uint32_t itileB_Nt;
    uint32_t itileA;
    uint32_t itileB;

    uint32_t dfb_intermed1_addr_initial = dfb_intermed1.get_read_ptr();
    uint32_t dfb_intermed2_addr_initial = dfb_intermed2.get_write_ptr();
    uint32_t dfb_intermed1_addr;
    uint32_t dfb_intermed2_addr;
    constexpr uint32_t bfloat16_row_bytes = fp32_acc_en ? 128 : 64;
    constexpr uint32_t num_rows_in_one_tile = 32;

    uint32_t local_noc_x = my_x[noc.get_noc_id()];
    uint32_t local_noc_y = my_y[noc.get_noc_id()];
    UnicastEndpoint local_src;

    for (uint32_t b = 0; b < blocks; b++) {
        itileA_Mt = itileA_batch;
        itileB_batch = itileB_start;

        for (uint32_t m = 0; m < Mt; m++) {
            itileB_Nt = itileB_batch;

            for (uint32_t n = 0; n < Nt; n++) {
                dfb_intermed1_addr = dfb_intermed1_addr_initial;
                dfb_intermed2_addr = dfb_intermed2_addr_initial;
                itileA = itileA_Mt;
                itileB = itileB_Nt;

                dfb_in0.reserve_back(Kt);
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    // Read A's tile at (mt, kt)
                    noc.async_read(s0, dfb_in0, in0_tile_bytes, {.page_id = itileA}, {.offset_bytes = 0});
                    noc.async_read_barrier();
                    dfb_in0.push_back(onetile);

                    itileA++;  // A is MK
                }

                dfb_intermed2.reserve_back(1);
                for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; tile_row_id++) {
                    for (uint32_t kt = 0; kt < Kt; kt++) {
                        // Read B's tile at (kt, nt)
                        dfb_in1.reserve_back(onetile);
                        noc.async_read(s1, dfb_in1, in1_tile_bytes, {.page_id = itileB}, {.offset_bytes = 0});
                        noc.async_read_barrier();
                        dfb_in1.push_back(onetile);

                        if constexpr (transpose_hw_bool) {
                            itileB++;  // Kt is in B[3], so it is contiguous in memory
                        } else {
                            itileB += Nt;  // Kt is in B[2], so stride is Nt
                        }
                    }  // Kt loop

                    // Read 32 untilized tiles and select correct rows to reconstruct single correct tile
                    dfb_intermed1.wait_front(1);
                    CoreLocalMem<uint32_t> local_dst(dfb_intermed2_addr);
                    noc.async_read(
                        local_src,
                        local_dst,
                        bfloat16_row_bytes,
                        {.noc_x = local_noc_x, .noc_y = local_noc_y, .addr = dfb_intermed1_addr},
                        {});
                    noc.async_read_barrier();
                    dfb_intermed1.pop_front(1);
                    dfb_intermed1_addr += bfloat16_row_bytes;
                    dfb_intermed2_addr += bfloat16_row_bytes;

                    itileB += in1_KtNt_skip;  // different depending on transpose_hw
                }  // 32 tiles loop
                dfb_intermed2.push_back(1);

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
