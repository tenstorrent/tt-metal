// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/dataflow/dataflow_buffer.h"

#define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

void kernel_main() {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    DataflowBuffer dfb_in0(tt::CBIndex::c_0);
    DataflowBuffer dfb_in1(tt::CBIndex::c_1);
    DataflowBuffer dfb_out0(tt::CBIndex::c_2);
#ifdef SFPU_OP_INIT_PRE_IN0_0
    DataflowBuffer dfb_inp0(tt::CBIndex::c_3);
#else
    DataflowBuffer dfb_inp0 = dfb_in0;
#endif
#ifdef SFPU_OP_INIT_PRE_IN1_0
    DataflowBuffer dfb_inp1(tt::CBIndex::c_4);
#else
    DataflowBuffer dfb_inp1 = dfb_in1;
#endif

    binary_op_init_common(dfb_inp0.get_id(), dfb_inp1.get_id(), dfb_out0.get_id());

#if not PRE_SCALE
    binary_tiles_init<false, ELTWISE_OP_TYPE>(dfb_inp0.get_id(), dfb_inp1.get_id());
#endif

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluConfig::zero())));
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
#if PRE_SCALE
        copy_tile_to_dst_init_short(dfb_in0.get_id());  // need to copy from CB to DST to be able to run sfpu math
#endif

#ifdef SFPU_OP_INIT_PRE_IN0_0
        reconfig_data_format_srca(dfb_inp0.get_id(), dfb_in0.get_id());
        pack_reconfig_data_format(dfb_out0.get_id(), dfb_inp0.get_id());
        dfb_in0.wait_front(per_core_block_size);
        dfb_inp0.reserve_back(per_core_block_size);

        tile_regs_acquire();
        SFPU_OP_INIT_PRE_IN0_0
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(dfb_in0.get_id(), i, i);  // copy from c_in[0] to DST[0]
            SFPU_OP_FUNC_PRE_IN0_0
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, dfb_inp0.get_id());  // DST[0]->cb
        }
        tile_regs_release();

        dfb_in0.pop_front(per_core_block_size);
        dfb_inp0.push_back(per_core_block_size);
#ifndef SFPU_OP_INIT_PRE_IN1_0
        reconfig_data_format_srca(dfb_in0.get_id(), dfb_inp0.get_id());
        pack_reconfig_data_format(dfb_inp0.get_id(), dfb_out0.get_id());
#endif
#endif

#ifdef SFPU_OP_INIT_PRE_IN1_0
#ifndef SFPU_OP_INIT_PRE_IN0_0
        reconfig_data_format_srca(dfb_inp0.get_id(), dfb_in1.get_id());
        pack_reconfig_data_format(dfb_out0.get_id(), dfb_inp1.get_id());
#else
        reconfig_data_format_srca(dfb_in0.get_id(), dfb_in1.get_id());
        pack_reconfig_data_format(dfb_inp0.get_id(), dfb_inp1.get_id());
#endif
        dfb_in1.wait_front(per_core_block_size);
        dfb_inp1.reserve_back(per_core_block_size);

        tile_regs_acquire();
        SFPU_OP_INIT_PRE_IN1_0
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(dfb_in1.get_id(), i, i);  // copy from c_in[0] to DST[0]
            SFPU_OP_FUNC_PRE_IN1_0
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, dfb_inp1.get_id());  // DST[0]->cb
        }
        tile_regs_release();

        dfb_in1.pop_front(per_core_block_size);
        dfb_inp1.push_back(per_core_block_size);
        reconfig_data_format_srca(dfb_in1.get_id(), dfb_inp0.get_id());
        pack_reconfig_data_format(dfb_inp1.get_id(), dfb_out0.get_id());
#endif

        dfb_inp0.wait_front(per_core_block_size);
        dfb_inp1.wait_front(per_core_block_size);
        dfb_out0.reserve_back(per_core_block_size);

#if PRE_SCALE
        binary_tiles_init<true, ELTWISE_OP_TYPE>(dfb_inp0.get_id(), dfb_inp1.get_id());
#endif

        tile_regs_acquire();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            ELTWISE_OP(dfb_inp0.get_id(), dfb_inp1.get_id(), i, i, i);

#ifdef SFPU_OP_INIT_0
            SFPU_OP_INIT_0
            SFPU_OP_FUNC_0
#endif

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, dfb_out0.get_id());
        }
        tile_regs_release();

        dfb_inp0.pop_front(per_core_block_size);
        dfb_inp1.pop_front(per_core_block_size);
        dfb_out0.push_back(per_core_block_size);
    }
}
