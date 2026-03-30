// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/comp.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t packed_scalar = get_arg_val<uint32_t>(1);
    const auto lambd = reinterpret_cast<const float*>(&packed_scalar);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    constexpr auto cb_tmp0 = tt::CBIndex::c_1;
    init_sfpu(cb_input, cb_output);

    // a·1(a+λ<0)+a·1(a−λ>0)
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_input, 1);
        cb_reserve_back(cb_output, 1);
        cb_reserve_back(cb_tmp0, 1);
        tile_regs_acquire();

        fill_tile(0, *lambd);

#ifdef INP_FLOAT32
        copy_tile_to_dst_init_short(cb_input);
        copy_tile(cb_input, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        ltz_tile(0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
#endif
#ifdef INP_FLOAT
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_input);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_input, 0, 0);
        ltz_tile(0);
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_input);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_input, 0, 0);
#endif

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_tmp0);
        tile_regs_release();

        cb_push_back(cb_tmp0, 1);
        cb_wait_front(cb_tmp0, 1);
        tile_regs_acquire();

#ifdef INP_FLOAT32
        fill_tile(1, *lambd);

        copy_tile_to_dst_init_short(cb_input);
        copy_tile(cb_input, 0, 0);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        gtz_tile(0);
        copy_tile_to_dst_init_short(cb_input);
        copy_tile(cb_input, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        copy_tile_to_dst_init_short(cb_tmp0);
        copy_tile(cb_tmp0, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
#endif
#ifdef INP_FLOAT
        fill_tile(0, *lambd);
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_input);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_input, 0, 0);
        gtz_tile(0);
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_input);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_input, 0, 0);

        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_tmp0);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_tmp0, 0, 0);
#endif

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_output);
        tile_regs_release();

        cb_pop_front(cb_input, 1);
        cb_pop_front(cb_tmp0, 1);
        cb_push_back(cb_output, 1);
    }
}
